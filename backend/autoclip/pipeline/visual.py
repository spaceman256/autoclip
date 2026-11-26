"""Visual highlight detection — for footage with little or no speech.

The speech pipeline picks clips from what people say. Wildlife, underwater,
drone and B-roll footage has nothing to say, so the transcript is empty (or a
few hallucinated words) and the language model has nothing to choose from.

This module chooses clips from what the camera sees instead, scored out of 100
on the Daily Catch footage rubric:

- Picture /25 — focus, light, and a steady camera.
- Subject /30 — a clear, detailed subject rather than empty water.
- Usable  /20 — made of long clean shots (8 s+), not 2-3 s fragments.
- Story   /25 — the animal itself moving, with camera movement taken out.

Bands: 75+ hero, 60-74 cutaway, 45-59 filler, under 45 rejected.

How: sample frames at a low rate straight out of ffmpeg, score each sample,
slide windows over the scores, snap their edges to shot cuts, and keep the best
non-overlapping set that clears the reject line.

What it cannot judge — species, an aquarium look, logos or captions, people on
screen — stays with a human review. Treat the score as a first sort, not a
sign-off.

No model download and no LLM call: it runs on the CPU in a minute or two for an
hour of footage, and works fully offline.

The per-sample subject position is saved too, so the reframe stage can hold a
9:16 crop on whatever is moving instead of centre-cropping it out of shot.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..db.models import Clip, new_id
from . import ffmpeg

log = logging.getLogger(__name__)

#: Frames sampled per second. Two is enough to see movement and judge a clip,
#: and keeps an hour of footage to ~7,200 small frames.
SAMPLE_FPS = 2.0

#: Width frames are downscaled to before scoring. Scoring cares about
#: structure, not detail; small frames make it fast enough for the CPU.
SAMPLE_WIDTH = 320

# -- Scoring follows the Daily Catch footage rubric --------------------------
# Picture 25 (focus, light, steadiness), Subject 30 (a clear subject, not empty
# water), Usable 20 (built from long clean shots), Story 25 (the animal moving).
# Bands: 75+ hero, 60-74 cutaway, 45-59 filler, under 45 reject.
# Thresholds are measured on SAMPLE_WIDTH-wide frames.

POINTS_PICTURE = 25.0
POINTS_SUBJECT = 30.0
POINTS_USABLE = 20.0
POINTS_STORY = 25.0

#: Laplacian variance from soft to crisp. Kept low at the soft end so murky
#: water isn't marked down for the murk alone.
FOCUS_SOFT, FOCUS_SHARP = 15.0, 400.0
#: Mean luma outside this range is too dark or blown out.
DARK_LUMA = 22.0
BRIGHT_LUMA = 235.0
#: Change in camera movement between samples, as a fraction of frame width.
#: A smooth pan barely changes; a shaky hand-held shot jumps about.
SHAKE_OK, SHAKE_BAD = 0.004, 0.03
#: Share of the frame holding strong detail. Open water sits near 0.01.
DETAIL_EDGE = 12.0
DETAIL_EMPTY, DETAIL_FULL = 0.02, 0.18
#: Colourfulness (Hasler & Susstrunk) from washed-out to vivid.
COLOUR_DULL, COLOUR_VIVID = 8.0, 45.0
#: Subject movement once camera movement is taken out.
MOTION_STILL, MOTION_ACTIVE = 0.003, 0.025
#: Shots shorter than SHOT_SHORT_S count as fragments; SHOT_LONG_S and up fade in well.
SHOT_SHORT_S, SHOT_LONG_S = 3.0, 8.0
#: A clip whose subject score averages below this is mostly empty water.
EMPTY_WATER = 0.25

BANDS = ((75.0, "Hero"), (60.0, "Cutaway"), (45.0, "Filler"), (0.0, "Reject"))

#: Step between candidate window starts, and between candidate lengths.
START_STEP_S = 1.0
LENGTH_STEP_S = 2.0

#: Window edges move to a shot cut this close by, so clips begin and end on cuts.
SNAP_S = 1.5

#: Clips may share at most this much footage, so a set of Shorts never
#: repeats the same moment.
MAX_OVERLAP_S = 1.0


def choose_mode(
    configured: str,
    *,
    words_per_minute: float,
    min_speech_wpm: float,
    has_video: bool,
    has_audio: bool = True,
) -> tuple[str, str]:
    """Return ``(mode, reason)``: whether a job clips by speech or by picture."""
    if not has_video:
        return "speech", "audio-only source"
    if not has_audio:
        return "visual", "no audio track"
    if configured in ("speech", "visual"):
        return configured, "set in settings"
    if words_per_minute < min_speech_wpm:
        return "visual", f"only {words_per_minute:.0f} spoken words per minute"
    return "speech", f"{words_per_minute:.0f} spoken words per minute"


class VisualHighlightError(RuntimeError):
    """Visual detection produced nothing usable."""


@dataclass
class FrameSample:
    """Raw measures for one sampled frame."""

    t: float
    #: Variance of the Laplacian: higher is sharper.
    sharpness: float
    #: Mean brightness, 0-255.
    luma: float
    #: Colourfulness (Hasler & Susstrunk).
    colour: float
    #: Share of the frame with strong detail. Empty water is near zero.
    detail: float
    #: Camera movement since the previous sample, as a fraction of frame width.
    camera_dx: float
    camera_dy: float
    #: Movement left once the camera's own movement is taken out, 0-1.
    subject_motion: float
    #: Subject position as fractions of frame width and height.
    subject_x: float
    subject_y: float


@dataclass
class VisualConfig:
    min_duration_s: float = 8.0
    max_duration_s: float = 30.0
    max_clips: int = 10
    #: Windows scoring below this (out of 100) are rejected.
    min_score: float = 45.0


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_video(
    video: Path,
    *,
    duration_s: float | None = None,
    fps: float = SAMPLE_FPS,
    width: int = SAMPLE_WIDTH,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[FrameSample]:
    """Decode the video at a low rate and measure every sampled frame."""
    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        raise VisualHighlightError(f"{video.name} has no video stream to score.")

    duration = duration_s or info.duration_s or 0.0
    out_w = width - (width % 2)
    out_h = max(2, round(out_w * info.height / info.width))
    out_h -= out_h % 2
    frame_bytes = out_w * out_h * 3

    command = [
        ffmpeg.ffmpeg_path(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-an",
        "-sn",
        "-vf",
        f"fps={fps},scale={out_w}:{out_h}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    samples: list[FrameSample] = []
    previous: np.ndarray | None = None
    index = 0

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        while True:
            if cancelled and cancelled():
                proc.kill()
                raise RuntimeError("Visual scoring cancelled.")
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3)
            gray = _to_gray(rgb)
            samples.append(score_frame(index / fps, rgb, gray, previous))
            previous = gray
            index += 1
            if on_progress and duration > 0 and index % 10 == 0:
                on_progress(min(1.0, (index / fps) / duration))
    finally:
        if proc.poll() is None:
            proc.wait()

    if not samples:
        raise VisualHighlightError(f"ffmpeg could not decode any frames from {video.name}.")
    if on_progress:
        on_progress(1.0)
    return samples


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def _camera_shift(previous: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Whole-frame shift between two samples, in pixels, by phase correlation."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 arrives with scenedetect
        return 0.0, 0.0
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(previous, gray, window)
    return float(dx), float(dy)


def _compensated_diff(previous: np.ndarray, gray: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Absolute difference after undoing the camera's shift; borders zeroed."""
    sx, sy = int(round(dx)), int(round(dy))
    aligned = np.roll(previous, shift=(sy, sx), axis=(0, 1))
    diff = np.abs(gray - aligned)
    h, w = gray.shape
    my = min(h // 2, abs(sy) + 2)
    mx = min(w // 2, abs(sx) + 2)
    diff[:my, :] = 0
    diff[h - my :, :] = 0
    diff[:, :mx] = 0
    diff[:, w - mx :] = 0
    return diff


def score_frame(
    t: float, rgb: np.ndarray, gray: np.ndarray, previous: np.ndarray | None
) -> FrameSample:
    """Raw measures for one frame."""
    h, w = gray.shape

    if previous is not None and previous.shape == gray.shape:
        dx, dy = _camera_shift(previous, gray)
        compensated = _compensated_diff(previous, gray, dx, dy)
        subject_motion = float(compensated.mean() / 255.0)
        diff: np.ndarray | None = compensated
    else:
        dx = dy = 0.0
        diff = None
        subject_motion = 0.0

    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    edges = np.abs(lap)

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colour = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    subject_x, subject_y = _subject_position(gray, diff, edges)
    return FrameSample(
        t=round(t, 3),
        sharpness=float(lap.var()),
        luma=float(gray.mean()),
        colour=colour,
        detail=float((edges > DETAIL_EDGE).mean()),
        camera_dx=dx / w,
        camera_dy=dy / h,
        subject_motion=subject_motion,
        subject_x=subject_x,
        subject_y=subject_y,
    )


def _subject_position(
    gray: np.ndarray, diff: np.ndarray | None, edges: np.ndarray
) -> tuple[float, float]:
    """Where the eye is drawn: moving, detailed areas, weighted to the centre.

    Movement says where the animal is; edge detail says what's in focus. Both
    are noisy alone (drifting particles move, coral is all edges), so they are
    combined, thresholded to the strongest areas, and pulled gently toward the
    centre so a stray speck at the border can't drag the crop.
    """
    h, w = gray.shape
    detail = np.zeros_like(gray)
    detail[1:-1, 1:-1] = edges
    weight = detail / (detail.max() + 1e-6)
    if diff is not None:
        weight = weight + 2.0 * (diff / (diff.max() + 1e-6))

    cutoff = np.quantile(weight, 0.9)
    weight = np.where(weight >= cutoff, weight, 0.0)

    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    centre_pull = np.exp(-((xs[None, :] - 0.5) ** 2) / 0.18) * np.exp(
        -((ys[:, None] - 0.5) ** 2) / 0.18
    )
    weight = weight * centre_pull

    total = float(weight.sum())
    if total <= 0:
        return 0.5, 0.5
    return float((weight * xs[None, :]).sum() / total), float((weight * ys[:, None]).sum() / total)


# --------------------------------------------------------------------------
# Rubric scoring
# --------------------------------------------------------------------------


def _ramp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """0 at or below ``low``, 1 at or above ``high``, linear between."""
    return np.clip((values - low) / (high - low), 0.0, 1.0)


@dataclass
class PartScores:
    """Per-sample rubric parts, each 0..1. Usable is judged per window."""

    times: np.ndarray
    picture: np.ndarray
    subject: np.ndarray
    story: np.ndarray
    shake: np.ndarray


def score_samples(samples: list[FrameSample]) -> PartScores:
    times = np.array([s.t for s in samples], dtype=np.float32)
    sharp = np.array([s.sharpness for s in samples], dtype=np.float32)
    luma = np.array([s.luma for s in samples], dtype=np.float32)
    colour = np.array([s.colour for s in samples], dtype=np.float32)
    detail = np.array([s.detail for s in samples], dtype=np.float32)
    motion = np.array([s.subject_motion for s in samples], dtype=np.float32)
    cam = np.array([[s.camera_dx, s.camera_dy] for s in samples], dtype=np.float32)

    focus = _ramp(np.log10(np.maximum(sharp, 1e-3)), np.log10(FOCUS_SOFT), np.log10(FOCUS_SHARP))
    light = np.where((luma < DARK_LUMA) | (luma > BRIGHT_LUMA), 0.2, 1.0)

    # Shake: how much the camera's movement changes between samples. A pan
    # moves steadily and scores as steady; a wobbling hand does not.
    shake = np.zeros(len(samples), dtype=np.float32)
    if len(samples) > 2:
        shake[2:] = np.linalg.norm(cam[2:] - cam[1:-1], axis=1)
        shake[:2] = shake[2]
    steady = 1.0 - _ramp(shake, SHAKE_OK, SHAKE_BAD)

    picture = 0.5 * focus + 0.25 * light + 0.25 * steady
    subject = 0.75 * _ramp(detail, DETAIL_EMPTY, DETAIL_FULL) + 0.25 * _ramp(
        colour, COLOUR_DULL, COLOUR_VIVID
    )
    # Murk or darkness can hide a subject that is there; a frame you can't see
    # into can't score as a clear subject.
    subject = subject * light

    story = _ramp(motion, MOTION_STILL, MOTION_ACTIVE)
    if len(story) > 1:
        story[0] = story[1]  # the first sample has nothing to compare against

    return PartScores(times=times, picture=picture, subject=subject, story=story, shake=shake)


def usable_fraction(start: float, end: float, shots: list[tuple[float, float]]) -> float:
    """0..1: how much of a window sits in shots long enough to fade into."""
    length = end - start
    if length <= 0:
        return 0.0
    if not shots:
        shots = [(start, end)]
    total = 0.0
    for shot_start, shot_end in shots:
        overlap = min(end, shot_end) - max(start, shot_start)
        if overlap <= 0:
            continue
        shot_len = shot_end - shot_start
        quality = float(_ramp(np.array([shot_len]), SHOT_SHORT_S, SHOT_LONG_S)[0])
        total += overlap * quality
    return min(1.0, total / length)


@dataclass
class Window:
    start_s: float
    end_s: float
    picture: float
    subject: float
    usable: float
    story: float
    shaky: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def score(self) -> float:
        return self.picture + self.subject + self.usable + self.story

    @property
    def band(self) -> str:
        # Banded on the whole number shown, so "75" never reads as a cutaway.
        shown = round(self.score)
        return next(label for floor, label in BANDS if shown >= floor)

    @property
    def empty_water(self) -> bool:
        return self.subject < EMPTY_WATER * POINTS_SUBJECT


def window_stats(
    parts: PartScores, start: float, end: float, shots: list[tuple[float, float]]
) -> Window | None:
    mask = (parts.times >= start) & (parts.times < end)
    if not mask.any():
        return None
    return Window(
        start_s=round(float(start), 3),
        end_s=round(float(end), 3),
        picture=POINTS_PICTURE * float(parts.picture[mask].mean()),
        subject=POINTS_SUBJECT * float(parts.subject[mask].mean()),
        usable=POINTS_USABLE * usable_fraction(start, end, shots),
        story=POINTS_STORY * float(parts.story[mask].mean()),
        shaky=float(np.median(parts.shake[mask])) > SHAKE_BAD,
    )


def find_windows(
    parts: PartScores,
    *,
    duration_s: float,
    config: VisualConfig,
    shots: list[tuple[float, float]] | None = None,
) -> list[Window]:
    """Score every candidate window; return the best non-overlapping ones.

    Includes windows under the reject line, best first, so the caller can say
    how close the footage came. Filter on ``min_score`` before exporting.
    """
    if len(parts.times) == 0:
        return []

    shots = shots or [(0.0, duration_s)]
    cuts = sorted({s for s, _ in shots[1:]})
    min_len = max(1.0, config.min_duration_s)
    max_len = max(min_len, config.max_duration_s)

    candidates: list[Window] = []
    if duration_s <= min_len:
        whole = window_stats(parts, 0.0, duration_s, shots)
        candidates = [whole] if whole else []
    else:
        lengths = list(np.arange(min_len, max_len + 1e-6, LENGTH_STEP_S))
        starts = list(np.arange(0.0, max(0.0, duration_s - min_len) + 1e-6, START_STEP_S))
        for start in starts:
            for length in lengths:
                end = start + length
                if end > duration_s + 1e-6:
                    break
                s, e = _snap(float(start), float(end), cuts, duration_s, min_len, max_len)
                window = window_stats(parts, s, e, shots)
                if window is not None:
                    candidates.append(window)

    # Mostly-empty water fails whatever else it scores.
    candidates = [w for w in candidates if not w.empty_water] or candidates
    cut_set = set(cuts) | {0.0, round(duration_s, 3)}

    def rank(w: Window) -> tuple[bool, float, int, float]:
        # Score first (rounded, so float noise can't decide); then clips that
        # start and end on cuts; then the longer of two equally good clips.
        on_cuts = int(w.start_s in cut_set) + int(w.end_s in cut_set)
        return (not w.empty_water, round(w.score, 1), on_cuts, w.duration_s)

    candidates.sort(key=rank, reverse=True)

    kept: list[Window] = []
    for window in candidates:
        if any(_overlap_s(window, other) > MAX_OVERLAP_S for other in kept):
            continue
        kept.append(window)
        if len(kept) >= config.max_clips:
            break
    return kept


def _snap(
    start: float, end: float, cuts: list[float], duration: float, min_len: float, max_len: float
) -> tuple[float, float]:
    """Move edges onto nearby shot cuts, if the result still fits the length range."""

    def nearest(t: float) -> float:
        best, best_gap = t, SNAP_S
        for cut in cuts:
            gap = abs(cut - t)
            if gap <= best_gap:
                best, best_gap = cut, gap
        return best

    s = max(0.0, nearest(start))
    e = min(duration, nearest(end))
    if min_len - 1e-6 <= e - s <= max_len + 1e-6:
        return s, e
    return start, min(end, duration)


def _overlap_s(a: Window, b: Window) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def describe(window: Window) -> str:
    """The review screen's "why": total, band, the four parts, and any flags."""
    line = (
        f"{window.band} {window.score:.0f}/100: "
        f"Picture {window.picture:.0f}/25, Subject {window.subject:.0f}/30, "
        f"Usable {window.usable:.0f}/20, Story {window.story:.0f}/25."
    )
    flags = []
    if window.shaky:
        flags.append("shaky camera")
    if window.empty_water:
        flags.append("mostly empty water")
    if window.usable < 0.4 * POINTS_USABLE:
        flags.append("short fragments")
    if flags:
        line += " Watch for: " + ", ".join(flags) + "."
    line += " Check species, logos and people by eye."
    return line


def _timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def windows_to_clips(windows: list[Window], *, job_id: str) -> list[Clip]:
    clips: list[Clip] = []
    for rank, window in enumerate(sorted(windows, key=lambda w: w.score, reverse=True), start=1):
        clips.append(
            Clip(
                id=new_id(),
                job_id=job_id,
                start_s=window.start_s,
                end_s=window.end_s,
                rank=rank,
                # Visual clips carry no words; the job's transcript is empty.
                start_word=0,
                end_word=0,
                title=f"{window.band} {window.score:.0f} · {_timestamp(window.start_s)}",
                hook="",
                score=int(round(window.score)),
                reason=describe(window),
            )
        )
    return clips


def detect(
    video: Path,
    *,
    duration_s: float,
    config: VisualConfig,
    job_id: str,
    samples_path: Path | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Clip]:
    """Score the video and return ranked clips that clear the reject line.

    Samples are cached at ``samples_path`` so a retry, and the reframe stage,
    reuse them instead of decoding the video again.
    """
    if samples_path is not None and samples_path.exists():
        samples = load_samples(samples_path)
    else:

        def sampling_progress(fraction: float) -> None:
            if on_progress:
                on_progress(0.85 * fraction)

        samples = sample_video(
            video, duration_s=duration_s, on_progress=sampling_progress, cancelled=cancelled
        )
        if samples_path is not None:
            save_samples(samples, samples_path)

    from .reframe.scenes import detect_shots

    shot_list = detect_shots(video, start_s=0.0, end_s=duration_s)
    shots = [(shot.start_s, shot.end_s) for shot in shot_list]

    windows = find_windows(
        score_samples(samples), duration_s=duration_s, config=config, shots=shots
    )
    if on_progress:
        on_progress(1.0)

    passing = [w for w in windows if w.score >= config.min_score and not w.empty_water]
    if not passing:
        best = f" The best stretch scored {windows[0].score:.0f}/100." if windows else ""
        raise VisualHighlightError(
            f"No clip cleared the reject line ({config.min_score:.0f}/100).{best} "
            "The footage may be too soft, too still or mostly empty water. "
            "You can lower the line in Settings → Clips."
        )

    log.info(
        "Visual detection kept %d of %d clip(s) from %d samples.",
        len(passing),
        len(windows),
        len(samples),
    )
    return windows_to_clips(passing, job_id=job_id)


# --------------------------------------------------------------------------
# Sample cache
# --------------------------------------------------------------------------


def save_samples(samples: list[FrameSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in samples]), encoding="utf-8")
    return path


def load_samples(path: Path) -> list[FrameSample]:
    return [FrameSample(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------
# Subject-following crop
# --------------------------------------------------------------------------


def subject_crop_path(
    samples: list[FrameSample],
    *,
    video: Path,
    start_s: float,
    end_s: float,
    aspect_w: int,
    aspect_h: int,
):
    """A crop that holds on the subject, one steady framing per shot.

    Each shot gets a single, locked crop centred on the median subject position
    within it. Locked framing never looks jittery, and since crops never move
    across a cut, a fish swimming between shots is simply re-framed at the cut.
    """
    from .reframe.croppath import CropKeyframe, CropPath, CropSegment, Strategy, target_crop_size
    from .reframe.scenes import detect_shots

    info = ffmpeg.probe(video)
    if not info.width or not info.height:
        raise VisualHighlightError(f"{video.name} has no video stream to reframe.")
    source_w, source_h = info.width, info.height
    crop_w, crop_h = target_crop_size(source_w, source_h, aspect_w, aspect_h)

    shots = detect_shots(video, start_s=start_s, end_s=end_s)
    segments = []
    for shot in shots:
        absolute_start = start_s + shot.start_s
        absolute_end = start_s + shot.end_s
        inside = [s for s in samples if absolute_start <= s.t < absolute_end]
        if inside:
            cx = float(np.median([s.subject_x for s in inside])) * source_w
            cy = float(np.median([s.subject_y for s in inside])) * source_h
        else:
            cx, cy = source_w / 2, source_h / 2

        x = min(max(0.0, cx - crop_w / 2), source_w - crop_w)
        y = min(max(0.0, cy - crop_h / 2), source_h - crop_h)
        segments.append(
            CropSegment(
                start_s=shot.start_s,
                end_s=shot.end_s,
                width=crop_w,
                height=crop_h,
                keyframes=[CropKeyframe(t=shot.start_s, x=round(x, 1), y=round(y, 1))],
                strategy=Strategy.TRACK,
            )
        )

    if not segments:
        segments.append(
            CropSegment(
                start_s=0.0,
                end_s=end_s - start_s,
                width=crop_w,
                height=crop_h,
                keyframes=[
                    CropKeyframe(t=0.0, x=(source_w - crop_w) / 2, y=(source_h - crop_h) / 2)
                ],
                strategy=Strategy.GENERAL,
            )
        )

    return CropPath(source_width=source_w, source_height=source_h, segments=segments)
