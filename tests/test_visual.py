"""Visual highlight mode: clipping footage that has no speech.

Unit tests cover the rubric maths on synthetic scores. The ``slow`` test renders
a real video with ffmpeg — empty water, then a busy sharp stretch, then a soft
one — and checks the picks land on the busy stretch and skip the water.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from autoclip.pipeline import visual
from autoclip.pipeline.visual import (
    FrameSample,
    PartScores,
    VisualConfig,
    choose_mode,
    find_windows,
    score_frame,
    usable_fraction,
)

# --------------------------------------------------------------------------
# Mode choice
# --------------------------------------------------------------------------


def test_auto_mode_picks_visual_for_silent_footage() -> None:
    mode, _ = choose_mode("auto", words_per_minute=3, min_speech_wpm=40, has_video=True)
    assert mode == "visual"


def test_auto_mode_keeps_speech_for_narration() -> None:
    mode, _ = choose_mode("auto", words_per_minute=140, min_speech_wpm=40, has_video=True)
    assert mode == "speech"


def test_explicit_mode_wins_over_word_rate() -> None:
    assert choose_mode("visual", words_per_minute=200, min_speech_wpm=40, has_video=True)[0] == (
        "visual"
    )
    assert choose_mode("speech", words_per_minute=0, min_speech_wpm=40, has_video=True)[0] == (
        "speech"
    )


def test_silent_video_always_goes_visual() -> None:
    mode, why = choose_mode(
        "speech", words_per_minute=0, min_speech_wpm=40, has_video=True, has_audio=False
    )
    assert mode == "visual" and "no audio" in why


def test_audio_only_never_goes_visual() -> None:
    assert choose_mode("visual", words_per_minute=0, min_speech_wpm=40, has_video=False)[0] == (
        "speech"
    )


# --------------------------------------------------------------------------
# Frame measures
# --------------------------------------------------------------------------


def _frame(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    return rgb, gray.astype(np.float32)


def test_flat_water_has_no_detail_and_no_motion() -> None:
    flat = np.full((90, 160), 60.0)
    rgb, gray = _frame(flat)
    sample = score_frame(0.5, rgb, gray, gray.copy())
    assert sample.detail < 0.01
    assert sample.subject_motion < 0.001


def test_camera_pan_is_not_counted_as_subject_motion() -> None:
    rng = np.random.default_rng(1)
    scene = rng.uniform(0, 255, (90, 200))
    before = scene[:, 20:180]
    after = scene[:, 26:186]  # the whole frame slid 6 px: a pan, nothing moved
    rgb, gray = _frame(after)
    sample = score_frame(0.5, rgb, gray, before.astype(np.float32))
    assert abs(sample.camera_dx * 160) == pytest.approx(6, abs=1.0)
    raw_diff = float(np.abs(after - before).mean() / 255.0)
    assert sample.subject_motion < 0.2 * raw_diff


def test_subject_position_follows_a_moving_object() -> None:
    background = np.full((90, 160), 40.0)
    before = background.copy()
    after = background.copy()
    before[40:55, 110:125] = 220.0
    after[42:57, 116:131] = 220.0
    rgb, gray = _frame(after)
    sample = score_frame(0.5, rgb, gray, before.astype(np.float32))
    assert sample.subject_x > 0.6


# --------------------------------------------------------------------------
# Windows and the rubric
# --------------------------------------------------------------------------


def _parts(duration: float, good: tuple[float, float], fps: float = 2.0) -> PartScores:
    times = np.arange(0, duration, 1 / fps, dtype=np.float32)
    inside = (times >= good[0]) & (times < good[1])
    high = np.where(inside, 0.9, 0.15).astype(np.float32)
    return PartScores(
        times=times, picture=high, subject=high, story=high, shake=np.zeros_like(times)
    )


def test_best_window_lands_on_the_good_stretch() -> None:
    parts = _parts(120, good=(60, 85))
    windows = find_windows(parts, duration_s=120, config=VisualConfig(8, 30, 3))
    best = windows[0]
    assert best.start_s >= 58 and best.end_s <= 87
    assert best.band in ("Hero", "Cutaway")


def test_clips_never_repeat_footage() -> None:
    parts = _parts(200, good=(20, 180))
    windows = find_windows(parts, duration_s=200, config=VisualConfig(8, 20, 6))
    assert len(windows) == 6
    for i, a in enumerate(windows):
        for b in windows[i + 1 :]:
            overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
            assert overlap <= 1.0 + 1e-6


def test_window_lengths_respect_settings() -> None:
    parts = _parts(300, good=(0, 300))
    for w in find_windows(parts, duration_s=300, config=VisualConfig(10, 24, 8)):
        assert 10 - 1e-6 <= w.duration_s <= 24 + 1e-6


def test_edges_snap_to_nearby_cuts() -> None:
    parts = _parts(100, good=(40, 70))
    shots = [(0.0, 40.6), (40.6, 69.4), (69.4, 100.0)]
    best = find_windows(parts, duration_s=100, config=VisualConfig(8, 30, 1), shots=shots)[0]
    assert best.start_s == pytest.approx(40.6)
    assert best.end_s == pytest.approx(69.4)


def test_fragments_score_low_on_usable() -> None:
    fragments = [(float(t), float(t + 2)) for t in range(0, 30, 2)]
    long_shot = [(0.0, 30.0)]
    assert usable_fraction(0, 30, fragments) == 0.0
    assert usable_fraction(0, 30, long_shot) == 1.0


def test_empty_water_is_never_picked_over_a_subject() -> None:
    times = np.arange(0, 60, 0.5, dtype=np.float32)
    first_half = times < 30
    # First half: steady and sharp but empty. Second half: a real subject.
    parts = PartScores(
        times=times,
        picture=np.full_like(times, 0.95),
        subject=np.where(first_half, 0.05, 0.7).astype(np.float32),
        story=np.where(first_half, 0.9, 0.5).astype(np.float32),
        shake=np.zeros_like(times),
    )
    best = find_windows(parts, duration_s=60, config=VisualConfig(8, 20, 1))[0]
    assert best.start_s >= 29


def test_describe_reports_band_and_parts() -> None:
    parts = _parts(40, good=(0, 40))
    window = find_windows(parts, duration_s=40, config=VisualConfig(8, 20, 1))[0]
    text = visual.describe(window)
    assert "/100" in text and "Picture" in text and "Story" in text


# --------------------------------------------------------------------------
# Real render
# --------------------------------------------------------------------------


def _render_footage(path: Path) -> None:
    """20 s of flat open water, 20 s of busy sharp detail, 15 s of blur."""
    water = "color=c=0x123a5a:s=640x360:r=25:d=20,noise=alls=4:allf=t"
    busy = "testsrc2=s=640x360:r=25:d=20"
    soft = "testsrc2=s=640x360:r=25:d=15,boxblur=12:4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            water,
            "-f",
            "lavfi",
            "-i",
            busy,
            "-f",
            "lavfi",
            "-i",
            soft,
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            str(path),
        ],
        check=True,
    )


@pytest.mark.slow
def test_detect_on_rendered_footage(tmp_path: Path) -> None:
    video = tmp_path / "footage.mp4"
    _render_footage(video)

    samples_path = tmp_path / "samples.json"
    clips = visual.detect(
        video,
        duration_s=55.0,
        config=VisualConfig(min_duration_s=8, max_duration_s=16, max_clips=3, min_score=0),
        job_id="job",
        samples_path=samples_path,
    )

    assert clips, "expected at least one clip"
    best = clips[0]
    midpoint = (best.start_s + best.end_s) / 2
    assert 20 <= midpoint <= 40, f"best clip {best.start_s}-{best.end_s} missed the busy stretch"
    assert best.start_s >= 18, "the best clip should not open on empty water"
    assert samples_path.exists()

    # The cache round-trips, so a retry doesn't decode the video again.
    cached = visual.load_samples(samples_path)
    assert isinstance(cached[0], FrameSample)

    crop = visual.subject_crop_path(
        cached, video=video, start_s=best.start_s, end_s=best.end_s, aspect_w=9, aspect_h=16
    )
    assert crop.segments
    assert crop.segments[0].start_s == pytest.approx(0.0, abs=0.05)
    assert crop.segments[-1].end_s == pytest.approx(best.end_s - best.start_s, abs=0.1)
    for segment in crop.segments:
        x = segment.keyframes[0].x
        assert 0 <= x <= crop.source_width - segment.width


@pytest.mark.slow
def test_empty_water_alone_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "water.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x123a5a:s=640x360:r=25:d=20,noise=alls=4:allf=t",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    with pytest.raises(visual.VisualHighlightError, match="reject line"):
        visual.detect(video, duration_s=20.0, config=VisualConfig(8, 16, 3), job_id="job")


# --------------------------------------------------------------------------
# Whole pipeline
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_runs_visual_mode_end_to_end(tmp_path: Path, initialised_db: int) -> None:
    """Silent footage through every stage: no Whisper, no LLM, muted 9:16 clips."""
    import asyncio
    import json

    from autoclip.config import Settings
    from autoclip.db import store
    from autoclip.db.models import Job, new_id
    from autoclip.pipeline import ingest
    from autoclip.pipeline.ffmpeg import probe
    from autoclip.pipeline.runner import PipelineRunner

    video = tmp_path / "footage.mp4"
    _render_footage(video)

    source = store.create_source(ingest.ingest_file(video, title="Footage"))
    settings = Settings()
    settings.clips.highlight_mode = "visual"
    settings.clips.visual_min_duration_s = 8
    settings.clips.visual_max_duration_s = 12
    settings.clips.max_clips = 2
    settings.clips.visual_min_score = 0

    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="none", settings={}))
    runner = PipelineRunner(job, source, settings=settings)
    clips = asyncio.run(runner.run())

    assert clips
    mode = json.loads(runner.workspace.highlight_mode.read_text(encoding="utf-8"))
    assert mode["mode"] == "visual"
    assert "visual mode" in json.loads(runner.workspace.transcript.read_text())["model"]

    from autoclip import paths

    outputs = sorted((paths.exports_dir() / job.id).glob("*.mp4"))
    assert len(outputs) == len(clips)
    info = probe(outputs[0])
    assert info.has_video and not info.has_audio
    assert info.height and info.width and info.height > info.width
