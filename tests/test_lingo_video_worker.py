from __future__ import annotations

# pyright: reportMissingImports=false

from pathlib import Path

import pytest

from lingo import Corpus, Reference, phrase, echo, Moderator
from lingo.storage import EmptyCorpus

import lingo_video_worker as worker


@worker.lang.conversation
def test_kokoro_tts_conversation(mod: Moderator, tmp_path: Path):
    voice_dir = worker._repo_dir() / "weights" / "Kokoro-82M" / "voices"
    if not voice_dir.exists():
        pytest.skip(f"Kokoro voices directory missing: {voice_dir}")

    voice_candidates = sorted(voice_dir.glob("*.pt"))
    if not voice_candidates:
        pytest.skip(f"No Kokoro voice weights found in: {voice_dir}")

    try:
        import torch  # noqa: F401
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        from kokoro import KPipeline  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Kokoro runtime dependencies unavailable: {exc}")

    target = Reference.from_path(tmp_path / "audio.wav", content_type="audio/wav")
    speech_text = "Hello, this is a test of Kokoro TTS in the MultiTalk worker."
    voice_id = voice_candidates[0].stem

    job = mod.say(phrase("kokoro_tts")(speech_text, voice_id, target, worker.KokoroSettings()))

    out = job.result(block=True)
    status = job.status()

    assert status.succeeded(), status.message or "kokoro_tts task failed"
    assert out is not None
    assert out.content_type == "audio/wav"
    assert (tmp_path / "audio.wav").exists()


@worker.lang.conversation
def test_multitalk_pipeline_conversation(mod: Moderator, tmp_path: Path):
    target = Reference.from_path(tmp_path / "video.mp4", content_type="video/mp4")
    avatar = Corpus.from_file(worker._repo_dir() / "assets" / "logo.png", content_type="image/png")

    audio = echo("kokoro_tts", reply=EmptyCorpus(content_type="audio/wav"))(
        "Hello, this is a mocked Kokoro task.",
        "bf_isabella",
        None,
        None,
    )

    video = echo("multitalk.video.render", reply=EmptyCorpus(content_type="video/mp4"))(
        audio,
        avatar,
        None,
        target,
        None,
    )

    job = mod.say(video)

    out = job.result(block=True)
    status = job.status()

    assert status.succeeded(), status.message or "multitalk.video.render task failed"
    assert out is not None
    assert out.content_type == "video/mp4"
