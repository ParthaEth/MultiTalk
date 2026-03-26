"""Lingo worker tasks for MultiTalk orchestration.

This worker intentionally exposes exactly two verbs:
1. Kokoro TTS -> stores generated audio as a local Corpus.
2. MultiTalk render -> consumes TTS Corpus and renders video.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import subprocess
import sys
import pytest
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from lingo import Corpus, Reference, Moderator, scribble, phrase, echo
from lingo.storage import EmptyCorpus
from lingo.celery import CeleryLanguage


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


class KokoroSettings(BaseModel):
    language_code: Optional[str] = None
    repo_id: str = "hexgrad/Kokoro-82M"
    repo_path: str = "weights/Kokoro-82M"
    speed: float = 1.0
    split_pattern: str = r"\n+"
    output_sample_rate: int = Field(default=16000, ge=1)


class MultiTalkSettings(BaseModel):
    video_prompt: str = Field("A professional speaks confidently directly to the camera", min_length=1)

    task: str = "multitalk-14B"
    size: str = "multitalk-480"
    ckpt_dir: str = "./weights/Wan2.1-I2V-14B-480P"
    wav2vec_dir: str = "./weights/chinese-wav2vec2-base"

    audio_mode: str = "localfile"
    mode: str = "streaming"
    sample_steps: int = Field(default=40, ge=1)
    num_persistent_param_in_dit: int = Field(default=0, ge=0)
    use_teacache: bool = True

    output_name: str = "output.mp4"
    extra_args: list[str] = Field(default_factory=list)


def _normalize_output_name(name: str) -> str:
    if name.lower().endswith(".mp4"):
        return name
    return f"{name}.mp4"


def _resolve_voice_path(voice: str, repo_dir: Path) -> str:
    normalized = voice.strip()
    if not normalized:
        raise RuntimeError("voice_id is required")

    if "/" in normalized or "\\" in normalized or normalized.endswith(".pt"):
        resolved = Path(normalized)
        if not resolved.is_absolute():
            resolved = (repo_dir / resolved).resolve()
        return str(resolved)

    candidate = (repo_dir / "weights" / "Kokoro-82M" / "voices" / f"{normalized}.pt").resolve()
    return str(candidate)


def _find_generated_video(requested_output_path: Path) -> Path:
    if requested_output_path.exists():
        return requested_output_path

    parent = requested_output_path.parent
    stem = requested_output_path.stem
    candidates = sorted(
        parent.glob(f"{stem}*.mp4"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise RuntimeError(f"No output video found near: {requested_output_path}")


def _run_command(command: list[str], cwd: Path) -> None:
    process = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(command)}\n"
            f"stdout:\n{process.stdout[-5000:]}\n"
            f"stderr:\n{process.stderr[-5000:]}"
        )


def _build_multitalk_input(video_prompt: str, avatar_file: Path, audio_file: Path, path: Path) -> None:
    data = {
        "prompt": video_prompt,
        "cond_image": str(avatar_file.resolve()),
        "cond_audio": {
            "person1": str(audio_file.resolve()),
        },
        "audio_type": "add",
        "tts_audio": {},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _run_multitalk_render(audio_file: Path, avatar_file: Path, video_prompt: str, param: MultiTalkSettings) -> Path:
    repo_dir = _repo_dir()
    run_dir = scribble()

    output_name = _normalize_output_name(param.output_name)
    output_video_path = (run_dir / output_name).resolve()

    input_json_path = run_dir / "lingo_input.json"
    audio_save_dir = run_dir / "audio"
    audio_save_dir.mkdir(parents=True, exist_ok=True)
    _build_multitalk_input(video_prompt, avatar_file, audio_file, input_json_path)

    command = [
        sys.executable,
        str((repo_dir / "generate_multitalk.py").resolve()),
        "--task",
        param.task,
        "--size",
        param.size,
        "--ckpt_dir",
        param.ckpt_dir,
        "--wav2vec_dir",
        param.wav2vec_dir,
        "--input_json",
        str(input_json_path),
        "--audio_mode",
        param.audio_mode,
        "--sample_steps",
        str(param.sample_steps),
        "--mode",
        param.mode,
        "--num_persistent_param_in_dit",
        str(param.num_persistent_param_in_dit),
        "--audio_save_dir",
        str(audio_save_dir),
        "--save_file",
        str(output_video_path),
    ]
    if param.use_teacache:
        command.append("--use_teacache")
    if param.extra_args:
        command.extend(param.extra_args)

    _run_command(command, cwd=repo_dir)
    return _find_generated_video(output_video_path)



lang = CeleryLanguage(name="multitalk@%h", routing_mode="task", scribble_root="local_data/local/")


# endregion
#####################################################
# region Tasks
#####################################################


@lang.verb
def kokoro_tts(
    speech_text: str,
    voice_id: str,
    target: Optional[Reference[bytes]] = None,
    param: Optional[KokoroSettings] = None,
) -> Corpus[bytes]:
    if param is None:
        param = KokoroSettings()

    run_dir = scribble()

    # Local imports keep worker startup light and only require Kokoro/Torch on TTS nodes.
    from kokoro import KPipeline
    import librosa
    import soundfile as sf
    import torch

    output_path = (run_dir / "tts.wav").resolve()

    repo_dir = _repo_dir()
    voice_path = _resolve_voice_path(voice_id, repo_dir)
    if not Path(voice_path).exists():
        raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

    lang_code = param.language_code or voice_id[0]
    pipeline = KPipeline(lang_code=lang_code, repo_id=param.repo_path)
    try:
        voice_tensor = torch.load(voice_path, weights_only=True)
    except TypeError:
        voice_tensor = torch.load(voice_path)

    chunks = []
    generator = pipeline(
        speech_text,
        voice=voice_tensor,
        speed=param.speed,
        split_pattern=param.split_pattern,
    )
    for _, _, audio in generator:
        if audio is not None:
            chunk = audio if isinstance(audio, torch.Tensor) else torch.tensor(audio)
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio samples")

    merged = torch.cat(chunks, dim=0)
    sf.write(str(output_path), merged.numpy(), 24000)

    # Normalize sample rate for downstream consistency.
    wav_normalized, _ = librosa.load(str(output_path), sr=param.output_sample_rate)
    sf.write(str(output_path), wav_normalized, param.output_sample_rate)

    if target is None:
        return Corpus.from_file(str(output_path), content_type="audio/wav")
    return target.dump_file(output_path, content_type="audio/wav")


@lang.verb("multitalk.video.render")
def render_multitalk(
    audio: Corpus[bytes],
    avatar: Corpus[bytes],
    video_prompt: str = None,
    dest: Optional[Reference] = None,
    param: Optional[MultiTalkSettings] = None,
) -> Corpus[bytes]:
    if param is None:
        param = MultiTalkSettings()
    if video_prompt is None:
        video_prompt = param.video_prompt

    run_dir = scribble()
    audio_path = audio.materialize_to_file(run_dir / "tts_input.wav")
    if not audio_path.exists():
        raise FileNotFoundError(f"TTS audio could not be materialized: {audio_path}")

    avatar_path = avatar.materialize_to_file(run_dir / "avatar_input.png")
    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar could not be materialized: {avatar_path}")

    output_video = _run_multitalk_render(audio_path, avatar_path, video_prompt, param)

    if dest is None:
        video_corpus = Corpus.from_file(str(output_video), content_type="video/mp4")
    else:
        video_corpus = dest.dump_file(output_video, content_type="video/mp4")

    return video_corpus




# endregion
#####################################################
# region Unit Tests
#####################################################


@lang.conversation
def test_kokoro_tts_conversation(mod: Moderator, tmp_path: Path):
    voice_dir = _repo_dir() / "weights" / "Kokoro-82M" / "voices"
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
    # voice_id = voice_candidates[0].stem
    voice_id = 'bf_isabella'

    job = mod.say(phrase("kokoro_tts")(speech_text, voice_id, target, KokoroSettings()))

    out = job.result(block=True)
    status = job.status()

    assert status.succeeded(), status.message or "kokoro_tts task failed"
    assert out is not None
    assert out.content_type == "audio/wav"
    assert (tmp_path / "audio.wav").exists()


@lang.conversation
def test_multitalk_pipeline_conversation(mod: Moderator, tmp_path: Path):
    target = Reference.from_path(tmp_path / "video.mp4", content_type="video/mp4")
    avatar = Corpus.from_file(_repo_dir() / "assets" / "logo.png", content_type="image/png")

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


# endregion
#####################################################