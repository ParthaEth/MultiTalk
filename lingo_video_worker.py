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
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from lingo import (
	Corpus,
	Reference,
	speak,
	phrase,
	scribble,
	speak,
	verb,
)

# TASK_TTS = "kokoro_tts"
# TASK_RENDER = "multitalk.video.render"


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


class KokoroSettings(BaseModel):
    language_code: str = "a"
    repo_id: str = "weights/Kokoro-82M"
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


def dispatch_multitalk_pipeline(
    speech_text: str,
    voice_id: str,
    avatar: Corpus[bytes],
    *,
    video_prompt: str | None = None,
    tts_param: Optional[KokoroSettings] = None,
    render_param: Optional[MultiTalkSettings] = None,
    dest: Optional[Reference] = None,
    timeout_seconds: int | None = None,
) -> Corpus[bytes]:
    """Dispatch the two-step lingo pipeline and wait for completion."""

    tts_step = phrase('kokoro_tts')(speech_text, voice_id, tts_param).local()
    render_step = phrase('multitalk.video.render')(tts_step, avatar, video_prompt, dest, render_param)
    dispatched = render_step.say()
    result = dispatched.get(timeout=timeout_seconds)
    return result


# This worker requires Redis broker + MinIO for claim-check video destinations.
lang = speak(redis=True, minio=True, mongo=False)


@verb('kokoro_tts')
def kokoro_tts(speech_text: str, voice_id: str, param: Optional[KokoroSettings] = None) -> Corpus[bytes]:
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

    pipeline = KPipeline(lang_code=param.language_code, repo_id=param.repo_id)
    voice_tensor = torch.load(voice_path, weights_only=True)

    chunks = []
    generator = pipeline(
        speech_text,
        voice=voice_tensor,
        speed=param.speed,
        split_pattern=param.split_pattern,
    )
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio samples")

    merged = torch.concat(chunks, dim=0)
    sf.write(str(output_path), merged, 24000)

    # Normalize sample rate for downstream consistency.
    wav_normalized, _ = librosa.load(str(output_path), sr=param.output_sample_rate)
    sf.write(str(output_path), wav_normalized, param.output_sample_rate)

    return Corpus.from_file(str(output_path), content_type="audio/wav")


@verb('multitalk.video.render')
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

    audio_path = Path(audio.materialize_path())
    if not audio_path.exists():
        raise FileNotFoundError(f"TTS audio could not be materialized: {audio_path}")

    avatar_path = Path(avatar.materialize_path())
    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar could not be materialized: {avatar_path}")

    output_video = _run_multitalk_render(audio_path, avatar_path, video_prompt, param)

    if dest is None:
        video_corpus = Corpus[bytes].from_file(str(output_video), content_type="video/mp4")
    else:
        video_corpus = dest.dump_file(output_video)

    return video_corpus
