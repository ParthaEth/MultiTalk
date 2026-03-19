"""Lingo worker tasks for MultiTalk orchestration.

This worker intentionally exposes exactly two verbs:
1. Kokoro TTS -> stores generated audio as a local Corpus.
2. MultiTalk render -> consumes TTS Corpus and stores video as a MinIO Corpus.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import requests

from pydantic import BaseModel, Field

try:
    from lingo import (
        BaseJobPayload,
        Corpus,
        StorageContext,
        get_celery_app,
        grammar,
        phrase,
        validate_channels,
        verb,
    )
except ModuleNotFoundError:
    workspace_lingo = Path(__file__).resolve().parents[3] / "lingo"
    if workspace_lingo.exists():
        sys.path.insert(0, str(workspace_lingo))
    from lingo import (
        BaseJobPayload,
        Corpus,
        StorageContext,
        get_celery_app,
        grammar,
        phrase,
        validate_channels,
        verb,
    )

TASK_TTS = "kokoro_tts"
TASK_RENDER = "multitalk.video.render"


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


class MultiTalkPayload(BaseJobPayload):
    speech_text: str = Field(min_length=1)
    video_prompt: str = Field(min_length=1)
    kokoro_voice: str = Field(min_length=1)
    avatar_path: str = Field(min_length=1)

    # Used by server orchestration and MinIO object layout.
    job_id: str | None = None
    creator_email: str = "unknown"

    output_dir: str = "./backend_runs"
    output_name: str = "generated.mp4"
    video_storage_base_path: str | None = None

    sample_steps: int = Field(default=40, ge=1)
    num_persistent_param_in_dit: int = Field(default=0, ge=0)
    use_teacache: bool = True


class KokoroTTSResult(BaseModel):
    pipeline_id: str
    audio_corpus: Corpus[Any]
    local_audio_path: str
    output_bytes: int = Field(ge=0)


class MultiTalkRenderResult(BaseModel):
    pipeline_id: str
    video_corpus: Corpus[Any]
    video_locator: str
    local_output_video_path: str
    output_bytes: int = Field(ge=0)



def _normalize_output_name(name: str) -> str:
    if name.lower().endswith(".mp4"):
        return name
    return f"{name}.mp4"


def _resolve_voice_path(voice: str, repo_dir: Path) -> str:
    normalized = voice.strip()
    if not normalized:
        raise RuntimeError("kokoro_voice is required")

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


def _run_kokoro_tts(payload: MultiTalkPayload, run_dir: Path) -> Path:
    # Local imports keep worker startup light and only require Kokoro/Torch on TTS nodes.
    from kokoro import KPipeline
    import librosa
    import soundfile as sf
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = (run_dir / "tts.wav").resolve()

    repo_dir = _repo_dir()
    voice_path = _resolve_voice_path(payload.kokoro_voice, repo_dir)
    if not Path(voice_path).exists():
        raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

    pipeline = KPipeline(lang_code="a", repo_id="weights/Kokoro-82M")
    voice_tensor = torch.load(voice_path, weights_only=True)

    chunks = []
    generator = pipeline(
        payload.speech_text,
        voice=voice_tensor,
        speed=1,
        split_pattern=r"\n+",
    )
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio samples")

    merged = torch.concat(chunks, dim=0)
    sf.write(str(output_path), merged, 24000)

    # Normalize to 16k for downstream consistency with MultiTalk's audio pipeline.
    wav_16k, _ = librosa.load(str(output_path), sr=16000)
    sf.write(str(output_path), wav_16k, 16000)
    return output_path


def _build_multitalk_input(payload: MultiTalkPayload, audio_file: Path, path: Path) -> None:
    avatar = _materialize_avatar(payload.avatar_path, path.parent)
    if not avatar.exists():
        raise FileNotFoundError(f"avatar_path not found after materialization: {payload.avatar_path}")

    data = {
        "prompt": payload.video_prompt,
        "cond_image": str(avatar.resolve()),
        "cond_audio": {
            "person1": str(audio_file.resolve()),
        },
        "audio_type": "add",
        "tts_audio": {},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _materialize_avatar(avatar_path: str, work_dir: Path) -> Path:
    parsed = urlparse(avatar_path)
    if parsed.scheme in {"http", "https"}:
        ext = Path(parsed.path).suffix or ".png"
        target = (work_dir / f"avatar{ext}").resolve()
        response = requests.get(avatar_path, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
        return target

    candidate = Path(avatar_path)
    if not candidate.is_absolute():
        candidate = (_repo_dir() / candidate).resolve()
    return candidate


def _run_multitalk_render(payload: MultiTalkPayload, audio_file: Path) -> Path:
    repo_dir = _repo_dir()
    pipeline_id = str(payload.pipeline_context.pipeline_id)

    output_dir = (repo_dir / payload.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = _normalize_output_name(payload.output_name)
    output_video_path = output_dir / output_name

    run_dir = output_dir / f"run_{pipeline_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_json_path = run_dir / "lingo_input.json"
    audio_save_dir = run_dir / "audio"
    audio_save_dir.mkdir(parents=True, exist_ok=True)
    _build_multitalk_input(payload, audio_file, input_json_path)

    command = [
        sys.executable,
        str(repo_dir / "generate_multitalk.py"),
        "--task",
        "multitalk-14B",
        "--size",
        "multitalk-480",
        "--ckpt_dir",
        "./weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir",
        "./weights/chinese-wav2vec2-base",
        "--input_json",
        str(input_json_path),
        "--audio_mode",
        "localfile",
        "--sample_steps",
        str(payload.sample_steps),
        "--mode",
        "streaming",
        "--num_persistent_param_in_dit",
        str(payload.num_persistent_param_in_dit),
        "--audio_save_dir",
        str(audio_save_dir),
        "--save_file",
        str(output_video_path),
    ]
    if payload.use_teacache:
        command.append("--use_teacache")

    _run_command(command, cwd=repo_dir)
    return _find_generated_video(output_video_path)


def _safe_creator(creator_email: str) -> str:
    return creator_email.replace("@", "_at_").replace("/", "_")


def _video_storage_context(payload: MultiTalkPayload) -> StorageContext:
    if payload.video_storage_base_path:
        base_path = payload.video_storage_base_path
    else:
        object_stem = payload.job_id or str(payload.pipeline_context.pipeline_id)
        base_path = f"{_safe_creator(payload.creator_email)}/{object_stem}"
    return StorageContext(strategy="minio", base_path=base_path)


def dispatch_multitalk_pipeline(
    payload: MultiTalkPayload,
    *,
    timeout_seconds: int | None = None,
) -> MultiTalkRenderResult:
    """Dispatch the two-step lingo pipeline and wait for completion."""

    tts_step = phrase(TASK_TTS, wait=True)(payload)
    render_step = phrase(TASK_RENDER, wait=True)(tts_step, payload)
    dispatched = render_step.say()
    result = dispatched.get(timeout=timeout_seconds)
    if isinstance(result, MultiTalkRenderResult):
        return result
    return MultiTalkRenderResult.model_validate(result)


def minio_object_key_from_locator(locator: str) -> str | None:
    """Extract MinIO object key from an `s3://bucket/key` locator."""

    if not locator.startswith("s3://"):
        return None
    match = re.match(r"^s3://[^/]+/(.+)$", locator)
    if not match:
        return None
    return match.group(1)


# This worker requires Redis broker + MinIO for the video Corpus claim-check.
validate_channels(redis=True, minio=True, mongo=False)
app = get_celery_app()
app.conf.worker_hostname = "multitalk@%h"


@verb(TASK_TTS, bind=True)
def kokoro_tts(self, payload: MultiTalkPayload) -> KokoroTTSResult:
    pipeline_id = str(payload.pipeline_context.pipeline_id)
    run_dir = (_repo_dir() / payload.output_dir / f"run_{pipeline_id}" / "tts").resolve()
    audio_path = _run_kokoro_tts(payload, run_dir)

    # No explicit storage_context on purpose: default Corpus backend is local.
    audio_corpus = Corpus.from_file(
        str(audio_path),
        content_type="audio/wav",
        metadata={
            "pipeline_id": pipeline_id,
            "task": TASK_TTS,
            "task_id": self.request.id or "unknown",
        },
    )

    return KokoroTTSResult(
        pipeline_id=pipeline_id,
        audio_corpus=audio_corpus,
        local_audio_path=str(audio_path),
        output_bytes=audio_path.stat().st_size,
    )


@verb(TASK_RENDER, bind=True)
def render_multitalk(
    self,
    tts: KokoroTTSResult,
    payload: MultiTalkPayload,
) -> MultiTalkRenderResult:
    source_audio = Path(tts.audio_corpus.materialize())
    if not source_audio.exists():
        raise FileNotFoundError(f"TTS audio corpus could not be materialized: {source_audio}")

    output_video = _run_multitalk_render(payload, source_audio)

    # Stage with deterministic filename before MinIO upload; ArtifactManager uses source.name.
    stage_dir = output_video.parent / "publish"
    stage_dir.mkdir(parents=True, exist_ok=True)
    object_name = _normalize_output_name(payload.output_name)
    staged_video = stage_dir / object_name
    if output_video.resolve() != staged_video.resolve():
        shutil.copy2(output_video, staged_video)

    video_corpus = Corpus.from_file(
        str(staged_video),
        storage_context=_video_storage_context(payload),
        content_type="video/mp4",
        metadata={
            "pipeline_id": str(payload.pipeline_context.pipeline_id),
            "task": TASK_RENDER,
            "task_id": self.request.id or "unknown",
            "source_audio_locator": tts.audio_corpus.locator,
        },
    )

    return MultiTalkRenderResult(
        pipeline_id=str(payload.pipeline_context.pipeline_id),
        video_corpus=video_corpus,
        video_locator=video_corpus.locator,
        local_output_video_path=str(output_video),
        output_bytes=output_video.stat().st_size,
    )
