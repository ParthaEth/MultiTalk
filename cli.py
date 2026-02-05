"""
@file cli.py
@brief Backend-facing CLI wrapper for the MultiTalk generator.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, Tuple

import config


def _run_command_streaming(command: list[str], cwd: str) -> None:
    """
    @brief Run a subprocess while streaming stdout/stderr to the current process.
    @param command Command list to execute.
    @param cwd Working directory for the subprocess.
    @throws RuntimeError when the command exits non-zero.
    """

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)

    rc = proc.wait()
    if rc != 0:
        tail_text = "".join(tail).strip()
        raise RuntimeError(
            f"multitalk generation failed with exit code {rc}. Last output:\n{tail_text}"
        )


def _resolve_path(base_dir: str, path_value: str) -> str:
    """
    @brief Resolve a path relative to the multitalk repo if needed.
    @param base_dir Base directory for relative resolution.
    @param path_value Path to resolve.
    @return Absolute path for the input value.
    """

    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _ensure_kokoro_weights(repo_dir: str) -> None:
    """
    @brief Ensure the Kokoro weights are reachable via `weights/Kokoro-82M` from repo_dir.
    @details MultiTalk's `generate_multitalk.py` uses a relative repo_id (`weights/Kokoro-82M`),
             so we provide a symlink to the absolute directory configured in `config.KOKORO_DIR`.
    @param repo_dir MultiTalk repo directory.
    """

    kokoro_dir = getattr(config, "KOKORO_DIR", "")
    if not kokoro_dir:
        return

    weights_dir = os.path.join(repo_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    link_path = os.path.join(weights_dir, "Kokoro-82M")

    if os.path.exists(link_path):
        return

    try:
        os.symlink(kokoro_dir, link_path)
    except Exception:
        # If symlinks are not permitted, fall back to doing nothing; the generator will error clearly.
        pass


def _select_avatar_assets(avatar_dir: str) -> Tuple[str, str]:
    """
    @brief Select the avatar JSON and image file from the avatar directory.
    @param avatar_dir Directory containing avatar assets.
    @return Tuple of (json_path, image_path).
    @throws RuntimeError if required files are missing.
    """

    if not os.path.isdir(avatar_dir):
        raise RuntimeError(f"Avatar directory not found: {avatar_dir}")

    entries = sorted(os.listdir(avatar_dir))
    json_path = ""
    image_path = ""
    for name in entries:
        path = os.path.join(avatar_dir, name)
        if os.path.isdir(path):
            continue
        lower = name.lower()
        if lower.endswith(".json") and not json_path:
            json_path = path
        elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")) and not image_path:
            image_path = path

    if not json_path or not image_path:
        raise RuntimeError(
            f"Avatar directory must contain one json and one image file: {avatar_dir}"
        )

    return json_path, image_path


def _build_input_payload(
    data: Dict[str, Any], base_dir: str, avatar_json: str, avatar_image: str
) -> Dict[str, Any]:
    """
    @brief Build the input payload expected by the multitalk generator.
    @param data Raw job data from the backend JSON file.
    @param base_dir Base directory for resolving paths.
    @param avatar_json Path to the base JSON file.
    @param avatar_image Path to the avatar image file.
    @return Payload dictionary ready for multitalk input_json.
    @throws RuntimeError when required fields are missing.
    """

    payload = _load_json(avatar_json)
    payload["cond_image"] = _resolve_path(base_dir, avatar_image)
    if "cond_audio" not in payload:
        payload["cond_audio"] = {}

    tts_audio: Dict[str, Any] = data.get("tts_audio", {})
    speech_text = data.get("speech_text")
    if not speech_text:
        raise RuntimeError("speech_text is required for multitalk TTS mode")

    tts_audio["text"] = speech_text

    if tts_audio:
        if "text" not in tts_audio:
            raise RuntimeError("tts_audio provided but missing 'text'")
        if "human1_voice" not in tts_audio:
            tts_audio["human1_voice"] = config.TTS_VOICE
        if "human2_voice" in tts_audio and tts_audio["human2_voice"]:
            tts_audio["human2_voice"] = _resolve_path(
                base_dir, tts_audio["human2_voice"]
            )
        tts_audio["human1_voice"] = _resolve_path(base_dir, tts_audio["human1_voice"])
        payload["tts_audio"] = tts_audio

    return payload


def _load_json(path: str) -> Dict[str, Any]:
    """
    @brief Load JSON content from disk.
    @param path File path to read.
    @return Parsed JSON dictionary.
    @throws RuntimeError if JSON cannot be read.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON from {path}: {exc}") from exc


def :quit_write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    @brief Write JSON content to disk.
    @param path File path to write.
    @param payload JSON-serializable dictionary.
    """

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main() -> None:
    """
    @brief CLI entrypoint for backend-triggered multitalk generation.
    @throws RuntimeError on invalid input or generation failure.
    """

    parser = argparse.ArgumentParser(
        description="Backend wrapper for MultiTalk generation",
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(repo_dir, "backend_runs", args.job_id)
    os.makedirs(work_dir, exist_ok=True)

    input_json_path = os.path.join(work_dir, f"{uuid.uuid4().hex}.json")
    audio_save_dir = os.path.join(work_dir, "audio")

    data = _load_json(args.data)
    avatar_json, avatar_image = _select_avatar_assets(config.AVATAR_DIR)
    payload = _build_input_payload(
        data=data,
        base_dir=repo_dir,
        avatar_json=avatar_json,
        avatar_image=avatar_image,
    )
    _write_json(input_json_path, payload)

    ckpt_dir = config.CKPT_DIR
    wav2vec_dir = config.WAV2VEC_DIR
    if not ckpt_dir or not wav2vec_dir:
        raise RuntimeError("Missing CKPT_DIR or WAV2VEC_DIR in config.py")

    ckpt_dir = _resolve_path(repo_dir, ckpt_dir)
    wav2vec_dir = _resolve_path(repo_dir, wav2vec_dir)

    _ensure_kokoro_weights(repo_dir)

    command = [
        sys.executable,
        os.path.join(repo_dir, "generate_multitalk.py"),
        "--ckpt_dir",
        ckpt_dir,
        "--wav2vec_dir",
        wav2vec_dir,
        "--input_json",
        input_json_path,
        "--sample_steps",
        str(data.get("sample_steps", getattr(config, "SAMPLE_STEPS", 40))),
        "--mode",
        str(data.get("mode", "streaming")),
        "--num_persistent_param_in_dit",
        str(data.get("num_persistent_param_in_dit", 0)),
        "--audio_mode",
        "tts",
        "--audio_save_dir",
        audio_save_dir,
        "--save_file",
        os.path.splitext(args.output)[0],
    ]

    if data.get("use_teacache", True):
        command.append("--use_teacache")

    try:
        _run_command_streaming(command, cwd=repo_dir)
    finally:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()

