"""
@file cli.py
@brief Backend-facing CLI wrapper for the MultiTalk generator.
"""

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import uuid
import wave
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import requests

import config

# Characters per second for speech duration estimation (matches app/services/eta_service.py)
CHARS_PER_SECOND = 15.0


def _guess_audio_extension(audio_url: str, content_type: str | None) -> str:
    """
    @brief Infer a suitable file extension for a downloaded audio asset.
    @param audio_url Source URL for the audio asset.
    @param content_type HTTP content type header value.
    @return File extension including the leading dot.
    """

    url_path = urlparse(audio_url).path
    url_ext = os.path.splitext(url_path)[1].lower()
    if url_ext:
        return url_ext

    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type:
        guessed_ext = mimetypes.guess_extension(normalized_type)
        if guessed_ext:
            return guessed_ext

    return ".bin"


def _estimate_speech_duration_seconds(speech_text: str) -> float | None:
    """
    @brief Estimate video duration in seconds from speech text using character count.
    @param speech_text The speech text to estimate duration for.
    @return Estimated duration in seconds, or None if speech_text is empty/missing.
    """
    if not speech_text or not isinstance(speech_text, str):
        return None
    
    # Estimate duration: characters / characters_per_second
    duration = len(speech_text) / CHARS_PER_SECOND
    return duration


def _estimate_audio_duration_seconds(audio_path: str) -> float | None:
    """
    @brief Estimate video duration in seconds from an input audio file.
    @param audio_path Path to the input audio file.
    @return Estimated duration in seconds, or None if audio_path is empty/missing.
    @details WAV files are measured directly. Other formats fall back to None so the
             wrapper can use the conservative default frame budget.
    @throws RuntimeError when the audio file cannot be opened.
    """

    if not audio_path:
        return None

    try:
        with wave.open(audio_path, "rb") as audio_handle:
            frame_rate = audio_handle.getframerate()
            if frame_rate <= 0:
                return None
            return audio_handle.getnframes() / float(frame_rate)
    except wave.Error:
        return None
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file {audio_path}: {exc}") from exc


def _download_audio_from_url(audio_url: str, download_dir: str) -> str:
    """
    @brief Download an audio asset from a URL into the working directory.
    @param audio_url Remote URL to download.
    @param download_dir Directory where the downloaded file should be stored.
    @return Absolute path to the downloaded file.
    @throws RuntimeError when the download fails.
    """

    if not audio_url:
        raise RuntimeError("audio_url must not be empty")

    try:
        response = requests.get(audio_url, stream=True, timeout=(10, 300))
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download audio from {audio_url}: {exc}") from exc

    extension = _guess_audio_extension(audio_url, response.headers.get("content-type"))
    download_path = os.path.join(download_dir, f"downloaded_audio_{uuid.uuid4().hex}{extension}")

    try:
        with open(download_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"Failed to write downloaded audio to {download_path}: {exc}") from exc

    return download_path


def _preprocess_audio_for_multitalk(audio_path: str, work_dir: str) -> str:
    """
    @brief Normalize downloaded audio into a WAV file that MultiTalk can ingest predictably.
    @param audio_path Source audio path.
    @param work_dir Working directory for generated intermediates.
    @return Path to the normalized audio file.
    @throws RuntimeError when ffmpeg conversion fails.
    """

    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path

    normalized_path = os.path.join(work_dir, f"normalized_audio_{uuid.uuid4().hex}.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        normalized_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess audio file {audio_path}: {exc}") from exc

    return normalized_path


def _resolve_input_audio_path(
    base_dir: str,
    work_dir: str,
    data: Dict[str, Any],
    audio_path: str | None = None,
    audio_url: str | None = None,
) -> str | None:
    """
    @brief Resolve the audio source to a local file path for MultiTalk.
    @details Precedence is CLI path, JSON path, CLI URL, JSON URL.
             URL inputs are downloaded locally and normalized to WAV when needed.
    @param base_dir Base directory for resolving relative local paths.
    @param work_dir Working directory for downloaded and normalized assets.
    @param data Raw job data.
    @param audio_path Optional CLI-supplied local audio path.
    @param audio_url Optional CLI-supplied audio URL.
    @return Absolute local file path, or None if no audio source was provided.
    """

    preferred_path = audio_path or data.get("audio_path")
    if preferred_path:
        return _resolve_path(base_dir, preferred_path)

    preferred_url = audio_url or data.get("audio_url")
    if not preferred_url:
        return None

    downloaded_path = _download_audio_from_url(preferred_url, work_dir)
    return _preprocess_audio_for_multitalk(downloaded_path, work_dir)


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
    @brief Select the avatar configuration JSON and image file from a directory.
    @details Finds the first .json file (assumed to be base.json or config) and
             the first image file (.png, .jpg, .jpeg, or .webp) in the directory.
             Files are selected in sorted order, so naming is predictable.
    @param avatar_dir Directory containing avatar assets.
    @return Tuple of (json_config_path, image_file_path).
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
            f"Avatar directory must contain one JSON config file and one image file (.png/.jpg/.jpeg/.webp): {avatar_dir}"
        )

    return json_path, image_path

def _resolve_voice_path(preferred_voice: str | None, base_dir: str) -> str:
    """
    @brief Resolve a preferred voice identifier to an absolute voice file path.
    @details Accepts a full relative path (e.g. ``weights/Kokoro-82M/voices/af_heart.pt``),
             a filename (``af_heart.pt``), or just a voice name (``af_heart``).
    @param preferred_voice Voice identifier supplied by the caller. May be *None*.
    @param base_dir Base directory for relative path resolution (multitalk repo root).
    @return Absolute path to the voice ``.pt`` file.
    """

    if not preferred_voice:
        return _resolve_path(base_dir, config.TTS_VOICE)

    # Already looks like a path (contains separator or ends with .pt)
    if "/" in preferred_voice or preferred_voice.endswith(".pt"):
        return _resolve_path(base_dir, preferred_voice)

    # Bare voice name → resolve inside the Kokoro voices directory.
    voice_path = f"weights/Kokoro-82M/voices/{preferred_voice}.pt"
    return _resolve_path(base_dir, voice_path)


def _build_input_from_template(
    data: Dict[str, Any], base_dir: str, avatar_image_path: str
) -> Dict[str, Any]:
    """
    @brief Build an input payload using the checked-in base TTS template.
    @details Used when the caller supplies a ``avatar`` name (found in S3 bucket under ``avatars/``)
             The template's ``cond_image``, ``tts_audio.text``, and ``tts_audio.human1_voice``
             are replaced with the caller-provided values.
    @param data Raw job data dictionary.
    @param base_dir Multitalk repo directory (for path resolution).
    @param avatar_image_path Absolute path to the downloaded avatar image.
    @return Payload dictionary ready for multitalk ``input_json``.
    @throws RuntimeError when required fields are missing.
    """

    template_path = os.path.join(base_dir, "base_tts_template.json")
    payload = _load_json(template_path)

    # Replace the avatar image with the downloaded one.
    payload["cond_image"] = avatar_image_path

    if "cond_audio" not in payload:
        payload["cond_audio"] = {}

    # Speech text is mandatory.
    speech_text = data.get("speech_text")
    if not speech_text:
        raise RuntimeError("speech_text is required for multitalk TTS mode")

    # Resolve the voice file path.
    preferred_voice = data.get("preferredVoice")
    voice_path = _resolve_voice_path(preferred_voice, base_dir)

    payload["tts_audio"] = {
        "text": speech_text,
        "human1_voice": voice_path,
    }

    return payload


def _build_input_payload(
    data: Dict[str, Any], base_dir: str, audio_path: str | None = None
) -> Dict[str, Any]:
    """
    @brief Build the input payload expected by the multitalk generator.
    @details Transforms job JSON into the multitalk generator format:
             - Resolves image path to absolute path
             - If audio input is present, uses cond_audio/person1 and skips TTS
             - Otherwise resolves Kokoro voice and speech_text into tts_audio
    @param data Raw job data from the backend JSON file.
    @param base_dir Base directory for resolving paths (repo directory).
    @param audio_path Optional local audio file path.
    @return Payload dictionary ready for multitalk input_json.
    @throws RuntimeError when required fields are missing.
    """

    # Extract required fields from avatar config
    prompt = data.get("video_prompt")  # Fallback to default prompt if not specified in avatar config
    if not prompt:
        raise RuntimeError(f"Job data must contain 'video_prompt' field: {data}")
    
    avatar_path = data.get("avatar_path")
    if not avatar_path:
        raise RuntimeError(f"Job data must contain 'avatar_path' field: {data}")

    resolved_audio_path = audio_path or data.get("audio_path")
    if resolved_audio_path:
        return {
            "prompt": prompt,
            "cond_image": _resolve_path(base_dir, avatar_path),
            "cond_audio": {
                "person1": _resolve_path(base_dir, resolved_audio_path),
            },
        }

    voice = data.get("kokoro_voice")
    if not voice:
        raise RuntimeError(f"Job data must contain 'kokoro_voice' field: {data}")

    speech_text = data.get("speech_text")
    if not speech_text:
        raise RuntimeError(f"Job data must contain 'speech_text' field: {data}")

    # Build the payload in the expected format
    payload = {
        "prompt": prompt,
        "cond_image": _resolve_path(base_dir, avatar_path),
        "tts_audio": {
            "text": speech_text,
            "human1_voice": _resolve_path(base_dir, f"weights/Kokoro-82M/voices/{voice}.pt"),
        },
        "cond_audio": {},
    }
    
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


def _write_json(path: str, payload: Dict[str, Any]) -> None:
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
    parser.add_argument("--audio", default=None)
    parser.add_argument("--audio-url", default=None)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    work_dir = args.work_dir
    if work_dir is None:
        print('WARNING: no --work-dir specified, using default "backend_runs/{job_id}"')
        work_dir = os.path.join(repo_dir, "backend_runs", args.job_id)
    os.makedirs(work_dir, exist_ok=True)

    input_json_path = os.path.join(work_dir, f"cond.json")
    audio_save_dir = os.path.join(work_dir, "audio")

    data = _load_json(args.data)
    resolved_audio_path = _resolve_input_audio_path(
        base_dir=repo_dir,
        work_dir=work_dir,
        data=data,
        audio_path=args.audio,
        audio_url=args.audio_url,
    )
    payload = _build_input_payload(
        data=data,
        base_dir=repo_dir,
        audio_path=resolved_audio_path,
    )
    _write_json(input_json_path, payload)
    audio_mode = "localfile" if payload.get("cond_audio") else "tts"

    ckpt_dir = config.CKPT_DIR
    wav2vec_dir = config.WAV2VEC_DIR
    if not ckpt_dir or not wav2vec_dir:
        raise RuntimeError("Missing CKPT_DIR or WAV2VEC_DIR in config.py")

    ckpt_dir = _resolve_path(repo_dir, ckpt_dir)
    wav2vec_dir = _resolve_path(repo_dir, wav2vec_dir)

    if audio_mode == "tts":
        _ensure_kokoro_weights(repo_dir)

    # Calculate frame_num and max_frames_num based on video duration
    FPS = 25.0  # Video frames per second
    frames_estimated: int | None = None
    if audio_mode == "localfile":
        input_audio_path = payload["cond_audio"]["person1"]
        video_duration_seconds = _estimate_audio_duration_seconds(input_audio_path)
    else:
        speech_text = data.get("speech_text", "")
        video_duration_seconds = (
            _estimate_speech_duration_seconds(speech_text) if speech_text else None
        )

    # Choose the largest safe frame_num for this text, with safety margin.
    # Keep it in [33, 81] and enforce frame_num = 4n+1.
    MIN_FRAMES = 33   # 4*8 + 1
    MAX_FRAMES = 81   # 4*20 + 1

    if video_duration_seconds is not None:
        # Estimated frames for the (TTS) audio, with a small safety margin
        frames_estimated = int(video_duration_seconds * FPS)
        frames_target = int(frames_estimated * 0.9) # 10% safety margin

        # Clamp into [MIN_FRAMES, MAX_FRAMES]
        frames_target = max(MIN_FRAMES, min(MAX_FRAMES, frames_target))

        # frame_num must be 4n+1 -> round DOWN to nearest 4n+1 <= frames_target
        remainder = (frames_target - 1) % 4
        frame_num = frames_target - remainder
        if frame_num < MIN_FRAMES:
            frame_num = MIN_FRAMES
    else:
        # Unknown duration: fall back to the most conservative (max) clip length
        frame_num = MAX_FRAMES

    # # Calculate max_frames_num for longer videos
    mode = "streaming"
    # if mode == "clip":
    #     max_frames_num = frame_num
    # else:
    #     # Streaming mode: calculate frames needed based on video duration
    #     if video_duration_seconds is not None:
    #         # Calculate frames needed: duration * fps
    #         frames_needed = int(video_duration_seconds * FPS)
    #         # Ensure it's at least frame_num
    #         max_frames_num = max(frame_num, frames_needed)
    #         # Round up to next 4n+1 if needed (to match frame_num pattern)
    #         remainder = (max_frames_num - 1) % 4
    #         if remainder != 0:
    #             max_frames_num = max_frames_num + (4 - remainder)
    #     else:
    #         max_frames_num = 1000  # default for streaming when duration unknown
    max_frames_num = 2000

    if frames_estimated is not None and frames_estimated > max_frames_num:
        raise RuntimeError(f"Estimated frames {frames_estimated} is greater than max_frames_num {max_frames_num}. "
                            "Max runtime will be exceeded")

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
        mode,
        "--num_persistent_param_in_dit",
        str(data.get("num_persistent_param_in_dit", 0)),
        "--audio_mode",
        audio_mode,
        "--audio_save_dir",
        audio_save_dir,
        "--save_file",
        os.path.splitext(args.output)[0],
        "--frame_num",
        str(frame_num),
    ]

    # Add max_frames_num argument for streaming mode
    if mode == "streaming":
        command.extend(["--max_frames_num", str(max_frames_num)])

    if data.get("use_teacache", True):
        command.append("--use_teacache")

    # FusionX LoRA path: 8-step acceleration. Do not pass sample_audio_guide_scale so
    # generate_multitalk keeps its default (4.0) for stronger lip sync.
    lora_dir = data.get("lora_dir", getattr(config, "LORA_DIR", "") or "")
    if str(lora_dir).strip():
        lora_path = _resolve_path(repo_dir, str(lora_dir).strip())
        if not os.path.isfile(lora_path):
            raise RuntimeError(f"LoRA weights not found: {lora_path}")
        command.extend(
            [
                "--lora_dir",
                lora_path,
                "--lora_scale",
                str(data.get("lora_scale", getattr(config, "LORA_SCALE", 1.0))),
                "--sample_shift",
                str(data.get("sample_shift", getattr(config, "SAMPLE_SHIFT", 2))),
                "--sample_text_guide_scale",
                str(
                    data.get(
                        "sample_text_guide_scale",
                        getattr(config, "SAMPLE_TEXT_GUIDE_SCALE", 1.0),
                    )
                ),
            ]
        )

    try:
        _run_command_streaming(command, cwd=repo_dir)
    finally:
        try:
            shutil.rmtree(work_dir)
            # print(f"Skipping cleanup of work_dir: {work_dir}\n command ran \n {command}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

