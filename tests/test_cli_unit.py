"""Unit tests for multitalk CLI helpers and command wiring."""

from __future__ import annotations

import importlib.util
import os
import sys
import wave
from pathlib import Path

import pytest


def _load_module(module_name: str, filename: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


cli = _load_module("multitalk_cli_for_tests", "cli.py")


pytestmark = pytest.mark.unit


def test_estimate_speech_duration_seconds() -> None:
    assert cli._estimate_speech_duration_seconds("") is None
    assert cli._estimate_speech_duration_seconds(None) is None
    assert cli._estimate_speech_duration_seconds("a" * 30) == 2.0


def test_estimate_audio_duration_seconds_for_wav_and_other_formats(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    assert cli._estimate_audio_duration_seconds(str(wav_path)) == 1.0

    mp3_path = tmp_path / "sample.mp3"
    mp3_path.write_bytes(b"not-a-real-mp3")
    assert cli._estimate_audio_duration_seconds(str(mp3_path)) is None


def test_guess_audio_extension_prefers_url_path_then_content_type() -> None:
    assert cli._guess_audio_extension("https://example.com/audio.mp3", None) == ".mp3"
    assert cli._guess_audio_extension("https://example.com/audio", "audio/mpeg") == ".mp3"
    assert cli._guess_audio_extension("https://example.com/audio", None) == ".bin"


def test_resolve_voice_path_supports_name_file_and_explicit_path(tmp_path) -> None:
    base_dir = str(tmp_path)
    cli.config.TTS_VOICE = "weights/Kokoro-82M/voices/default.pt"

    assert cli._resolve_voice_path(None, base_dir) == os.path.join(
        base_dir, "weights/Kokoro-82M/voices/default.pt"
    )
    assert cli._resolve_voice_path("af_heart", base_dir) == os.path.join(
        base_dir, "weights/Kokoro-82M/voices/af_heart.pt"
    )
    assert cli._resolve_voice_path("voices/af.pt", base_dir) == os.path.join(
        base_dir, "voices/af.pt"
    )
    assert cli._resolve_voice_path("af.pt", base_dir) == os.path.join(base_dir, "af.pt")


def test_build_input_payload_requires_expected_keys(tmp_path) -> None:
    base_dir = str(tmp_path)
    payload = cli._build_input_payload(
        {
            "video_prompt": "confident spokesperson",
            "kokoro_voice": "af_heart",
            "speech_text": "hello there",
            "avatar_path": "avatar.png",
        },
        base_dir,
    )

    assert payload["prompt"] == "confident spokesperson"
    assert payload["cond_image"] == os.path.join(base_dir, "avatar.png")
    assert payload["tts_audio"]["human1_voice"] == os.path.join(
        base_dir, "weights/Kokoro-82M/voices/af_heart.pt"
    )

    with pytest.raises(RuntimeError, match="speech_text"):
        cli._build_input_payload(
            {
                "video_prompt": "confident spokesperson",
                "kokoro_voice": "af_heart",
                "avatar_path": "avatar.png",
            },
            base_dir,
        )


def test_build_input_payload_uses_audio_file_when_present(tmp_path) -> None:
    base_dir = str(tmp_path)
    payload = cli._build_input_payload(
        {
            "video_prompt": "confident spokesperson",
            "avatar_path": "avatar.png",
            "audio_path": "voice.wav",
        },
        base_dir,
    )

    assert payload["prompt"] == "confident spokesperson"
    assert payload["cond_image"] == os.path.join(base_dir, "avatar.png")
    assert payload["cond_audio"]["person1"] == os.path.join(base_dir, "voice.wav")
    assert "tts_audio" not in payload


def test_resolve_input_audio_path_prefers_local_path_over_url(tmp_path) -> None:
    resolved = cli._resolve_input_audio_path(
        base_dir=str(tmp_path),
        work_dir=str(tmp_path),
        data={"audio_path": "voice.wav", "audio_url": "https://example.com/voice.mp3"},
    )

    assert resolved == os.path.join(str(tmp_path), "voice.wav")


def test_resolve_input_audio_path_downloads_and_preprocesses_url(monkeypatch, tmp_path) -> None:
    captured = {"download_url": None, "preprocess_path": None}

    monkeypatch.setattr(
        cli,
        "_download_audio_from_url",
        lambda audio_url, download_dir: captured.update({"download_url": audio_url}) or os.path.join(download_dir, "download.mp3"),
    )
    monkeypatch.setattr(
        cli,
        "_preprocess_audio_for_multitalk",
        lambda audio_path, work_dir: captured.update({"preprocess_path": audio_path}) or os.path.join(work_dir, "normalized.wav"),
    )

    resolved = cli._resolve_input_audio_path(
        base_dir=str(tmp_path),
        work_dir=str(tmp_path),
        data={"audio_url": "https://example.com/voice.mp3"},
    )

    assert captured["download_url"] == "https://example.com/voice.mp3"
    assert captured["preprocess_path"] == os.path.join(str(tmp_path), "download.mp3")
    assert resolved == os.path.join(str(tmp_path), "normalized.wav")


def test_run_command_streaming_raises_with_tail_output(monkeypatch) -> None:
    class FakeProc:
        def __init__(self, rc, lines):
            self.stdout = iter(lines)
            self._rc = rc

        def wait(self):
            return self._rc

    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProc(1, ["line-1\n", "line-2\n"]),
    )

    with pytest.raises(RuntimeError, match="line-2"):
        cli._run_command_streaming(["fake"], cwd="/")


def test_main_builds_expected_command_and_cleans_workdir(monkeypatch, tmp_path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    wav2vec_dir = tmp_path / "wav2vec"
    ckpt_dir.mkdir()
    wav2vec_dir.mkdir()

    monkeypatch.setattr(cli.config, "CKPT_DIR", str(ckpt_dir))
    monkeypatch.setattr(cli.config, "WAV2VEC_DIR", str(wav2vec_dir))

    monkeypatch.setattr(
        cli,
        "_load_json",
        lambda _path: {
            "video_prompt": "intro",
            "speech_text": "hello" * 50,
            "kokoro_voice": "af_heart",
            "avatar_path": "avatar.png",
            "use_teacache": True,
        },
    )

    captured = {"command": None, "cwd": None, "write_path": None, "cleanup_path": None}

    monkeypatch.setattr(
        cli,
        "_write_json",
        lambda path, payload: captured.update({"write_path": path, "payload": payload}),
    )
    monkeypatch.setattr(
        cli,
        "_run_command_streaming",
        lambda command, cwd: captured.update({"command": command, "cwd": cwd}),
    )
    monkeypatch.setattr(cli, "_ensure_kokoro_weights", lambda _repo_dir: None)
    monkeypatch.setattr(
        cli.shutil,
        "rmtree",
        lambda path: captured.update({"cleanup_path": path}),
    )

    output_path = tmp_path / "output.mp4"
    data_path = tmp_path / "data.json"
    work_dir = tmp_path / "work"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "--job-id",
            "job123",
            "--output",
            str(output_path),
            "--data",
            str(data_path),
            "--work-dir",
            str(work_dir),
        ],
    )

    cli.main()

    assert captured["write_path"] == os.path.join(str(work_dir), "cond.json")
    assert captured["cwd"].endswith("video_generators/multitalk")
    assert captured["cleanup_path"] == str(work_dir)

    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1].endswith("generate_multitalk.py")
    assert "--audio_mode" in command and command[command.index("--audio_mode") + 1] == "tts"
    assert "--max_frames_num" in command

    frame_num = int(command[command.index("--frame_num") + 1])
    assert 33 <= frame_num <= 81
    assert (frame_num - 1) % 4 == 0


def test_main_uses_local_audio_mode_when_audio_file_is_supplied(monkeypatch, tmp_path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    wav2vec_dir = tmp_path / "wav2vec"
    ckpt_dir.mkdir()
    wav2vec_dir.mkdir()

    monkeypatch.setattr(cli.config, "CKPT_DIR", str(ckpt_dir))
    monkeypatch.setattr(cli.config, "WAV2VEC_DIR", str(wav2vec_dir))

    monkeypatch.setattr(
        cli,
        "_load_json",
        lambda _path: {
            "video_prompt": "intro",
            "avatar_path": "avatar.png",
        },
    )
    monkeypatch.setattr(cli, "_estimate_audio_duration_seconds", lambda _path: 2.0)

    captured = {"command": None, "payload": None, "ensure_called": False}

    monkeypatch.setattr(
        cli,
        "_write_json",
        lambda path, payload: captured.update({"write_path": path, "payload": payload}),
    )
    monkeypatch.setattr(
        cli,
        "_run_command_streaming",
        lambda command, cwd: captured.update({"command": command, "cwd": cwd}),
    )
    monkeypatch.setattr(
        cli,
        "_ensure_kokoro_weights",
        lambda _repo_dir: captured.update({"ensure_called": True}),
    )
    monkeypatch.setattr(cli.shutil, "rmtree", lambda path: None)

    output_path = tmp_path / "output.mp4"
    data_path = tmp_path / "data.json"
    work_dir = tmp_path / "work"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "--job-id",
            "job123",
            "--output",
            str(output_path),
            "--data",
            str(data_path),
            "--audio",
            str(tmp_path / "voice.wav"),
            "--work-dir",
            str(work_dir),
        ],
    )

    cli.main()

    assert captured["payload"]["cond_audio"]["person1"] == str(tmp_path / "voice.wav")
    assert "tts_audio" not in captured["payload"]
    assert captured["ensure_called"] is False

    command = captured["command"]
    assert "--audio_mode" in command
    assert command[command.index("--audio_mode") + 1] == "localfile"


def test_main_uses_audio_url_when_provided(monkeypatch, tmp_path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    wav2vec_dir = tmp_path / "wav2vec"
    ckpt_dir.mkdir()
    wav2vec_dir.mkdir()

    monkeypatch.setattr(cli.config, "CKPT_DIR", str(ckpt_dir))
    monkeypatch.setattr(cli.config, "WAV2VEC_DIR", str(wav2vec_dir))

    monkeypatch.setattr(
        cli,
        "_load_json",
        lambda _path: {
            "video_prompt": "intro",
            "avatar_path": "avatar.png",
        },
    )
    monkeypatch.setattr(
        cli,
        "_resolve_input_audio_path",
        lambda **kwargs: os.path.join(str(tmp_path), "normalized.wav"),
    )
    monkeypatch.setattr(cli, "_estimate_audio_duration_seconds", lambda _path: 1.5)

    captured = {"command": None, "payload": None}

    monkeypatch.setattr(
        cli,
        "_write_json",
        lambda path, payload: captured.update({"write_path": path, "payload": payload}),
    )
    monkeypatch.setattr(
        cli,
        "_run_command_streaming",
        lambda command, cwd: captured.update({"command": command, "cwd": cwd}),
    )
    monkeypatch.setattr(cli, "_ensure_kokoro_weights", lambda _repo_dir: None)
    monkeypatch.setattr(cli.shutil, "rmtree", lambda path: None)

    output_path = tmp_path / "output.mp4"
    data_path = tmp_path / "data.json"
    work_dir = tmp_path / "work"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "--job-id",
            "job123",
            "--output",
            str(output_path),
            "--data",
            str(data_path),
            "--audio-url",
            "https://example.com/voice.mp3",
            "--work-dir",
            str(work_dir),
        ],
    )

    cli.main()

    assert captured["payload"]["cond_audio"]["person1"] == os.path.join(str(tmp_path), "normalized.wav")
    command = captured["command"]
    assert command[command.index("--audio_mode") + 1] == "localfile"
