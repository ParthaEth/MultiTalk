"""Unit tests for multitalk readiness healthcheck logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(module_name: str, filename: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


healthcheck = _load_module("multitalk_healthcheck_for_tests", "healthcheck.py")


pytestmark = pytest.mark.unit


def test_collect_weight_stats_tracks_non_empty_and_empty_files(tmp_path) -> None:
    (tmp_path / "good.safetensors").write_bytes(b"abc")
    (tmp_path / "empty.pt").write_bytes(b"")
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

    stats = healthcheck._collect_weight_stats(str(tmp_path), (".safetensors", ".pt"))

    assert stats["non_empty_count"] == 1
    assert len(stats["empty_files"]) == 1
    assert stats["empty_files"][0].endswith("empty.pt")


def test_healthcheck_ok_when_required_files_and_weights_exist(tmp_path, monkeypatch) -> None:
    repo_dir = tmp_path / "multitalk"
    repo_dir.mkdir()

    for filename in ("cli.py", "generate_multitalk.py", "base_tts_template.json"):
        (repo_dir / filename).write_text("{}", encoding="utf-8")

    (repo_dir / "weights" / "wan").mkdir(parents=True)
    (repo_dir / "weights" / "wav2vec").mkdir(parents=True)
    (repo_dir / "weights" / "kokoro" / "voices").mkdir(parents=True)

    (repo_dir / "weights" / "wan" / "model.safetensors").write_bytes(b"x")
    (repo_dir / "weights" / "wav2vec" / "model.bin").write_bytes(b"x")
    (repo_dir / "weights" / "kokoro" / "model.pt").write_bytes(b"x")
    (repo_dir / "weights" / "kokoro" / "voices" / "af_heart.pt").write_bytes(b"x")

    (repo_dir / "config.py").write_text(
        "\n".join(
            [
                "CKPT_DIR = './weights/wan'",
                "WAV2VEC_DIR = './weights/wav2vec'",
                "KOKORO_DIR = './weights/kokoro'",
                "TTS_VOICE = './weights/kokoro/voices/af_heart.pt'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(healthcheck, "_repo_dir", lambda: str(repo_dir))

    result = healthcheck.healthcheck()

    assert result["ok"] is True
    assert result["errors"] == []


def test_healthcheck_reports_missing_config_paths(tmp_path, monkeypatch) -> None:
    repo_dir = tmp_path / "multitalk"
    repo_dir.mkdir()

    for filename in ("cli.py", "generate_multitalk.py", "base_tts_template.json"):
        (repo_dir / filename).write_text("{}", encoding="utf-8")

    (repo_dir / "config.py").write_text(
        "\n".join(
            [
                "CKPT_DIR = './weights/missing_ckpt'",
                "WAV2VEC_DIR = './weights/missing_wav2vec'",
                "KOKORO_DIR = './weights/missing_kokoro'",
                "TTS_VOICE = './weights/missing_kokoro/voices/af_heart.pt'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(healthcheck, "_repo_dir", lambda: str(repo_dir))
    result = healthcheck.healthcheck()

    assert result["ok"] is False
    assert any("CKPT_DIR" in error for error in result["errors"])
    assert any("WAV2VEC_DIR" in error for error in result["errors"])
    assert any("KOKORO_DIR" in error for error in result["errors"])
    assert any("TTS_VOICE" in error for error in result["errors"])
