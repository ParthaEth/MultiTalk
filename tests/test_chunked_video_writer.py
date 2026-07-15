"""Unit tests for ChunkedVideoWriter and streaming trim equivalence."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "wan" / "utils" / "multitalk_utils.py"


def _ensure_stub(module_name: str, attrs: dict | None = None):
    if module_name in sys.modules:
        return
    import types

    module = types.ModuleType(module_name)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[module_name] = module


def _load_utils():
    """Load multitalk_utils without importing the full Multitalk package graph."""
    import types

    def _stub(*_args, **_kwargs):
        raise RuntimeError("dependency stub should not be called in these tests")

    # Optional heavy deps used elsewhere in this file.
    for name in ("einops", "skimage", "skimage.color"):
        try:
            importlib.import_module(name)
        except Exception:
            if name == "einops":
                _ensure_stub("einops", {"rearrange": _stub, "repeat": _stub})
            elif name == "skimage":
                _ensure_stub("skimage")
            else:
                _ensure_stub("skimage.color", {"rgb2lab": _stub, "lab2rgb": _stub})

    try:
        import xfuser  # noqa: F401
    except Exception:
        _ensure_stub("xfuser")
        _ensure_stub("xfuser.core")
        _ensure_stub(
            "xfuser.core.distributed",
            {
                "get_sequence_parallel_rank": _stub,
                "get_sequence_parallel_world_size": _stub,
                "get_sp_group": _stub,
            },
        )

    # torchvision is imported at module level but unused by ChunkedVideoWriter.
    try:
        import torchvision  # noqa: F401
    except Exception:
        _ensure_stub("torchvision")
        _ensure_stub("torchvision.utils", {"make_grid": _stub})

    spec = importlib.util.spec_from_file_location("multitalk_utils_under_test", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def _ffprobe_frame_count(path: str) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def test_chunked_video_writer_appends_multiple_chunks(tmp_path):
    out = tmp_path / "stream-temp.mp4"
    writer = utils.ChunkedVideoWriter(str(out), fps=25, quality=5)
    try:
        # Two chunks of 4 frames at tiny resolution; values in [-1, 1].
        chunk1 = torch.zeros(3, 4, 16, 16)
        chunk2 = torch.ones(3, 3, 16, 16) * 0.5
        assert writer.append(chunk1) == 4
        assert writer.append(chunk2) == 3
        assert writer.frames_written == 7
    finally:
        writer.close()

    assert out.exists()
    assert out.stat().st_size > 0
    assert _ffprobe_frame_count(str(out)) == 7


def test_chunked_video_writer_accepts_batched_tensor(tmp_path):
    out = tmp_path / "batched-temp.mp4"
    writer = utils.ChunkedVideoWriter(str(out), fps=25, quality=5)
    try:
        batched = torch.zeros(1, 3, 2, 8, 8)
        assert writer.append(batched) == 2
    finally:
        writer.close()
    assert writer.frames_written == 2
    assert _ffprobe_frame_count(str(out)) == 2


def test_streaming_trim_matches_legacy_concat_frame_count():
    """Reproduce generate()'s per-chunk trim vs full torch.cat then trim."""
    max_frames_num = 2000
    frame_num = 81
    motion_frame = 25
    miss_lengths = [7]

    # Synthetic decoded chunks matching streaming semantics.
    first = torch.randn(1, 3, frame_num, 4, 4)
    second_full = torch.randn(1, 3, frame_num, 4, 4)
    second_new = second_full[:, :, motion_frame:]

    # Legacy: cat all then trim
    legacy = torch.cat([first, second_new], dim=2)[:, :, : max_frames_num]
    legacy = legacy[:, :, : -miss_lengths[0]]

    # Streaming: trim only the final chunk, then cap by remaining budget
    frames_written = 0
    kept_chunks = []

    for is_first, arrive_last, videos in (
        (True, False, first),
        (False, True, second_full),
    ):
        if is_first:
            chunk_to_keep = videos
        else:
            chunk_to_keep = videos[:, :, motion_frame:]

        if arrive_last and max_frames_num > frame_num and sum(miss_lengths) > 0:
            trim = int(miss_lengths[0])
            chunk_to_keep = chunk_to_keep[:, :, :-trim] if trim < chunk_to_keep.shape[2] else chunk_to_keep[:, :, :0]

        remaining = int(max_frames_num) - frames_written
        if remaining <= 0:
            chunk_to_keep = chunk_to_keep[:, :, :0]
        elif chunk_to_keep.shape[2] > remaining:
            chunk_to_keep = chunk_to_keep[:, :, :remaining]

        kept_chunks.append(chunk_to_keep)
        frames_written += int(chunk_to_keep.shape[2])

    streamed = torch.cat(kept_chunks, dim=2)
    assert streamed.shape == legacy.shape
    assert torch.equal(streamed, legacy)


def test_normalize_matches_previous_save_path():
    samples = torch.linspace(-1, 1, 3 * 2 * 4 * 4).reshape(3, 2, 4, 4)
    got = utils._normalize_video_tensor_to_uint8(samples)
    expected = (samples + 1) / 2
    expected = expected.permute(1, 2, 3, 0).cpu().numpy()
    expected = np.clip(expected * 255, 0, 255).astype(np.uint8)
    assert got.shape == (2, 4, 4, 3)
    assert np.array_equal(got, expected)
