"""Worker-node multitalk checks that validate installed runtime dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.worker_node

_RUN_WORKER_TESTS = os.getenv("VIDLINK_WORKER_TESTS", "0") == "1"
_ON_GITHUB = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
_SKIP_WORKER_TESTS = (not _RUN_WORKER_TESTS) or _ON_GITHUB

_SKIP_REASON = (
    "Worker-node tests are disabled by default. Set VIDLINK_WORKER_TESTS=1 on a GPU worker node. "
    "These tests are always skipped on GitHub Actions."
)


def _multitalk_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.skipif(_SKIP_WORKER_TESTS, reason=_SKIP_REASON)
def test_multitalk_healthcheck_succeeds_on_worker_node() -> None:
    work_dir = _multitalk_dir()
    code = (
        "import json; "
        "from healthcheck import healthcheck; "
        "result = healthcheck(); "
        "print(json.dumps(result)); "
        "raise SystemExit(0 if result.get('ok') else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, (
        "multitalk healthcheck failed on worker node. "
        f"stdout={completed.stdout} stderr={completed.stderr}"
    )


@pytest.mark.skipif(_SKIP_WORKER_TESTS, reason=_SKIP_REASON)
def test_generate_multitalk_help_runs_with_installed_dependencies() -> None:
    work_dir = _multitalk_dir()
    completed = subprocess.run(
        [sys.executable, "generate_multitalk.py", "--help"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, (
        "generate_multitalk.py --help failed; worker node dependencies may be incomplete. "
        f"stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}"
    )
