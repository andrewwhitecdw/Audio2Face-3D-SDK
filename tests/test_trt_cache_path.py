# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Regression tests for get_trt_cache_path() in audio2x.data_utils.

Bug: get_trt_cache_path() did ``open(shutil.which("trtexec"), "rb")`` without
checking for None, so when trtexec is not on PATH it crashed with a confusing
``TypeError: expected str, bytes or os.PathLike object, not NoneType`` instead
of a descriptive error. The fix raises FileNotFoundError.

Run with:
    uv run --with pytest --with numpy pytest tests/test_trt_cache_path.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio2x-common" / "scripts"))

# audio2x.common calls argparse.parse_args() at import time, which would
# choke on pytest's own command line; hide it during the import.
_argv = sys.argv
sys.argv = [_argv[0]]
try:
    from audio2x.data_utils import get_trt_cache_path
finally:
    sys.argv = _argv


def _make_onnx(tmp_path):
    onnx = tmp_path / "network.onnx"
    onnx.write_bytes(b"fake onnx model bytes")
    return onnx


def _make_fake_trtexec(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trtexec = bin_dir / "trtexec"
    trtexec.write_bytes(b"#!/bin/sh\necho fake trtexec\n")
    trtexec.chmod(0o755)
    return bin_dir


def _cmd(onnx):
    return ["/abs/path/trtexec", str(onnx), "/abs/path/model.trt", "--fp16"]


def test_missing_trtexec_raises_file_not_found(tmp_path, monkeypatch):
    """Without trtexec on PATH, must raise FileNotFoundError (not TypeError)."""
    onnx = _make_onnx(tmp_path)
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(FileNotFoundError, match="trtexec not found in PATH"):
        get_trt_cache_path(str(onnx), _cmd(onnx))


def test_cache_path_deterministic_and_arg_sensitive(tmp_path, monkeypatch):
    """Identical inputs hash identically; changing args past cmd[3] rehashes."""
    onnx = _make_onnx(tmp_path)
    bin_dir = _make_fake_trtexec(tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir))

    cmd = _cmd(onnx)
    path1 = get_trt_cache_path(str(onnx), cmd)
    path2 = get_trt_cache_path(str(onnx), cmd)
    assert path1 == path2, "cache path must be deterministic for identical inputs"

    other_cmd = cmd[:3] + ["--int8"]
    path3 = get_trt_cache_path(str(onnx), other_cmd)
    assert path3 != path1, "changing trtexec args must change the cache key"
