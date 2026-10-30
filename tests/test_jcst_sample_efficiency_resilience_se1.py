"""Supplemental postlaunch engineering checks of frozen SE1 failure paths."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import jcst_sample_efficiency_se1 as collector
import jcst_sample_efficiency_pipeline_se1 as pipeline


def test_ram_cap_rejects_excess_rss(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.resource, "getrusage",
                        lambda _: SimpleNamespace(ru_maxrss=8 * 1024 * 1024 + 1))
    with pytest.raises(MemoryError):
        collector.cap_check(tmp_path, time.monotonic() + 10)


def test_disk_cap_rejects_sparse_oversize_file(tmp_path):
    large = tmp_path / "sparse"
    with large.open("wb") as stream:
        stream.truncate(20 * 1024**3 + 1)
    try:
        with pytest.raises(OSError, match="disk cap"):
            collector.cap_check(tmp_path, time.monotonic() + 10, disk=True)
    finally:
        large.unlink()


def test_failed_stage_preserves_log_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path.parent)
    state = {"status": "RUNNING", "stages": []}
    with pytest.raises(RuntimeError, match="exited"):
        pipeline.stage(tmp_path, state, "synthetic-failure",
                       [sys.executable, "-c", "raise RuntimeError('engineering fixture')"],
                       time.monotonic() + 5)
    assert (tmp_path / "synthetic-failure.log").exists()
    assert (tmp_path / "pipeline-status.json").exists()
    assert state["stages"][0]["status"] == "FAILED"


def test_timed_out_stage_stops_only_own_child_and_preserves_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path.parent)
    state = {"status": "RUNNING", "stages": []}
    with pytest.raises(TimeoutError, match="cap"):
        pipeline.stage(tmp_path, state, "synthetic-timeout",
                       [sys.executable, "-c", "import time; time.sleep(2)"],
                       time.monotonic() + .2)
    assert (tmp_path / "synthetic-timeout.log").exists()
    assert (tmp_path / "pipeline-status.json").exists()
    assert state["stages"][0]["status"] == "INCOMPLETE"
