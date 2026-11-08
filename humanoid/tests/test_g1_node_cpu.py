"""g1_node CPU pinning (PERUN job 145060): every onnxruntime session gets 1 intra-op + 1 inter-op thread and no spinning
(ORT then starts no pool threads and pins none outside the sim's cpus), and repin() puts every thread back on the start
affinity."""
from __future__ import annotations

import os
import threading

import onnxruntime as ort
import pytest

import humanoid.g1_node as g

ONNX = g.EXT / "agile/agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/Velocity-G1-v0.onnx"


def threads_of(s: ort.InferenceSession) -> tuple[int, int, str | None, str | None]:
    o = s.get_session_options()

    def entry(k: str) -> str | None:
        try:
            return o.get_session_config_entry(k)
        except RuntimeError:  # ORT raises for a key that was never set
            return None
    return (o.intra_op_num_threads, o.inter_op_num_threads, entry("session.intra_op.allow_spinning"),
            entry("session.inter_op.allow_spinning"))


@pytest.mark.skipif(not ONNX.exists(), reason="AGILE policy not cloned")
def test_every_session_one_thread_no_spinning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort, "InferenceSession", ort.InferenceSession)  # undone after the test
    assert threads_of(ort.InferenceSession(str(ONNX)))[0] == 0  # ORT default: one pinned thread per core
    g.ort_one_thread()
    g.ort_one_thread()  # idempotent: no wrapper of a wrapper
    assert ort.InferenceSession.__mro__[1].__name__ == "InferenceSession"
    assert threads_of(ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])) == (1, 1, "0", "0")
    so = ort.SessionOptions()  # as leapp's export_backend passes it: caller options are kept, threads overridden
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    assert threads_of(s) == (1, 1, "0", "0")
    assert s.get_session_options().graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert threads_of(ort.InferenceSession(str(ONNX), sess_options=ort.SessionOptions())) == (1, 1, "0", "0")


def test_repin_restores_every_thread() -> None:
    want = os.sched_getaffinity(0)
    if len(want) < 2:
        pytest.skip("needs 2 cpus")
    one = {min(want)}
    moved, go = threading.Event(), threading.Event()

    def stray() -> None:  # a library thread that pinned itself elsewhere
        os.sched_setaffinity(0, one)
        moved.set()
        go.wait(5)
        got.append(os.sched_getaffinity(0))
    got: list[set[int]] = []
    th = threading.Thread(target=stray)
    th.start()
    moved.wait(5)
    r = g.repin(want)
    go.set()
    th.join(5)
    assert r["start"] == r["after_init_union"] == g.cpus(want) and r["threads_repinned"] >= 1
    assert got == [want]
    assert all(os.sched_getaffinity(int(t)) == want for t in os.listdir("/proc/self/task"))


def test_cpus_ranges() -> None:
    assert g.cpus({2, 3, 4, 7, 9, 10}) == "2-4,7,9-10" and g.cpus({5}) == "5"
