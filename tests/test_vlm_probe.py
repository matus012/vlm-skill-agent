import json
import subprocess
import uuid
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "tests"))
import vlm_probe  # noqa: E402
from mock_openai import MockServer, tool_call_msg  # noqa: E402

OK_NAV = tool_call_msg("navigate_to", {"place": "ward"})


@pytest.fixture
def img(tmp_path: Path) -> Path:
    p = tmp_path / "f.png"
    Image.new("RGB", (640, 480), (200, 30, 30)).save(p)
    return p


def run(script: list[tuple[int, Any]], img: Path, tmp_path: Path, n: int = 1, require: int = 1,
        thinking: str = "off") -> tuple[int, list[dict[str, Any]], MockServer]:
    srv = MockServer(script)
    out = tmp_path / f"o{uuid.uuid4().hex}.jsonl"
    try:
        rc = vlm_probe.main(["--image", str(img), "--base-url", srv.base_url, "--n", str(n),
                             "--require", str(require), "--thinking", thinking, "--out", str(out)])
    finally:
        srv.close()
    return rc, [json.loads(line) for line in out.read_text().splitlines()], srv


def test_valid_calls_pass(img: Path, tmp_path: Path) -> None:
    for msg in [OK_NAV, tool_call_msg("look_around", {}),
                tool_call_msg("report", {"answer": "ward", "photo": 1, "note": "x"})]:
        rc, recs, _ = run([(200, msg)], img, tmp_path)
        assert rc == 0 and recs[0]["valid"] and recs[0]["status"] == 200 and recs[0]["finish_reason"] == "tool_calls"
        assert recs[0]["usage"]["total_tokens"] == 15 and recs[0]["latency_s"] > 0


@pytest.mark.parametrize("msg", [
    tool_call_msg("fly_away", {}),                                          # wrong tool name
    tool_call_msg("navigate_to", {"place": "kitchen"}),                     # bad enum
    tool_call_msg("navigate_to", {"place": "ward", "speed": 2}),            # extra field
    tool_call_msg("navigate_to", {}),                                       # missing required
    tool_call_msg("report", {"answer": "ward", "photo": "one", "note": "x"}),  # wrong type
    tool_call_msg("look_around", {"x": 1}),                                 # extra field, no-arg tool
    tool_call_msg("navigate_to", "not json {"),                             # non-JSON arguments
    {"role": "assistant", "content": "I would go to the ward."},            # no tool call
], ids=["name", "enum", "extra", "missing", "type", "extra-noarg", "nonjson", "notool"])
def test_invalid_calls_fail(msg: dict[str, Any], img: Path, tmp_path: Path) -> None:
    rc, recs, _ = run([(200, msg)], img, tmp_path)
    assert rc == 1 and not recs[0]["valid"]


def test_http_500_recorded_not_crash(img: Path, tmp_path: Path) -> None:
    rc, recs, _ = run([(500, None)], img, tmp_path)
    assert rc == 1 and recs[0]["status"] == 500 and not recs[0]["valid"] and recs[0]["latency_s"] is not None


def test_one_valid_among_calls_counts(img: Path, tmp_path: Path) -> None:
    bad = tool_call_msg("fly_away", {})
    assert vlm_probe.verdict(bad["tool_calls"] + OK_NAV["tool_calls"]) == (True, "navigate_to")


def test_require_exit_codes(img: Path, tmp_path: Path) -> None:
    script: list[tuple[int, Any]] = [(200, OK_NAV)] * 9 + [(200, tool_call_msg("fly_away", {}))]
    rc9, recs, _ = run(script, img, tmp_path, n=10, require=9)
    assert sum(r["valid"] for r in recs) == 9 and rc9 == 0
    assert run(script, img, tmp_path, n=10, require=10)[0] == 1
    assert run([(200, OK_NAV)] * 3, img, tmp_path, n=3, require=3)[0] == 0


@pytest.mark.parametrize("thinking,flag", [("on", True), ("off", False)])
def test_thinking_flag_and_image_sent(thinking: str, flag: bool, img: Path, tmp_path: Path) -> None:
    _, _, srv = run([(200, OK_NAV)], img, tmp_path, thinking=thinking)
    body = srv.requests[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": flag}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert {t["function"]["name"] for t in body["tools"]} == {"navigate_to", "look_around", "report"}


def test_percentile_and_median() -> None:
    xs = [float(i) for i in range(1, 11)]
    assert vlm_probe.percentile(xs, 0.9) == 9.0
    assert vlm_probe.percentile([5.0], 0.9) == 5.0
    assert vlm_probe.percentile([3.0, 1.0, 2.0], 0.9) == 3.0


def test_shell_scripts_parse() -> None:
    for s in ["vlm_up.sh", "vlm_down.sh", "serve_vllm.sbatch"]:
        subprocess.run(["bash", "-n", str(ROOT / "hpc" / s)], check=True)


# ---------- API key (design notes: vLLM --api-key from Phase 3; added in task 002) ----------
def run_key(img: Path, tmp_path: Path, key: str | None, served: str) -> tuple[int, list[dict[str, Any]], MockServer]:
    srv = MockServer([(200, OK_NAV)], api_key=served)
    out = tmp_path / f"o{uuid.uuid4().hex}.jsonl"
    try:
        rc = vlm_probe.main(["--image", str(img), "--base-url", srv.base_url, "--n", "1", "--require", "1",
                             "--out", str(out)] + (["--api-key", key] if key is not None else []))
    finally:
        srv.close()
    return rc, [json.loads(line) for line in out.read_text().splitlines()], srv


def test_api_key_right_passes(img: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    rc, recs, srv = run_key(img, tmp_path, "s3cret", "s3cret")
    assert rc == 0 and recs[0]["valid"] and len(srv.requests) == 1
    monkeypatch.setenv("VLM_API_KEY", "s3cret")                    # env default
    assert run_key(img, tmp_path, None, "s3cret")[0] == 0


@pytest.mark.parametrize("key", ["wrong", None], ids=["wrong", "missing"])
def test_api_key_wrong_or_missing_is_failed_call(key: str | None, img: Path, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    rc, recs, srv = run_key(img, tmp_path, key, "s3cret")
    assert rc == 1 and recs[0]["status"] == 401 and not recs[0]["valid"] and not srv.requests


def test_vlm_up_models_check_sends_key_from_stdin() -> None:
    """The exact `models()` line of hpc/vlm_up.sh: key as a header read from stdin; 401 without it; /health open."""
    line = next(x for x in (ROOT / "hpc" / "vlm_up.sh").read_text().splitlines() if x.startswith("models()"))
    srv = MockServer([(200, OK_NAV)], api_key="k")
    try:
        for key, rc in [("k", 0), ("bad", 22), ("", 22)]:
            r = subprocess.run(["bash", "-c", line + "\nmodels"], env={"BASE": srv.base_url, "KEY": key,
                                                                       "PATH": "/usr/bin:/bin"})
            assert r.returncode == rc, key
        code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", srv.base_url + "/models"],
                              capture_output=True, text=True).stdout
        health = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                                 srv.base_url.removesuffix("/v1") + "/health"], capture_output=True, text=True).stdout
        assert (code, health) == ("401", "200")
    finally:
        srv.close()


def test_key_never_on_a_command_line() -> None:
    up = (ROOT / "hpc" / "vlm_up.sh").read_text()
    sb = (ROOT / "hpc" / "serve_vllm.sbatch").read_text()
    code = [x for x in (up + sb).splitlines() if not x.lstrip().startswith("#")]
    assert not [x for x in code if "--export" in x or "--api-key" in x]
    assert 'cat > $KEYTMP" <<<"$KEY"' in up and "-H @-" in up
    assert "APPTAINERENV_VLLM_API_KEY" in sb and "--cleanenv" in sb
