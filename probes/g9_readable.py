"""G9 readable objects (task 002): can the VLM name each task-target class in 3 cropped photos?

Input: one or more photo dirs (probes/photos.py output, sim/INTERFACES.md section 4), each paired with the truth.json
of the world variant it was taken in (`--photos D1 --truth T1 --photos D2 --truth T2`, paired in order). The cabinet
state of a photo comes from its own dir's truth.json, so an open-variant and a closed-variant dir give both states.

Selection: per group (a class, or class:state for classes with states) every photo where one instance of the class
has >= MIN_PX pixels in the cropped seg image is a candidate; 3 distinct photos are picked greedily, preferring a
not-yet-used instance, then a not-yet-used place, then more pixels.
Calls: one "List the objects you see, with colour." per selected photo (shared by every group that selected it),
plus "Is the cabinet open or closed?" per selected photo of a group with states. Thinking on, model-card sampling.
Match: probes/g9_targets.yaml synonyms, case-insensitive whole word on the answer content (never the reasoning).
Verdict: each gated group named in >= NAME_MIN of N_PHOTOS photos; state right in >= STATE_MIN of N_PHOTOS.
Exit 0 only on pass. `--select-only`: selection + contact sheets, no VLM; exit 0 iff every gated group has 3 photos.
`--selection FILE` (instead of --photos/--truth; task 008 G9 for new placements): a ready selection {group: [photo, ...]}
(probes/spot_check.py --g9-select; photo rgb/seg paths relative to FILE's dir). Every group is gated, class = the group
name before "@", no state: pass = named in >= NAME_MIN of N_PHOTOS photos (the same rule and calls as above). A group
whose photos carry "class" / "state" (task 012 H-P3: bed, cabinet:<state>) takes them from its first photo; a state
group then gets the state call and the state bar too.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from vlm_probe import SAMPLING, image_data_url  # noqa: E402

# Thresholds: task 002 -> Gates -> G9 (binding).
MIN_PX, N_PHOTOS, NAME_MIN, STATE_MIN = 1000, 3, 2, 3
LIST_PROMPT = "List the objects you see, with colour."
STATE_PROMPT = "Is the cabinet open or closed?"
TARGETS = Path(__file__).with_name("g9_targets.yaml")
LINK_ENV = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "vlm_link.env"


def link_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def synonym_regex(synonyms: list[str]) -> re.Pattern[str]:
    """Whole word, case-insensitive, optional plural s/es, multi-word synonyms joined by spaces or hyphens."""
    alts = [r"[\s-]+".join(re.escape(w) for w in s.split()) for s in synonyms]
    return re.compile(r"\b(?:" + "|".join(alts) + r")(?:s|es)?\b", re.IGNORECASE)


STATE_RE = re.compile(r"\b(not\s+(?:\*{1,2}|_)?)?(open|closed)\b", re.IGNORECASE)
FLIP = {"open": "closed", "closed": "open"}


def parse_state(text: str) -> str | None:
    """The first whole-word 'open' / 'closed' in the text; 'not open' / 'not closed' (markdown emphasis allowed) reads
    as the opposite; neither word = None (wrong). design ruling 2026-11-04, design notes a recorded deviation: the answers state the
    verdict first and often name the other state in the explanation ("If it were closed, ...")."""
    m = STATE_RE.search(text)
    if m is None:
        return None
    word = m.group(2).lower()
    return FLIP[word] if m.group(1) else word


def groups_of(targets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """group name -> {class, state, gated}; a class with states gives one group per state."""
    out = {}
    for cls, t in targets.items():
        for st in t.get("states") or [None]:
            out[f"{cls}:{st}" if st else cls] = {"class": cls, "state": st, "gated": bool(t.get("gated"))}
    return out


def candidates(sets: list[tuple[Path, Path]], targets: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """group -> one candidate per photo (the instance of that class with the most pixels, if >= MIN_PX)."""
    out: dict[str, list[dict[str, Any]]] = {g: [] for g in groups_of(targets)}
    for si, (pdir, tpath) in enumerate(sets):
        truth = json.loads(tpath.read_text())
        missing = {o["class"] for o in truth["objects"]} - set(targets)
        assert not missing, f"classes in {tpath} missing from {TARGETS.name}: {sorted(missing)}"
        by_label = {o["label"]: o for o in truth["objects"]}
        metas = sorted(pdir.glob("*/photos.json"))
        assert metas, f"no <place>/photos.json under {pdir}"
        for mpath in metas:
            meta = json.loads(mpath.read_text())
            for p in meta["photos"]:
                seg = Image.open(mpath.parent / p["seg"])
                assert seg.mode == "L", f"{mpath.parent / p['seg']}: mode {seg.mode}, expected 8-bit label image"
                best: dict[str, dict[str, Any]] = {}
                for label, px in enumerate(seg.histogram()):
                    o = by_label.get(label)
                    if o is None or px < MIN_PX:
                        continue
                    t = targets[o["class"]]
                    g = f"{o['class']}:{o['state']}" if t.get("states") else o["class"]
                    if g not in best or px > best[g]["pixels"]:
                        best[g] = {"group": g, "class": o["class"], "instance": o["id"], "pixels": px,
                                   "state": o["state"], "set": si, "place": meta["place"], "k": p["k"],
                                   "photo": f"{si}:{pdir.name}/{meta['place']}/{p['k']}",
                                   "rgb": str(mpath.parent / p["rgb"]), "seg": str(mpath.parent / p["seg"])}
                for g, c in best.items():
                    assert g in out, f"{c['instance']}: state {c['state']!r} not in {TARGETS.name} states"
                    out[g].append(c)
    return out


def select(cands: list[dict[str, Any]], n: int = N_PHOTOS) -> list[dict[str, Any]]:
    """n distinct viewpoints: never the same (place, k) twice, whatever the world variant (an open- and a closed-world
    shot of one pose are one view, design review 002 B4); prefer an unused instance, then an unused place, then more px."""
    pool, chosen = list(cands), []
    while pool and len(chosen) < n:
        inst = {c["instance"] for c in chosen}
        places = {c["place"] for c in chosen}
        best = min(pool, key=lambda c: (c["instance"] in inst, c["place"] in places, -c["pixels"], c["photo"]))
        chosen.append(best)
        pool = [c for c in pool if (c["place"], c["k"]) != (best["place"], best["k"])]
    return chosen


def group_pass(gated: bool, n_photos: int, name_hits: int, state_hits: int | None) -> bool:
    if not gated:
        return True
    ok = n_photos >= N_PHOTOS and name_hits >= NAME_MIN
    return ok and (state_hits is None or state_hits >= STATE_MIN)


def load_selection(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """(selection with rgb/seg resolved against the file's dir, groups: every one gated; class / state from the group's
    first photo when it has them, else class = the name before "@", no state)."""
    sel = json.loads(path.read_text())
    for s in sel.values():
        for c in s:
            c["rgb"], c["seg"] = str(path.parent / c["rgb"]), str(path.parent / c["seg"])
    return sel, {g: {"class": s[0].get("class", g.split("@")[0]) if s else g.split("@")[0],
                     "state": s[0].get("state") if s else None, "gated": True} for g, s in sel.items()}


def contact_sheet(sel: list[dict[str, Any]], title: str, out: Path) -> None:
    ims = [Image.open(c["rgb"]).convert("RGB") for c in sel]
    w = sum(i.width for i in ims) or 640
    h = max((i.height for i in ims), default=100)
    sheet = Image.new("RGB", (w, h + 40), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), f"G9 {title}: {len(sel)} photo(s)", fill="black")
    x = 0
    for im, c in zip(ims, sel):
        sheet.paste(im, (x, 40))
        d.text((x + 4, 20), f"{c['photo']}  {c['instance']}  {c['pixels']} px", fill="black")
        x += im.width
    sheet.save(out)


def one_call(client: OpenAI, model: str, url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    sp = SAMPLING[True]
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": prompt}]}]
    rec: dict[str, Any] = {"status": None, "content": "", "reasoning_chars": None, "finish_reason": None,
                           "usage": None, "error": None}
    t0 = time.perf_counter()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=msgs, max_tokens=max_tokens, temperature=sp["temperature"], top_p=sp["top_p"],
            presence_penalty=sp["presence_penalty"],
            extra_body={"top_k": sp["top_k"], "chat_template_kwargs": {"enable_thinking": True}})
        rec["status"] = raw.status_code
        resp = raw.parse()
        ch = resp.choices[0]
        rec["usage"] = resp.usage.model_dump() if resp.usage else None
        rec["finish_reason"] = ch.finish_reason
        rec["content"] = ch.message.content or ""
        extra = ch.message.model_extra or {}
        rec["reasoning_chars"] = len(extra.get("reasoning") or extra.get("reasoning_content") or "")
    except Exception as e:  # network/HTTP error (incl. 401) becomes a recorded failed call, never a crash
        rec["status"] = getattr(e, "status_code", None) or 0
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["latency_s"] = time.perf_counter() - t0
    return rec


def main(argv: list[str] | None = None) -> int:
    env = link_env(LINK_ENV)
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, action="append", default=[], help="photos.py output dir (repeat)")
    ap.add_argument("--truth", type=Path, action="append", default=[], help="truth.json of that dir's world (repeat)")
    ap.add_argument("--selection", type=Path, help="ready selection JSON instead of --photos/--truth (module doc)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", type=Path, default=TARGETS)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--state-only", action="store_true",
                    help="only the state calls of the groups with states (G9 confirmation run, a recorded deviation); "
                         "verdict = state hits only, names come from the full run")
    ap.add_argument("--base-url", default=env.get("BASE_URL") or "http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default=None, help=f"default: env VLM_API_KEY, else API_KEY in {LINK_ENV}, else EMPTY")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8, help="concurrent VLM calls (vLLM batches them)")
    a = ap.parse_args(argv)
    if len(a.photos) != len(a.truth):
        ap.error("give one --truth per --photos, in the same order")
    if bool(a.selection) == bool(a.photos):
        ap.error("give either --selection or --photos/--truth")
    targets = yaml.safe_load(a.targets.read_text())["classes"]
    if a.selection:
        sel, groups = load_selection(a.selection)
    else:
        groups = groups_of(targets)
        sel = {g: select(c) for g, c in candidates(list(zip(a.photos, a.truth)), targets).items()}

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "g9_selection.json").write_text(json.dumps(sel, indent=1))
    for g, s in sel.items():
        contact_sheet(s, g, a.out / f"g9_sheet_{g.replace(':', '_')}.png")
    print(f"{'group':24} {'gate':5} photos (instance px)")
    for g, s in sel.items():
        print(f"{g:24} {'yes' if groups[g]['gated'] else 'info':5} {len(s)}/{N_PHOTOS}  "
              + "  ".join(f"{c['photo']} ({c['instance']} {c['pixels']})" for c in s))
    if a.select_only:
        short = [g for g, s in sel.items() if groups[g]["gated"] and len(s) < N_PHOTOS]
        print("selection " + ("OK" if not short else f"SHORT: {short}"))
        return 0 if not short else 1

    key = a.api_key or os.environ.get("VLM_API_KEY") or env.get("API_KEY") or "EMPTY"
    client = OpenAI(base_url=a.base_url, api_key=key, timeout=300, max_retries=0)
    pats = {cls: synonym_regex(t["synonyms"]) for cls, t in targets.items()}
    if a.state_only:
        sel = {g: s for g, s in sel.items() if groups[g]["state"] is not None}
    photos = {c["photo"]: c for s in sel.values() for c in s}
    urls = {p: image_data_url(Path(c["rgb"])) for p, c in photos.items()}
    listed = {} if a.state_only else photos
    rows: list[dict[str, Any]] = []
    calls = 0

    def log(rec: dict[str, Any], prompt: str, photo: str, evals: list[dict[str, Any]]) -> None:
        for ev in evals:
            rows.append({"call": calls, "photo": photo, "rgb": photos[photo]["rgb"], "prompt": prompt, **ev,
                         "content": rec["content"], "reasoning_chars": rec["reasoning_chars"],
                         "latency_s": round(rec["latency_s"], 3), "status": rec["status"],
                         "finish_reason": rec["finish_reason"], "usage": rec["usage"], "error": rec["error"]})

    state_keys = [(g, c["photo"]) for g, s in sel.items() if groups[g]["state"] is not None for c in s]
    total, finished = len(listed) + len(state_keys), 0

    def run_all(jobs: dict[Any, tuple[str, str]], pool: ThreadPoolExecutor) -> dict[Any, dict[str, Any]]:
        # concurrent calls; results keyed by job so the logging order below stays the sequential one
        nonlocal finished
        futs: dict[Future[dict[str, Any]], Any] = {
            pool.submit(one_call, client, a.model, urls[ph], pr, a.max_tokens): k for k, (ph, pr) in jobs.items()}
        out: dict[Any, dict[str, Any]] = {}
        for fu in as_completed(futs):
            out[futs[fu]] = fu.result()
            finished += 1
            print(f"done {finished}/{total}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list_recs = run_all({p: (p, LIST_PROMPT) for p in listed}, pool)
        state_recs = run_all({k: (k[1], STATE_PROMPT) for k in state_keys}, pool)
    for p in listed:
        rec = list_recs[p]
        log(rec, LIST_PROMPT, p, [{"group": g, "class": c["class"], "instance": c["instance"], "pixels": c["pixels"],
                                    "kind": "name", "matched": bool(pats[c["class"]].search(rec["content"]))}
                                   for g, s in sel.items() for c in s if c["photo"] == p])
        calls += 1
    for g, s in sel.items():
        if groups[g]["state"] is None:
            continue
        for c in s:
            rec = state_recs[(g, c["photo"])]
            got = parse_state(rec["content"])
            log(rec, STATE_PROMPT, c["photo"], [{"group": g, "class": c["class"], "instance": c["instance"],
                                                 "pixels": c["pixels"], "kind": "state", "expected": c["state"],
                                                 "parsed": got, "matched": got == c["state"]}])
            calls += 1
    with (a.out / "g9_calls.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary: dict[str, Any] = {"model": a.model, "base_url": a.base_url, "max_tokens": a.max_tokens, "workers": a.workers,
                               "calls": calls, "state_only": a.state_only,
                               "thresholds": {"min_px": MIN_PX, "n_photos": N_PHOTOS, "name_min": NAME_MIN,
                                              "state_min": STATE_MIN}, "groups": {}}
    print(f"\n{'group':24} {'gate':5} {'name':6} {'state':6} result")
    for g, s in sel.items():
        # state-only: no name calls here (names were judged in the full run), so the verdict is the state bar alone
        name = None if a.state_only else sum(r["matched"] for r in rows if r["group"] == g and r["kind"] == "name")
        state = sum(r["matched"] for r in rows if r["group"] == g and r["kind"] == "state") \
            if groups[g]["state"] else None
        ok = group_pass(groups[g]["gated"], len(s), NAME_MIN if name is None else name, state)
        summary["groups"][g] = {**groups[g], "n_photos": len(s), "name_hits": name, "state_hits": state, "pass": ok}
        print(f"{g:24} {'yes' if groups[g]['gated'] else 'info':5} {'-' if name is None else f'{name}/{len(s)}':6} "
              f"{'-' if state is None else f'{state}/{len(s)}':6} {('PASS' if ok else 'FAIL') if groups[g]['gated'] else '-'}")
    summary["pass"] = all(v["pass"] for v in summary["groups"].values())
    lat = list({r["call"]: r["latency_s"] for r in rows}.values())
    summary["latency_s"] = {"median": statistics.median(lat), "max": max(lat)} if lat else None
    (a.out / "g9_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"G9 {'PASS' if summary['pass'] else 'FAIL'}  ({calls} calls) -> {a.out}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
