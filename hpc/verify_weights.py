#!/usr/bin/env python3
"""Verify downloaded weights vs the Hub: shard presence (index.json), size, sha256 (LFS oid).
Run on the login node: $P/tools/venv/bin/python verify_weights.py [repo] [local_dir]. Exit 0 only if all pass."""
import os
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
d = Path(sys.argv[2] if len(sys.argv) > 2 else f"{os.environ['PROJ']}/models/{repo.split('/')[1]}")
info = HfApi().model_info(repo, files_metadata=True)
meta = {s.rfilename: s for s in info.siblings}
print(f"repo {repo} rev {info.sha}")
shards = sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values()))
missing = [s for s in shards if not (d / s).is_file()]
print(f"shards in index: {len(shards)}, present: {len(shards) - len(missing)}, missing: {missing}")

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()

files = [f for f in meta if f != ".gitattributes"]
def check(f: str) -> tuple[str, bool, str]:
    p, m = d / f, meta[f]
    if not p.is_file():
        return f, False, "missing"
    if p.stat().st_size != m.size:
        return f, False, f"size {p.stat().st_size} != {m.size}"
    if m.lfs is not None and sha(p) != m.lfs.sha256:
        return f, False, "sha256 mismatch"
    return f, True, "lfs-sha256 ok" if m.lfs else "size ok (non-LFS)"

with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(check, files))
for f, ok, why in res:
    print(("OK  " if ok else "FAIL"), f, meta[f].size, why)
tot = sum(meta[f].size for f in shards if f in meta)
nok = sum(ok for f, ok, _ in res if f in shards)
print(f"safetensors total bytes {tot} ({tot/2**30:.1f} GiB); shards sha-verified {nok}/{len(shards)}; all files ok {sum(o for _,o,_ in res)}/{len(res)}")
sys.exit(0 if not missing and all(ok for _, ok, _ in res) else 1)
