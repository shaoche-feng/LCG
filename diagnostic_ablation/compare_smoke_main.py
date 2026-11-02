"""The main arms' first updates must equal the separate smoke arms' (same snapshot, prompt seed and code; the smoke ran
30 updates with every start frame checked). usage: python compare_smoke_main.py SMOKE_ROOT MAIN_ROOT"""
import json
import sys
from pathlib import Path

smoke, main = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for arm in "ABC":
    s = [json.loads(l) for l in open(smoke / arm / "updates.jsonl")]
    m = [json.loads(l) for l in open(main / arm / "updates.jsonl")][:len(s)]
    si = [json.loads(l)["batches"] for l in open(smoke / arm / "indices.jsonl")]
    mi = [json.loads(l)["batches"] for l in open(main / arm / "indices.jsonl")][:len(si)]
    res[arm] = dict(updates_compared=len(s), metrics_identical=s == m, indices_identical=si == mi)
res["all_identical"] = all(v["metrics_identical"] and v["indices_identical"] for v in res.values() if isinstance(v, dict))
print(json.dumps(res, indent=1))
sys.exit(0 if res["all_identical"] else 1)
