"""raw 결과 → 비교표(CSV/MD) + 관계별 분석 + paired bootstrap(vs TransE) + 그림.

    python scripts/report.py --run results/rerun
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.stats import paired_bootstrap  # noqa: E402

REL_EN = {"type": "type", "소속": "belongsTo", "포함": "contains", "외부인용": "citesExternal",
          "인용": "cites", "소관기관": "competentAuthority", "관할법령": "governingLaw",
          "위임": "delegates", "위임받음": "delegatedBy", "상위법령": "parentLaw",
          "하위법령": "childLaw", "발령기관": "issuedBy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/rerun")
    ap.add_argument("--ref", default="TransE", help="bootstrap 기준 모델")
    args = ap.parse_args()
    run = ROOT / args.run
    runs = defaultdict(list)
    for p in sorted((run / "raw").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_rr"] = np.load(p.with_suffix(".npz"))["rr"]
        runs[r["model"]].append(r)
    if not runs:
        sys.exit("raw 결과가 없습니다. scripts/run_benchmark.py 를 먼저 실행하세요.")
    order = [m for m in ["TransE", "RotatE", "DistMult", "ComplEx", "TKRL", "TransC", "TransO",
                         "Ours", "OursC"] if m in runs] + [m for m in runs if m not in
                         ["TransE", "RotatE", "DistMult", "ComplEx", "TKRL", "TransC", "TransO", "Ours", "OursC"]]

    def mean(m, *path):
        vals = []
        for r in runs[m]:
            x = r
            for k in path:
                x = x[k]
            vals.append(x)
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    for m in order:
        row = {"model": m, "n_seeds": len(runs[m])}
        row["MRR"], row["MRR_std"] = mean(m, "overall", "mrr")
        for k in ["hits@1", "hits@3", "hits@10"]:
            row[k.replace("hits", "Hits")] = mean(m, "overall", k)[0]
        row["dir_MRR"] = mean(m, "groups", "directional", "mrr")[0]
        row["hier_MRR"] = mean(m, "groups", "hierarchy", "mrr")[0]
        if args.ref in runs and m != args.ref:
            a = np.mean([r["_rr"] for r in runs[m]], axis=0)
            b = np.mean([r["_rr"] for r in runs[args.ref]], axis=0)
            bs = paired_bootstrap(a, b, n_boot=1000)
            row[f"dMRR_vs_{args.ref}"], row["p_value"] = bs["delta"], bs["p_value"]
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (run / "main_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    L = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    L += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    # 관계별 MRR
    rels = sorted({rel for m in order for rel in runs[m][0]["per_relation"]},
                  key=lambda x: list(REL_EN).index(x) if x in REL_EN else 99)
    L += ["", "## 관계별 MRR (seed 평균)", "", "| 관계 | n | " + " | ".join(order) + " |",
          "|---|---|" + "---|" * len(order)]
    for rel in rels:
        n = runs[order[0]][0]["per_relation"].get(rel, {}).get("n", 0)
        L.append(f"| {rel} ({REL_EN.get(rel, rel)}) | {n} | " + " | ".join(
            f"{statistics.mean(r['per_relation'].get(rel, {'mrr': 0})['mrr'] for r in runs[m]):.3f}"
            for m in order) + " |")
    (run / "report.md").write_text("# 베이스라인 비교\n\n" + "\n".join(L) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["MRR"] for r in rows], 0.4, yerr=[r["MRR_std"] for r in rows], label="MRR", capsize=3)
    ax.bar(x + 0.2, [r["Hits@10"] for r in rows], 0.4, label="Hits@10")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in rows])
    ax.set_title("Link prediction on Defense-Law KG (test, filtered)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "comparison.png", dpi=150)
    print("\n".join(L[: len(rows) + 2]))
    print(f"→ {args.run}/main_results.csv, report.md, comparison.png")


if __name__ == "__main__":
    main()
