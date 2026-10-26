"""KGE 베이스라인 7종 학습·평가 (국방 법령 KG v1.0).

    python scripts/run_benchmark.py                                   # 7모델 × 5seed × 150epoch (약 40~60분, CPU)
    python scripts/run_benchmark.py --models TransE RotatE --seeds 42 # 일부만
    python scripts/run_benchmark.py --epochs 5 --seeds 42 --out results/smoke   # 1~2분 스모크 테스트

결과 : <out>/raw/<model>_s<seed>.json (지표) + .npz (트리플별 reciprocal rank, bootstrap 용)
       이미 있는 결과는 건너뛰므로 중단 후 같은 명령으로 이어서 실행할 수 있다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.data import load_kg  # noqa: E402
from kgekit.evaluate import evaluate  # noqa: E402
from kgekit.train import config_for, train_chunk  # noqa: E402

BASELINES = ["TransE", "RotatE", "DistMult", "ComplEx", "TKRL", "TransC", "TransO"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=BASELINES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/rerun")
    args = ap.parse_args()

    kg = load_kg(ROOT / args.data)
    print(f"KG: |E|={kg.num_ent} |R|={kg.num_rel} train={len(kg.train)} valid={len(kg.valid)} test={len(kg.test)}")
    raw = ROOT / args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        for s in args.seeds:
            out = raw / f"{m}_s{s}.json"
            if out.exists():
                print(f"skip {out.name}")
                continue
            t0 = time.time()
            cfg = config_for(m, seed=s, epochs=args.epochs, dim=args.dim)
            model, P, hist, _ = train_chunk(kg, kg.train, cfg)
            res = evaluate(model, P, kg, kg.test)
            np.savez_compressed(raw / f"{m}_s{s}.npz", rr=res["_rr"], rel_of=res["_rel_of"],
                                dir_mask=res["_dir_mask"], hier_mask=res["_hier_mask"])
            rec = {"model": m, "seed": s, "config": cfg.__dict__,
                   **{k: v for k, v in res.items() if not k.startswith("_")}}
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            o = res["overall"]
            print(f"{m:8s} s{s}  MRR={o['mrr']:.4f}  H@1={o['hits@1']:.4f}  H@10={o['hits@10']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
