"""在标注数据上搜索规则参数。

用法:
  python tools/tune.py labels.csv [--final]

labels.csv 列: path,label,split[,id]   label: 1=奶龙 0=其他   split: tune/val/test
- 在 tune 上做坐标下降（多次随机重启），在 val 上挑最好的一组；
- test 只在加 --final 时评估一次，调参过程不看它。
特征会缓存到 labels.csv 同目录的 features.cache.json。
"""
from __future__ import annotations

import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nailong.features import extract  # noqa: E402
from nailong.score import DEFAULT_PARAMS, score  # noqa: E402

# 每个参数的候选取值
SPACE = {
    "gate": [0.1, 0.12, 0.15, 0.2], "gate_o": [0, 1],
    "g0": [0.03, 0.06, 0.1], "g1": [0.2, 0.3, 0.45], "w_g": [2, 3, 4],
    "w_pair": [0, 0.5, 1, 1.5],
    "gl": [0.4, 0.5, 0.6, 0.7], "w_glint": [0, 0.5, 1, 1.5, 2],
    "ye": [0.3, 0.5, 0.7], "w_ye": [0, 0.5, 1, 1.5],
    "sc": [0.3, 0.4, 0.5, 0.6], "w_sc": [0, 0.5, 1, 1.5, 2],
    "hue": [29, 30, 31, 32], "w_hue": [0, 0.5, 1, 1.5],
    "asp": [0.95, 1.05, 1.15], "w_asp": [0, 0.5, 1, 1.5],
    "red": [0.005, 0.01, 0.02, 0.04], "w_red": [0, 0.5, 1, 2],
    "out": [0.25, 0.35, 0.5], "w_out": [0, 0.5, 1, 1.5],
    "tex": [4.5, 5, 6, 7, 8], "w_tex": [0, 1, 2, 3],
    "e_lo": [0.03, 0.04, 0.05, 0.06], "e_hi": [0.1, 0.13, 0.16], "w_e": [0, 0.5, 1, 1.5],
    "belly": [0.05, 0.1, 0.15, 0.25, 0.4], "w_belly": [0, 0.5, 1, 1.5],
    "shade": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "w_shade": [0, 0.5, 1, 1.5],
    "orange": [0.05, 0.1, 0.15, 0.25], "w_orange": [0, 0.5, 1, 1.5, 2],
    "mouth": [0.005, 0.01, 0.02, 0.04, 0.08], "w_mouth": [0, 0.5, 1, 1.5, 2],
    "tex2": [6, 7, 8, 9, 11, 13], "w_tex2": [0, 0.5, 1, 2, 3],
    "small": [48, 64, 80, 120], "w_small": [0, 0.5, 1, 2],
    "thr": [2, 2.5, 3, 3.5, 4, 4.5, 5],
}
BETA = 1.0  # Fβ：β>1 偏向召回


def _feat(path):
    try:
        return path, extract(path)
    except Exception as e:  # noqa: BLE001
        return path, None


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    cache_path = csv_path.with_name("features.cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [r["path"] for r in rows if r["path"] not in cache]
    if todo:
        with ProcessPoolExecutor(4) as ex:
            for p, f in ex.map(_feat, todo, chunksize=16):
                cache[p] = f
        cache_path.write_text(json.dumps(cache))
    return [(cache[r["path"]], int(r["label"]), r["split"], r) for r in rows if cache.get(r["path"])]


def metrics(data, p, split):
    tp = fp = fn = 0
    for f, y, s, _ in data:
        if s != split:
            continue
        pred = score(f, p) >= p["thr"]
        tp += pred and y
        fp += pred and not y
        fn += (not pred) and y
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = BETA * BETA
    f1 = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, prec=round(prec, 3), rec=round(rec, 3), f1=round(f1, 3))


def coordinate_descent(data, start, rounds=4):
    p = dict(start)
    best = metrics(data, p, "tune")["f1"]
    for _ in range(rounds):
        improved = False
        keys = list(SPACE)
        random.shuffle(keys)
        for k in keys:
            for v in SPACE[k]:
                q = dict(p, **{k: v})
                if q["g1"] <= q["g0"] or q["e_hi"] <= q["e_lo"]:
                    continue
                f1 = metrics(data, q, "tune")["f1"]
                if f1 > best + 1e-9:
                    best, p, improved = f1, q, True
        if not improved:
            break
    return p, best


def main():
    global BETA
    csv_path = Path(sys.argv[1])
    if "--beta" in sys.argv:
        BETA = float(sys.argv[sys.argv.index("--beta") + 1])
    data = load(csv_path)
    print(f"样本 {len(data)}：", {s: sum(1 for _, y, x, _ in data if x == s) for s in ("tune", "val", "test")})
    print("默认参数  tune", metrics(data, DEFAULT_PARAMS, "tune"), "val", metrics(data, DEFAULT_PARAMS, "val"))
    random.seed(0)
    results = []
    for i in range(int(sys.argv[sys.argv.index("--restarts") + 1]) if "--restarts" in sys.argv else 12):
        start = dict(DEFAULT_PARAMS) if i == 0 else {k: random.choice(v) for k, v in SPACE.items()}
        p, f1 = coordinate_descent(data, start)
        results.append((metrics(data, p, "val")["f1"], f1, p))
        print(f"  重启 {i:2d}: tune F1 {f1:.3f}  val F1 {results[-1][0]:.3f}")
    val_f1, tune_f1, p = max(results, key=lambda r: (r[0], r[1]))
    print("选中参数:", json.dumps(p, ensure_ascii=False))
    print("tune", metrics(data, p, "tune"))
    print("val ", metrics(data, p, "val"))
    csv_path.with_name("params.json").write_text(json.dumps(p, ensure_ascii=False, indent=1))
    if "--final" in sys.argv:
        print("TEST", metrics(data, p, "test"))


if __name__ == "__main__":
    main()
