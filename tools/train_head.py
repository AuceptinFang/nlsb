"""用自己的标注重新训练分类头（src/nailong/head.npz）。骨干网络不动，只训练 768 维特征上的逻辑回归。

用法:
  python tools/train_head.py labels.csv [--out head.npz] [--lam 30] [--thr 0.5]

labels.csv 列: path,label[,split,id]   label: 1=奶龙 0=其他；split 列不用，改为 5 折交叉验证报告准确度。
特征缓存到 labels.csv 同目录的 <labels>.emb.npz（按路径复用，加新图只算新的）。
训练时每张图用：首帧、首帧水平翻转、动图另外 3 个均匀采样帧。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nailong.model import FRAMES, Classifier  # noqa: E402


def embed(clf, x):
    h = clf.sess.run(None, {clf.inp: x[None]})[0][0]
    e = np.concatenate([h[0], h[1:].mean(0)])
    return e / np.linalg.norm(e)


def features(clf, path):
    """返回 (FRAMES + 1, 768)：首帧、翻转首帧、其余帧（静图或帧不够时重复首帧）。"""
    fr = clf.frames(path)
    es = [embed(clf, x) for x in fr]
    flip = embed(clf, np.ascontiguousarray(fr[0][:, :, ::-1]))
    es += [es[0]] * (FRAMES - len(es))
    return np.stack([es[0], flip, *es[1:]])


def fit_lr(X, y, lam, iters=30):
    """牛顿法求解 L2 正则逻辑回归（截距不正则）。"""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        step = np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + R, Xb.T @ (p - y) + R @ w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def train(F, y, lam):
    """F: (N, FRAMES + 1, 768)。返回折叠了标准化的 (a, b)：分数 = sigmoid(e·a + b)。"""
    A = F.reshape(-1, F.shape[2])
    Y = np.repeat(y, F.shape[1])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    w = fit_lr((A - mu) / sd, Y, lam)
    return w[:-1] / sd, w[-1] - (mu / sd) @ w[:-1]


def score(a, b, F):
    """推理时只看首帧和其余帧（不翻转），取最高分。"""
    G = np.concatenate([F[:, :1], F[:, 2:]], 1)
    return (1 / (1 + np.exp(-(G @ a + b)))).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src/nailong/head.npz"))
    ap.add_argument("--lam", type=float, default=30)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.labels)))
    cache_path = Path(a.labels).with_suffix(".emb.npz")
    cache = {}
    if cache_path.exists():
        c = np.load(cache_path)
        cache = dict(zip(c["paths"], c["F"]))
    clf = Classifier(threads=4, tolerant=True)
    F, y = [], []
    for k, r in enumerate(rows):
        if r["path"] not in cache:
            try:
                cache[r["path"]] = features(clf, r["path"])
            except Exception as e:  # noqa: BLE001
                print(f"跳过 {r['path']}: {e}", file=sys.stderr)
                continue
        F.append(cache[r["path"]])
        y.append(int(r["label"]))
        if k % 200 == 199:
            print(f"  特征 {k + 1}/{len(rows)}", flush=True)
    np.savez(cache_path, paths=list(cache), F=np.stack(list(cache.values())))
    F, y = np.stack(F), np.array(y)
    print(f"{len(y)} 张，奶龙 {y.sum()}")

    fold = np.random.default_rng(0).permutation(len(y)) % 5
    s = np.zeros(len(y))
    for k in range(5):
        s[fold == k] = score(*train(F[fold != k], y[fold != k], a.lam), F[fold == k])
    for t in sorted({0.2, 0.3, a.thr, 0.7}):
        tp, fp, fn = ((s >= t) & (y == 1)).sum(), ((s >= t) & (y == 0)).sum(), ((s < t) & (y == 1)).sum()
        print(f"5 折交叉验证 阈值 {t:.2f}：精确率 {tp / max(tp + fp, 1):.1%} 召回率 {tp / max(tp + fn, 1):.1%}"
              f"（误判 {fp}，漏判 {fn}）")
    w, b = train(F, y, a.lam)
    np.savez(a.out, a=w.astype(np.float32), b=np.float32(b), thr=np.float32(a.thr))
    print(f"已写入 {a.out}（全部数据训练，λ={a.lam}，阈值 {a.thr}）")


if __name__ == "__main__":
    main()
