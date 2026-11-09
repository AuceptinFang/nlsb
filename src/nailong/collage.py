"""把拼图切成子图：按纯色背景上整行、整列的空隙递归切分（XY-cut）。

分类器把整张图缩到 224×224，一张 4×4 的奶龙拼图里每只只剩 40 px 左右，认不出来；切开分别打分就认得出。
不用固定网格去切：黄色角色的局部放大后很容易被认成奶龙，网格切块让难负样本误判多了几十倍。
沿空隙切出来的是一个个完整的画面，和训练时的图是同一种分布。
"""
import numpy as np
from PIL import Image

A = 256          # 在长边 A px 的缩略图上找空隙
TOL = 24         # 与背景色的最大通道差，以内算背景
EDGE = 0.8       # 四条边上至少这么多像素是背景色，才当作纯色底的拼图（照片之类不切）
GAP = 0.015      # 空隙至少占整图对应边长的比例（太小会把一行字切成单个字）
MIN_SIDE = 0.12  # 子图短边至少占整图短边的比例（太小的是头像、图标、文字）
MAX_AREA = 0.4   # 子图面积不超过整图的这个比例（再大就和整图打分重复了）
DEPTH = 6


def panels(im, max_n=16):
    """im：PIL 图。返回子图框 [(l, t, r, b)]（im 的坐标），按面积从大到小；切不出两块以上时返回 []。"""
    im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im.convert("RGBA")).convert("RGB")
    W, H = im.size
    k = max(W, H) / A
    a = np.asarray(im.resize((max(1, round(W / k)), max(1, round(H / k)))), np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, 0)
    if (np.abs(border - bg).max(1) <= TOL).mean() < EDGE:
        return []
    fg = np.abs(a - bg).max(2) > TOL
    leaves = []

    def cut(t, b, l, r, axis, depth, fresh):
        m = fg[t:b, l:r]
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if not len(ys):
            return
        t, b, l, r = t + ys[0], t + ys[-1] + 1, l + xs[0], l + xs[-1] + 1  # 收紧到内容
        idx = np.flatnonzero(fg[t:b, l:r].any(1 - axis))
        breaks = np.flatnonzero(np.diff(idx) > max(2, round(GAP * (h, w)[axis])))
        segs = list(zip(idx[np.r_[0, breaks + 1]], idx[np.r_[breaks, len(idx) - 1]] + 1))
        if len(segs) > 1 and depth < DEPTH:
            for s, e in segs:
                if axis == 0:
                    cut(t + s, t + e, l, r, 1, depth + 1, True)
                else:
                    cut(t, b, l + s, l + e, 0, depth + 1, True)
        elif fresh and depth < DEPTH:  # 这个方向切不开，换个方向再试一次
            cut(t, b, l, r, 1 - axis, depth + 1, False)
        else:
            leaves.append((l, t, r, b))

    cut(0, h, 0, w, 0, 0, True)
    short = min(h, w)
    keep = [(l, t, r, b) for l, t, r, b in leaves
            if min(b - t, r - l) >= MIN_SIDE * short and (b - t) * (r - l) <= MAX_AREA * h * w]
    if len(keep) < 2:
        return []
    keep.sort(key=lambda x: (x[0] - x[2]) * (x[3] - x[1]))
    return [(max(0, round((l - 2) * k)), max(0, round((t - 2) * k)), min(W, round((r + 2) * k)), min(H, round((b + 2) * k)))
            for l, t, r, b in keep[:max_n]]
