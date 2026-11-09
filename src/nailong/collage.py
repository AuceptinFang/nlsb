"""把拼图切成子图，思路借鉴漫画分格工具 Kumiko（https://github.com/njean42/kumiko）：

1. 找纯色背景上的连通区域，丢掉太小的（文字、标签、图标），合并互相重叠的框；
2. 每个框里再找分隔带递归切开：
   - 背景色的空带，允许夹着一行小字（格子之间常有说明文字，会把上下两格连成一个连通区域）；
   - 一到三行颜色均匀的细线：图框贴着图框时，中间只剩一条框线。图片里大片的纯色（天空、墙）是很多行连成一片，不算。

分类器把整张图缩到 224×224，一张 4×4 的奶龙拼图里每只只剩 40 px 左右，认不出来；切开分别打分就认得出。
不用固定网格去切：黄色角色的局部放大后很容易被认成奶龙，网格切块让难负样本误判多了几十倍。
沿空隙和框线切出来的是一个个完整的画面，和训练时的图是同一种分布。
"""
import numpy as np
from PIL import Image

A = 384          # 在长边 A px 的缩略图上分析
TOL = 24         # 与背景色（或一行的主色）的最大通道差，以内算同色
EDGE = 0.5       # 四条边上至少这么多像素是背景色，才当作纯色底的拼图（照片之类不切）
MIN_SIDE = 0.12  # 子图短边至少占整图短边的比例（太小的是文字、头像、图标）
MAX_AREA = 0.4   # 子图面积不超过整图的这个比例（再大就和整图打分重复了）
GAP = 0.005      # 背景色空带至少占整图对应边长的比例
SPARSE = 0.2     # 空带里允许的前景比例（夹着的说明文字）
LINE = 3         # 均匀细线最多几行
UNIFORM = 0.8    # 一行里至少这么多像素与这一行的主色相同，才算均匀
TALL = 2.0       # 高宽比到这个值的竖图不切：几乎都是手机截图（聊天、表情搜索、商品页），
                 # 里面夹着几张奶龙表情很常见，用户放出过这类图，不该整张屏蔽


def components(fg):
    """8 连通区域的外接框 [(t, b, l, r)]。按行程编码做并查集，比逐像素快得多。"""
    h, _ = fg.shape
    d = np.diff(np.pad(fg, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    ry, rs = np.nonzero(d == 1)
    re = np.nonzero(d == -1)[1]  # 与 rs 一一对应（都按行优先），右端不含
    n = len(ry)
    if not n:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row = np.searchsorted(ry, np.arange(h + 1))
    for y in range(h - 1):
        a0, a1, b1 = row[y], row[y + 1], row[y + 2]
        if a0 == a1 or a1 == b1:
            continue
        lo = np.searchsorted(re[a1:b1], rs[a0:a1], "left")   # 下一行里与它挨着（含斜挨着）的段
        hi = np.searchsorted(rs[a1:b1], re[a0:a1], "right")
        for i, (u, v) in enumerate(zip(lo, hi)):
            for j in range(u, v):
                x, z = find(a0 + i), find(a1 + j)
                if x != z:
                    parent[z] = x
    lab = np.unique([find(i) for i in range(n)], return_inverse=True)[1]
    m = lab.max() + 1
    t, b, l, r = np.full(m, h), np.zeros(m, int), np.full(m, fg.shape[1]), np.zeros(m, int)
    np.minimum.at(t, lab, ry)
    np.maximum.at(b, lab, ry + 1)
    np.minimum.at(l, lab, rs)
    np.maximum.at(r, lab, re)
    return [tuple(int(v) for v in x) for x in zip(t, b, l, r)]


def _merge(boxes):
    """重叠的框合并成一个（同一个图形的几块、压在图上的字）。"""
    boxes = [list(x) for x in boxes]
    i = 0
    while i < len(boxes):
        p = boxes[i]
        for j in range(i + 1, len(boxes)):
            q = boxes[j]
            if p[0] < q[1] and q[0] < p[1] and p[2] < q[3] and q[2] < p[3]:
                boxes[i] = [min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3])]
                del boxes[j]
                break
        else:
            i += 1
    return boxes


def _runs(mask):
    """mask 里连续为真的段 [(起, 止)]。"""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def panels(im, max_n=16):
    """im：PIL 图。返回子图框 [(l, t, r, b)]（im 的坐标），按面积从大到小；切不出两块以上时返回 []。"""
    W, H = im.size
    if H >= TALL * W:
        return []
    im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im.convert("RGBA")).convert("RGB")
    k = max(W, H) / A
    a = np.asarray(im.resize((max(1, round(W / k)), max(1, round(H / k)))), np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, 0)
    if (np.abs(border - bg).max(1) <= TOL).mean() < EDGE:
        return []
    fg = np.abs(a - bg).max(2) > TOL
    minside = MIN_SIDE * min(h, w)

    def split(t, b, l, r, axis, fresh):
        box = a[t:b, l:r] if axis == 0 else a[t:b, l:r].transpose(1, 0, 2)  # 统一按行找分隔带
        f = fg[t:b, l:r] if axis == 0 else fg[t:b, l:r].T
        n = len(f)
        main = np.median(box, 1)
        uniform = (np.abs(box - main[:, None]).max(2) <= TOL).mean(1) >= UNIFORM
        cuts = [(s, e) for s, e in _runs(f.mean(1) <= SPARSE) if e - s >= max(1, round(GAP * (h, w)[axis]))]
        cuts += [(s, e) for s, e in _runs(uniform) if e - s <= LINE]
        cuts = sorted((s, e) for s, e in cuts if 0 < s and e < n)
        pieces, start = [], 0
        for s, e in cuts:
            if s >= start:
                pieces.append((start, s))
                start = e
        pieces.append((start, n))
        if len(pieces) > 1 and all(e - s >= minside for s, e in pieces):
            return [q for s, e in pieces
                    for q in (split(t + s, t + e, l, r, 1, True) if axis == 0 else split(t, b, l + s, l + e, 0, True))]
        if fresh:  # 这个方向切不开，换个方向再试一次
            return split(t, b, l, r, 1 - axis, False)
        return [(t, b, l, r)]

    boxes = _merge(x for x in components(fg) if min(x[1] - x[0], x[3] - x[2]) >= minside)
    boxes = [q for t, b, l, r in boxes for q in split(t, b, l, r, 0, True)]
    keep = [(l, t, r, b) for t, b, l, r in boxes if (b - t) * (r - l) <= MAX_AREA * h * w]
    if len(keep) < 2:
        return []
    keep.sort(key=lambda x: (x[0] - x[2]) * (x[3] - x[1]))
    return [(max(0, round((l - 1) * k)), max(0, round((t - 1) * k)), min(W, round((r + 1) * k)), min(H, round((b + 1) * k)))
            for l, t, r, b in keep[:max_n]]
