"""手工特征提取。全部是确定性的图像处理，不依赖任何学习得到的权重。

两种分辨率：
- 颜色 / 形状 / 纹理特征在长边 128px 上计算；
- 眼睛检测在长边 256px 上计算（奶龙的眼睛在缩略图里常常只有几个像素）。

PIL 的 HSV 色相取值 0-255（对应 0-360°）。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

Source = Union[str, Path, bytes, Image.Image]

EYE_SIZE = 256
COLOR_SIZE = 128
# 眼睛检测的瞳孔方框半径（256px 图上），覆盖 2px 到 30px 左右的瞳孔
EYE_RADII = (1, 2, 3, 4, 6, 8, 11, 15)

FEATURE_NAMES = (
    "yfrac", "yhue", "ysat", "yaspect", "yfill", "outline", "edges", "ytex",
    "shading", "belly", "red", "ofrac", "mouth", "src",
    "green_best", "green_pair", "yellow_eye", "glint", "sclera", "tex256",
)


# ---------- 读图 ----------

def load(src: Source, size: int = EYE_SIZE, frame: int = 0) -> Image.Image:
    """读第 frame 帧，透明处铺白底，缩放到长边 = size（小图会放大）。"""
    return _load(src, size, frame)[0]


def _load(src: Source, size: int, frame: int):
    if isinstance(src, Image.Image):
        im = src
    else:
        im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
        if im.format == "JPEG":
            d = size if FAST_JPEG else size * 2
            im.draft("RGB", (d, d))  # 大照片按缩小尺寸解码（DCT 域缩放，几乎零成本）
    if frame:
        im.seek(min(frame, getattr(im, "n_frames", 1) - 1))
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    im = im.convert("RGBA" if has_alpha else "RGB")
    w, h = im.size
    orig = max(w, h)
    f = max(w, h) // size
    if f >= 2:  # 先整数倍盒式降采样（很快），再做精确缩放
        im = im.reduce(f)
        w, h = im.size
    k = size / max(w, h)
    im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR)
    if not has_alpha:
        return im, orig
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB"), orig


def hsv(im: Image.Image):
    a = np.asarray(im.convert("HSV"), dtype=np.int16)
    return a[..., 0], a[..., 1], a[..., 2]


# ---------- 积分图工具 ----------

def integral(mask: np.ndarray, pad: int) -> np.ndarray:
    m = np.pad(mask.astype(np.float32), pad)
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float32)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    return ii


def box(ii: np.ndarray, r: int, pad: int, shape) -> np.ndarray:
    """以每个像素为中心、边长 2r+1 的方框内的和（越界部分按 0 计）。"""
    h, w = shape
    y0, x0 = pad - r, pad - r
    y1, x1 = pad + r + 1, pad + r + 1
    return (ii[y1:y1 + h, x1:x1 + w] - ii[y0:y0 + h, x1:x1 + w]
            - ii[y1:y1 + h, x0:x0 + w] + ii[y0:y0 + h, x0:x0 + w])


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    ii = integral(a, r)
    return box(ii, r, r, a.shape) / float((2 * r + 1) ** 2)


# ---------- 颜色 / 形状 / 纹理 ----------

def color_features(im: Image.Image, min_yellow: float = 0.0) -> dict:
    """颜色 / 形状 / 纹理特征。min_yellow>0 时，黄 + 橙占比不到它就只返回这两个占比（其余记 0）。"""
    H, S, V = hsv(im)
    yellow = (H >= 22) & (H <= 48) & (S >= 70) & (V >= 110)
    orange = (H >= 12) & (H < 22) & (S >= 90) & (V >= 100)
    f = dict(yfrac=float(yellow.mean()), ofrac=float(orange.mean()))
    if min_yellow and f["yfrac"] + f["ofrac"] < min_yellow:
        f.update(red=0.0, mouth=0.0, yhue=0.0, ysat=0.0, yaspect=0.0, yfill=0.0, outline=0.0,
                 edges=0.0, ytex=0.0, shading=0.0, belly=0.0)
        f["skipped"] = True
        return f
    cream = (H >= 18) & (H <= 48) & (S >= 25) & (S < 70) & (V >= 170)
    red = ((H <= 12) | (H >= 235)) & (S >= 120) & (V >= 120)
    n = yellow.sum()
    f["red"] = float(red.mean())
    # 张嘴：深红色区域，四周是黄 / 橙色皮肤
    skin = (yellow | orange).astype(np.float32)
    mouth = (V >= 25) & (V < 120) & (S >= 70) & ((H <= 16) | (H >= 235))
    near_skin = box_blur(skin, 3) >= 0.35
    f["mouth"] = float((mouth & near_skin).sum() / max(1.0, skin.sum()))
    if n < 20:
        f.update(yhue=0.0, ysat=0.0, yaspect=0.0, yfill=0.0, outline=0.0,
                 edges=0.0, ytex=0.0, shading=0.0, belly=0.0)
        return f
    f["yhue"] = float(np.median(H[yellow]))
    f["ysat"] = float(np.median(S[yellow]))

    # 黄色主体的稳健外接框（去掉两端各 2% 的离群像素）
    ys, xs = np.nonzero(yellow)
    y0, y1 = np.percentile(ys, (2, 98)).astype(int)
    x0, x1 = np.percentile(xs, (2, 98)).astype(int)
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    f["yaspect"] = bh / bw
    f["yfill"] = float(yellow[y0:y1 + 1, x0:x1 + 1].mean())

    Vf = V.astype(np.float32)
    # 描边：黄色边界向外 1-2 像素内是深色
    dark = V < 90
    border = hits = 0
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(yellow, (-dy, -dx), (0, 1))
        edge = yellow & ~nb
        d1 = np.roll(dark, (-dy, -dx), (0, 1))
        d2 = np.roll(dark, (-2 * dy, -2 * dx), (0, 1))
        border += edge.sum()
        hits += (edge & (d1 | d2)).sum()
    f["outline"] = float(hits / border) if border else 0.0

    # 硬边密度（外接框内）：线条画高，3D 渲染低
    sub = Vf[y0:y1 + 1, x0:x1 + 1]
    gx = np.abs(np.diff(sub, axis=1))[:-1, :] > 40 if sub.shape[0] > 1 and sub.shape[1] > 1 else np.zeros((1, 1), bool)
    gy = np.abs(np.diff(sub, axis=0))[:, :-1] > 40 if sub.shape[0] > 1 and sub.shape[1] > 1 else np.zeros((1, 1), bool)
    f["edges"] = float((gx | gy).mean())

    # 细纹理：相邻黄色像素的平均亮度差（毛发、照片噪点高）
    hx = yellow[:, 1:] & yellow[:, :-1]
    hy = yellow[1:, :] & yellow[:-1, :]
    dsum = np.abs(np.diff(Vf, axis=1))[hx].sum() + np.abs(np.diff(Vf, axis=0))[hy].sum()
    f["ytex"] = float(dsum / max(1, hx.sum() + hy.sum()))

    # 光影比：大尺度明暗起伏 / 细碎残差。3D 渲染高；平涂 emoji、毛发低
    blur = box_blur(Vf, 4)
    low = float(blur[yellow].std())
    resid = float(np.abs(Vf - blur)[yellow].mean())
    f["shading"] = low / (resid + 1.0)

    # 奶油色肚皮：外接框内浅奶油色像素相对黄色像素的比例
    f["belly"] = float(cream[y0:y1 + 1, x0:x1 + 1].sum() / n)
    return f


# ---------- 眼睛：中心-环绕滤波 ----------

MAX_CANDIDATES = 12000
# 速度开关（见 tools/bench.py 和 README 里的准确度对比）
FAST_JPEG = True      # JPEG 直接按 ≥ 目标尺寸解码（DCT 缩放），否则按 2 倍目标尺寸
FAST_SMALL = False    # 128px 图用 2 倍盒式降采样生成（会降召回，默认关），否则双线性
CANDIDATE_STRIDE = 2  # 瞳孔候选点按棋盘格隔一个取一个


def _ring_table():
    """每个尺度的 (r, r2, r3, 核心面积, 环面积, 外圈面积)。GPU 版共用这张表。"""
    out = []
    for r in EYE_RADII:
        r2 = int(round(2.2 * r + 1))
        r3 = int(round(3.6 * r + 3))
        a1, a2, a3 = (2 * r + 1) ** 2, (2 * r2 + 1) ** 2, (2 * r3 + 1) ** 2
        out.append((r, r2, r3, a1, a2 - a1, a3 - a2))
    return tuple(out)


RINGS = _ring_table()


def _peaks(score, rad, ys, xs, thr: float, k: int = 4):
    """非极大值抑制取前 k 个峰：[(score, r, y, x)]。候选需按光栅顺序给出；稳定排序保证结果确定。"""
    order = np.argsort(-score, kind="stable")
    out = []
    for i in order:
        v = float(score[i])
        if v < thr or len(out) >= k:
            break
        y, x, r = int(ys[i]), int(xs[i]), int(rad[i])
        if all(abs(y - py) > 3 * pr + 2 or abs(x - px) > 3 * pr + 2 for _, pr, py, px in out):
            out.append((v, r, y, x))
    return out


def green_pair(g_best, g_rad, ys, xs) -> float:
    peaks = _peaks(g_best, g_rad, ys, xs, 0.2)
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            (_, ra, ya, _), (_, rb, yb, _) = peaks[i], peaks[j]
            if 0.5 <= ra / rb <= 2 and abs(ya - yb) <= 3 * max(ra, rb) + 2:
                return 1.0
    return 0.0


def eye_masks(H, S, V):
    dark = V < 50
    green = (H >= 38) & (H <= 110) & (S >= 45) & (V >= 40) & (V <= 235)
    yel = (H >= 22) & (H <= 48) & (S >= 60) & (V >= 100)
    white = (V >= 200) & (S < 60)
    return dark, green, yel, white


def texture(V, mask) -> float:
    """相邻（右、下）都在 mask 内的像素对的平均亮度差。整数运算，结果精确。"""
    Vi = V.astype(np.int32)
    hx = mask[:, 1:] & mask[:, :-1]
    hy = mask[1:, :] & mask[:-1, :]
    total = int(np.abs(np.diff(Vi, axis=1))[hx].sum()) + int(np.abs(np.diff(Vi, axis=0))[hy].sum())
    return total / max(1, int(hx.sum()) + int(hy.sum()))


def candidates(dark):
    """瞳孔候选点：深色像素，按棋盘格隔一取一；太多时按光栅顺序等间隔抽样。"""
    ys, xs = np.nonzero(dark)
    if CANDIDATE_STRIDE > 1 and len(ys):
        keep = ((ys + xs) % CANDIDATE_STRIDE) == 0
        ys, xs = ys[keep], xs[keep]
    if len(ys) > MAX_CANDIDATES:  # 大片深色（暗背景照片）
        step = len(ys) // MAX_CANDIDATES + 1
        ys, xs = ys[::step], xs[::step]
    return ys, xs


def eye_features(im: Image.Image) -> dict:
    """中心-环绕滤波：瞳孔（深色核心）→ 虹膜环 → 外圈皮肤。

    只在候选的深色像素处求值，用积分图 O(1) 取任意方框和。阈值判断全部是整数交叉相乘，
    没有浮点舍入问题（GPU 版与之逐位一致）。
    """
    return eye_features_hsv(*hsv(im))


def eye_features_hsv(H, S, V) -> dict:
    dark, green, yel, white = eye_masks(H, S, V)
    tex256 = texture(V, yel)
    ys, xs = candidates(dark)
    if len(ys) == 0:
        return dict(green_best=0.0, green_pair=0.0, yellow_eye=0.0, glint=0.0, sclera=0.0, tex256=tex256)

    # 四个通道（深色/绿/黄/白）按 16 位一段打包进一个 uint64，只做一次积分图。
    # 方框和是线性运算，在 mod 2^64 下照样成立；每个方框的计数 < 2^16，移位即可无损拆开。
    # 不补边：越界坐标裁剪到边界，等价于补零。
    h, w = V.shape
    packed = (dark.astype(np.uint64) | (green.astype(np.uint64) << np.uint64(16))
              | (yel.astype(np.uint64) << np.uint64(32)) | (white.astype(np.uint64) << np.uint64(48)))
    ii = np.zeros((h + 1, w + 1), np.uint64)
    ii[1:, 1:] = packed.cumsum(0).cumsum(1)
    flat = ii.ravel()
    W1 = w + 1
    M16 = np.uint64(0xFFFF)
    cache = {}

    def box4(r):  # 候选点上边长 2r+1 方框内四个通道的计数（int64）；同一半径只算一次
        if r not in cache:
            y0 = np.clip(ys - r, 0, h) * W1
            y1 = np.clip(ys + r + 1, 0, h) * W1
            x0 = np.clip(xs - r, 0, w)
            x1 = np.clip(xs + r + 1, 0, w)
            s = flat[y1 + x1] - flat[y0 + x1] - flat[y1 + x0] + flat[y0 + x0]
            cache[r] = tuple(((s >> np.uint64(16 * i)) & M16).astype(np.int64) for i in range(4))
        return cache[r]

    n = len(ys)
    g_best = np.zeros(n, np.float32); g_rad = np.zeros(n, np.int16)
    ye_best = gl_best = sc_best = 0.0
    for r, r2, r3, a1, ring_n, out_n in RINGS:
        cd, cg, cy, cw = box4(r)
        md, mg, my, mw = box4(r2)
        _, og, oy, _ = box4(r3)
        ring_y_n = my - cy
        pupil = (20 * cd >= 11 * a1) & (2 * (md - cd) <= ring_n)           # 核心 ≥55% 深色，环 ≤50% 深色
        skin = (10 * (oy - my) >= 3 * out_n) & (4 * (og - mg) <= out_n)    # 外圈 ≥30% 黄，且 ≤25% 绿
        glint = ((20 * cd >= 7 * a1) & (50 * cw >= a1) & (10 * cw <= 3 * a1)  # 核心 ≥35% 深色、含 2%-30% 高光
                 & (5 * ring_y_n >= 2 * ring_n) & skin)                       # 环 ≥40% 黄
        if not (pupil.any() or glint.any()):
            continue
        g = np.where(pupil & skin, (mg - cg).astype(np.float32) / np.float32(ring_n), np.float32(0))  # 绿虹膜
        upd = g > g_best
        g_best[upd] = g[upd]; g_rad[upd] = r
        if pupil.any():
            ye_best = max(ye_best, float(np.float32(ring_y_n[pupil].max()) / np.float32(ring_n)))   # 眼睛长在黄皮肤上
            sc_best = max(sc_best, float(np.float32((mw - cw)[pupil].max()) / np.float32(ring_n)))  # emoji 式眼白
        if glint.any():  # 瞳孔内有高光（3D 渲染的大黑眼睛），外圈是黄皮肤
            gl_best = max(gl_best, float(np.float32(ring_y_n[glint].max()) / np.float32(ring_n)))

    return dict(green_best=float(g_best.max()), green_pair=green_pair(g_best, g_rad, ys, xs),
                yellow_eye=ye_best, glint=gl_best, sclera=sc_best, tex256=tex256)


EYE_KEYS = ("green_best", "green_pair", "yellow_eye", "glint", "sclera", "tex256")


def extract(src: Source, frame: int = 0, min_yellow: float = 0.0) -> dict:
    """提取全部特征。min_yellow>0 时，黄色（+橙色）占比低于它就跳过眼睛检测，眼睛特征记为 0。"""
    big, orig = _load(src, EYE_SIZE, frame)
    if FAST_SMALL and big.width >= 2 and big.height >= 2:
        small = big.reduce(2)
    else:
        small = big.resize((max(1, big.width // 2), max(1, big.height // 2)), Image.BILINEAR)
    f = color_features(small, min_yellow)
    f["src"] = float(orig)
    if f.get("skipped"):
        f.update(dict.fromkeys(EYE_KEYS, 0.0))
        return f
    f.update(eye_features(big))
    return f
