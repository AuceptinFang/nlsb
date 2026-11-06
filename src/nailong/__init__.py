"""奶龙识别：纯手工图像特征 + 规则打分，不使用任何机器学习模型。

>>> import nailong
>>> r = nailong.detect("sticker.gif")
>>> r.is_nailong, r.score
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import FEATURE_NAMES, Source, extract
from .score import PRESETS, score

__all__ = ["detect", "Result", "PRESETS", "FEATURE_NAMES", "extract", "score"]
__version__ = "0.1.0"


def min_yellow(p: dict) -> float:
    """提前退出的门槛：低于它，无论其他特征如何分数都是 -10。"""
    return min(p["gate"], p["gate_lo"])


@dataclass
class Result:
    is_nailong: bool
    score: float
    threshold: float
    features: dict = field(repr=False)


def detect(src: Source, preset: str = "balanced", params: dict | None = None) -> Result:
    """识别一张图。src 可以是路径、bytes 或 PIL.Image；只看动图的第一帧。

    preset: "balanced"（误判漏判平衡）或 "aggressive"（宁可多拦）；
    params: 自定义参数，覆盖 preset。
    """
    p = params or PRESETS[preset]
    # 黄+橙占比不到 gate_lo 的图分数恒为 -10，无需后续计算（大部分表情都不黄，这一步省掉大半时间）
    f = extract(src, min_yellow=min_yellow(p))
    s = score(f, p)
    return Result(s >= p["thr"], s, p["thr"], f)
