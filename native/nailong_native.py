"""libnailong.so 的 Python 封装：多线程批量识别图片文件，结果与 nailong.detect 逐位一致。

    import nailong_native as nn
    res = nn.detect_files(paths, threads=8)       # [nailong.Result 或 None（读不了）]

C 不处理的少见格式（例如 CMYK JPEG、16 位灰度 PNG）会自动回退到 Python 参考实现。
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

import nailong
from nailong.features import FEATURE_NAMES, extract
from nailong.score import score

_LIB = None
PRESET_NAMES = list(nailong.PRESETS)
PARAM_NAMES = list(nailong.PRESETS["balanced"])


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("NAILONG_NATIVE_LIB") or str(Path(__file__).resolve().parent / "libnailong.so")
        lib = ctypes.CDLL(path)
        lib.nl_run_files.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                     ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _LIB = lib
    return _LIB


def run(paths, threads=None):
    """返回 (feats[n,20], scores[n,预设数], status[n])。status: 0 成功，2 坏文件，3 读失败。"""
    n = len(paths)
    min_yellow = nailong.min_yellow(nailong.PRESETS[PRESET_NAMES[0]])
    assert all(nailong.min_yellow(p) == min_yellow for p in nailong.PRESETS.values())
    arr = (ctypes.c_char_p * n)(*[os.fsencode(p) for p in paths])
    params = np.array([[nailong.PRESETS[k][q] for q in PARAM_NAMES] for k in PRESET_NAMES], np.float64)
    feats = np.zeros((n, len(FEATURE_NAMES)), np.float64)
    scores = np.zeros((n, len(PRESET_NAMES)), np.float64)
    status = np.zeros(n, np.int32)
    _lib().nl_run_files(arr, n, threads or os.cpu_count() or 1, min_yellow, params.ctypes.data, len(PRESET_NAMES),
                        feats.ctypes.data, scores.ctypes.data, status.ctypes.data)
    for i in np.nonzero(status == 1)[0]:  # C 不支持的格式：回退到 Python
        try:
            with open(paths[i], "rb") as fh:
                f = extract(fh.read(), min_yellow=min_yellow)
            feats[i] = [f[k] for k in FEATURE_NAMES]
            scores[i] = [score(f, nailong.PRESETS[k]) for k in PRESET_NAMES]
            status[i] = 0
        except Exception:  # noqa: BLE001
            status[i] = 2
    return feats, scores, status


def detect_files(paths, preset="balanced", threads=None):
    feats, scores, status = run(paths, threads)
    j = PRESET_NAMES.index(preset)
    thr = nailong.PRESETS[preset]["thr"]
    return [nailong.Result(bool(scores[i, j] >= thr), float(scores[i, j]), thr, dict(zip(FEATURE_NAMES, feats[i])))
            if status[i] == 0 else None for i in range(len(paths))]
