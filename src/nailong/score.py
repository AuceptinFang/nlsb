"""规则打分：每条规则对应一个可解释的视觉线索，加减分后与阈值比较。

参数由 tools/tune.py 在标注数据上搜索得到（只是挑选阈值和权重，推理时没有任何模型）。
"""
from __future__ import annotations

# 线索说明（键名 → 含义）：
#   gate / gate_o     奶油黄像素最少占比；gate_o=1 时橙色像素也计入
#   gate_lo, eye_gb   眼睛特例：黄+橙占比 ≥ gate_lo 且有一对绿眼（green_best ≥ eye_gb）时，
#                     即使没过 gate 也照常打分（奶龙头 P 在照片上，画面里黄色很少）
#   g0, g1, w_g       绿虹膜环比例：从 g0 到 g1 线性加到 w_g 分
#   w_pair            一对绿眼
#   gp, w_gp          一对很强的绿眼（green_best ≥ gp）：额外加分，用来抵消描边 / 硬边 / 眼白等
#                     针对 emoji 的扣分（平面卡通版、"奶思"花形头像等衍生形象）
#   gl, w_glint       瞳孔内高光 + 外圈黄皮肤（3D 渲染的大黑眼）
#   ye, w_ye          眼睛直接长在黄皮肤上
#   sc, w_sc          眼白（emoji 的特征，扣分）
#   hue, w_hue        偏暖的奶油黄
#   asp, w_asp        竖长团子形
#   red, w_red        红舌头 / 腮红（emoji，扣分）
#   out, w_out        深色描边（扣分）
#   tex, w_tex        128px 上的毛发 / 照片纹理（扣分）
#   tex2, w_tex2      256px 上的毛发纹理（扣分）
#   e_lo, e_hi, w_e   硬边密度：低加分、高扣分
#   belly, w_belly    奶油色肚皮
#   shade, w_shade    大尺度光影（3D 渲染）
#   orange, w_orange  橙色皮肤（橙色大笑版）
#   mouth, w_mouth    被皮肤包围的张大的深红色嘴
#   small, w_small    原图长边很小（放大后特征不可靠，扣分）
#   thr               判定阈值

PRESETS = {
    # F1 最优：误判和漏判大致平衡
    "balanced": {
        "gate": 0.15, "gate_o": 0, "gate_lo": 0.1, "eye_gb": 0.6, "g0": 0.06, "g1": 0.45, "w_g": 4, "w_pair": 0.5,
        "gl": 0.6, "w_glint": 1, "ye": 0.3, "w_ye": 1.5, "sc": 0.6, "w_sc": 1.5,
        "hue": 30, "w_hue": 1, "asp": 1.15, "w_asp": 1.5, "red": 0.04, "w_red": 2,
        "out": 0.5, "w_out": 0, "tex": 4.5, "w_tex": 3, "e_lo": 0.06, "e_hi": 0.16, "w_e": 0,
        "belly": 0.25, "w_belly": 0, "shade": 2.5, "w_shade": 1, "orange": 0.05, "w_orange": 1,
        "mouth": 0.005, "w_mouth": 1.5, "tex2": 11, "w_tex2": 3, "small": 80, "w_small": 0.5,
        "gp": 0.7, "w_gp": 2,
        "thr": 5,
    },
    # F2 最优：宁可多拦，尽量不漏
    "aggressive": {
        "gate": 0.15, "gate_o": 0, "gate_lo": 0.1, "eye_gb": 0.6, "g0": 0.06, "g1": 0.2, "w_g": 4, "w_pair": 0,
        "gl": 0.7, "w_glint": 0.5, "ye": 0.3, "w_ye": 0.5, "sc": 0.5, "w_sc": 2,
        "hue": 30, "w_hue": 1.5, "asp": 1.05, "w_asp": 0, "red": 0.04, "w_red": 0,
        "out": 0.35, "w_out": 0, "tex": 6, "w_tex": 3, "e_lo": 0.06, "e_hi": 0.1, "w_e": 0,
        "belly": 0.25, "w_belly": 0, "shade": 2.0, "w_shade": 0.5, "orange": 0.15, "w_orange": 2,
        "mouth": 0.005, "w_mouth": 0.5, "tex2": 7, "w_tex2": 0, "small": 80, "w_small": 1,
        "gp": 0.7, "w_gp": 3,
        "thr": 2.5,
    },
}
DEFAULT_PARAMS = PRESETS["balanced"]


def score(f: dict, p: dict = DEFAULT_PARAMS) -> float:
    if f["yfrac"] + p["gate_o"] * f["ofrac"] < p["gate"]:
        if not (f["yfrac"] + f["ofrac"] >= p["gate_lo"] and f["green_pair"] and f["green_best"] >= p["eye_gb"]):
            return -10.0
    s = p["w_g"] * min(1.0, max(0.0, (f["green_best"] - p["g0"]) / (p["g1"] - p["g0"])))
    s += p["w_pair"] * f["green_pair"]
    s += p["w_glint"] * (f["glint"] >= p["gl"])
    s += p["w_ye"] * (f["yellow_eye"] >= p["ye"])
    s -= p["w_sc"] * (f["sclera"] >= p["sc"])
    s += p["w_hue"] * (f["yhue"] <= p["hue"])
    s += p["w_asp"] * (f["yaspect"] >= p["asp"])
    s -= p["w_red"] * (f["red"] >= p["red"])
    s -= p["w_out"] * (f["outline"] >= p["out"])
    s -= p["w_tex"] * (f["ytex"] >= p["tex"])
    s -= p["w_tex2"] * (f["tex256"] >= p["tex2"])
    s += p["w_e"] * (f["edges"] <= p["e_lo"])
    s -= p["w_e"] * (f["edges"] >= p["e_hi"])
    s += p["w_belly"] * (f["belly"] >= p["belly"])
    s += p["w_shade"] * (f["shading"] >= p["shade"])
    s += p["w_orange"] * (f["ofrac"] >= p["orange"])
    s += p["w_mouth"] * (f["mouth"] >= p["mouth"])
    s -= p["w_small"] * (f["src"] < p["small"])
    s += p["w_gp"] * (f["green_pair"] and f["green_best"] >= p["gp"])
    return float(s)
