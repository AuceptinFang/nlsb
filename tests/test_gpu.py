"""GPU 后端与 CPU 逐位一致（没有 wgpu 或没有可用 GPU 时跳过）。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nailong.features as F  # noqa: E402
from test_features import blob_with_eyes, emoji_face  # noqa: E402


def _detector():
    try:
        from nailong.gpu import Detector
        return Detector("aggressive", batch=8)
    except Exception as e:  # noqa: BLE001
        print("跳过：", e)
        return None


def _images():
    rng = np.random.default_rng(0)
    out = [blob_with_eyes(), emoji_face(), blob_with_eyes(size=180).resize((300, 140))]
    for _ in range(12):  # 随机拼贴：黄底 + 若干绿圈黑点 + 噪声，覆盖各种边界情况
        w, h = int(rng.integers(40, 600)), int(rng.integers(40, 600))
        im = Image.new("RGB", (w, h), tuple(int(c) for c in rng.integers(150, 256, 3)))
        d = ImageDraw.Draw(im)
        d.ellipse((w * .1, h * .05, w * .9, h * .95), fill=(240, 200, 80))
        for _ in range(int(rng.integers(0, 6))):
            cx, cy, r = rng.integers(0, w), rng.integers(0, h), int(rng.integers(2, 25))
            d.ellipse((cx - 2 * r, cy - 2 * r, cx + 2 * r, cy + 2 * r), fill=(120, 160, 40))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 20))
        a = np.asarray(im).astype(np.int16) + rng.integers(-20, 20, (h, w, 3))
        out.append(Image.fromarray(a.clip(0, 255).astype(np.uint8)))
    return out


def test_gpu_matches_cpu():
    det = _detector()
    if det is None:
        return
    ims = _images()
    for im, g in zip(ims, det.extract_many(ims)):
        assert F.extract(im, min_yellow=min(det.p["gate"], det.p["gate_lo"])) == g


if __name__ == "__main__":
    test_gpu_matches_cpu()
    print("ok test_gpu_matches_cpu")
