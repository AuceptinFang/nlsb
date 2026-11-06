"""用合成图做端到端校验（不需要真实数据）：各种格式 / 尺寸 / 透明度，C 与 Python 参考实现逐位一致。

python test_synth.py
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nailong_native as nn  # noqa: E402
import nailong  # noqa: E402
from nailong.features import FEATURE_NAMES, extract  # noqa: E402
from nailong.score import score  # noqa: E402


def images(rng, n=60):
    for k in range(n):
        w, h = int(rng.integers(8, 900)), int(rng.integers(8, 900))
        im = Image.new("RGB", (w, h), tuple(int(c) for c in rng.integers(0, 256, 3)))
        d = ImageDraw.Draw(im)
        d.ellipse((w * .1, h * .05, w * .9, h * .95), fill=(240, 200, 80))
        for _ in range(int(rng.integers(0, 5))):
            cx, cy, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(2, 30))
            d.ellipse((cx - 2 * r, cy - 2 * r, cx + 2 * r, cy + 2 * r), fill=(120, 160, 40))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 20))
        a = (np.asarray(im).astype(np.int16) + rng.integers(-25, 25, (h, w, 3))).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(a)
        kind = k % 5
        buf = io.BytesIO()
        if kind == 0:
            im.save(buf, "JPEG", quality=int(rng.integers(50, 96)))
        elif kind == 1:
            im.save(buf, "PNG")
        elif kind == 2:
            rgba = im.convert("RGBA")
            rgba.putalpha(Image.fromarray(rng.integers(0, 256, (h, w)).astype(np.uint8)))
            rgba.save(buf, "PNG")
        elif kind == 3:
            im.convert("P", palette=Image.ADAPTIVE).save(buf, "GIF")
        else:
            im.save(buf, "WEBP", quality=80)
        yield f"{k}.{['jpg', 'png', 'png', 'gif', 'webp'][kind]}", buf.getvalue()


def test_native_matches_python():
    gate = nailong.min_yellow(nailong.PRESETS["balanced"])
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, data in images(np.random.default_rng(0)):
            p = Path(tmp) / name
            p.write_bytes(data)
            paths.append(str(p))
        feats, scores, status = nn.run(paths, threads=4)
        for i, p in enumerate(paths):
            f = extract(Path(p).read_bytes(), min_yellow=gate)
            assert status[i] == 0, p
            assert [feats[i, j] for j in range(len(FEATURE_NAMES))] == [f[k] for k in FEATURE_NAMES], p
            assert list(scores[i]) == [score(f, nailong.PRESETS[k]) for k in nn.PRESET_NAMES], p


if __name__ == "__main__":
    test_native_matches_python()
    print("ok test_native_matches_python")
