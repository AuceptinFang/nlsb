"""拼图切分（collage.panels）的行为，只用合成图。可直接 python tests/test_collage.py 运行，也兼容 pytest。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nailong.collage import panels  # noqa: E402


def grid(rows=4, cols=4, cell=280, bg="white"):
    im = Image.new("RGB", (cols * cell, rows * cell), bg)
    d = ImageDraw.Draw(im)
    for i in range(rows):
        for j in range(cols):
            x, y = j * cell, i * cell
            d.text((x + 100, y + 10), "label", fill="black")  # 每格上方一行字，离图形有空隙
            d.ellipse((x + 50, y + 50, x + cell - 50, y + cell - 20), fill=(240, 200, 80))
    return im


def test_grid_split():
    bs = panels(grid())
    assert len(bs) == 16, bs  # 文字太小，不算子图
    w, h = bs[0][2] - bs[0][0], bs[0][3] - bs[0][1]
    assert 180 <= w < 215 and 210 <= h < 245, (w, h)  # 图形 180×210，外加一点边


def test_transparent_background():
    im = grid().convert("RGBA")
    a = np.asarray(im).copy()
    a[(a[..., :3] == 255).all(2), 3] = 0
    assert len(panels(Image.fromarray(a))) == 16


def test_touching_frames_with_captions():
    """2×6 对比图：上下两格的图框几乎贴着，说明文字压在上面那格的下边框上，左边还有竖排的行标题。"""
    rng = np.random.default_rng(1)
    im = Image.new("RGB", (2400, 920), "white")
    d = ImageDraw.Draw(im)
    for row, top in enumerate((140, 500)):
        for j in range(6):
            x = 40 + j * 395
            tile = Image.fromarray(rng.integers(0, 256, (12, 12, 3), np.uint8)).resize((350, 350), Image.BICUBIC)
            im.paste(tile, (x, top))
            d.rectangle((x, top, x + 350, top + 350), outline="black", width=2)
            d.text((x + 150, top - 18 if row == 0 else top - 4), "p=0.98", fill="black")
    bs = panels(im)
    assert len(bs) == 12, bs
    assert all(330 < b[2] - b[0] < 380 and 330 < b[3] - b[1] < 380 for b in bs), bs


def test_tall_screenshot_skipped():
    assert panels(grid(rows=8, cols=3)) == []  # 高宽比 ≥ 2：当作手机截图


def test_not_collage():
    single = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(single).ellipse((60, 40, 450, 480), fill=(240, 200, 80))
    assert panels(single) == []
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 256, (400, 600, 3), np.uint8))
    assert panels(noise) == []  # 不是纯色底（照片之类）
    assert panels(Image.new("RGB", (300, 300), "white")) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
