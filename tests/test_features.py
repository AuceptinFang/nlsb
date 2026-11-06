"""用合成图检查各个特征的行为（不需要任何真实数据）。可直接 python tests/test_features.py 运行，也兼容 pytest。"""
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nailong  # noqa: E402
from nailong.features import extract  # noqa: E402

YELLOW = (240, 200, 80)
GREEN = (120, 160, 40)
PUPIL = (20, 20, 20)


def blob_with_eyes(iris=GREEN, size=256):
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((50, 20, 206, 250), fill=YELLOW)
    for cx in (100, 156):
        d.ellipse((cx - 13, 77, cx + 13, 103), fill=iris)
        d.ellipse((cx - 6, 84, cx + 6, 96), fill=PUPIL)
    return im


def emoji_face(size=256):
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((20, 20, 236, 236), fill=(255, 205, 0), outline=(60, 40, 0), width=6)
    for cx in (90, 166):
        d.ellipse((cx - 22, 70, cx + 22, 120), fill="white", outline=(60, 40, 0), width=3)
        d.ellipse((cx - 8, 88, cx + 8, 108), fill=PUPIL)
    d.ellipse((90, 160, 166, 210), fill=(230, 40, 60))
    return im


def test_green_eyes_found():
    f = extract(blob_with_eyes())
    assert f["green_best"] > 0.3, f["green_best"]
    assert f["green_pair"] == 1
    assert f["sclera"] < 0.3


def test_no_green_without_iris():
    f = extract(blob_with_eyes(iris=YELLOW))
    assert f["green_best"] < 0.05, f["green_best"]
    assert f["yellow_eye"] > 0.3


def test_emoji_cues():
    f = extract(emoji_face())
    assert f["sclera"] > 0.4, f["sclera"]
    assert f["outline"] > 0.2, f["outline"]
    assert f["red"] > 0.01, f["red"]


def test_blob_beats_emoji():
    for preset in nailong.PRESETS:
        assert nailong.detect(blob_with_eyes(), preset).score > nailong.detect(emoji_face(), preset).score


def test_blank_and_inputs():
    blank = Image.new("RGB", (40, 30), "white")
    assert not nailong.detect(blank).is_nailong
    buf = io.BytesIO()
    blob_with_eyes().save(buf, "PNG")
    a = nailong.detect(buf.getvalue()).features
    b = nailong.detect(blob_with_eyes()).features
    assert abs(a["green_best"] - b["green_best"]) < 1e-6
    rgba = Image.new("RGBA", (300, 200), (0, 0, 0, 0))  # 全透明 → 白底
    assert extract(rgba)["yfrac"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
