"""命令行：python -m nailong [--preset balanced|aggressive] [--json] 图片..."""
import argparse
import json
import sys

from . import PRESETS, detect


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nailong", description="识别图片里的奶龙（无模型，纯图像特征）")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    ap.add_argument("--json", action="store_true", help="每行输出一个 JSON，含全部特征")
    ap.add_argument("--gpu", action="store_true", help="批量交给 GPU 做眼睛检测（需要 pip install .[gpu]）")
    a = ap.parse_args(argv)
    hits = 0
    if a.gpu:
        from .gpu import Detector
        results = zip(a.images, Detector(a.preset).detect_many(a.images))
    else:
        results = ((p, None) for p in a.images)
    for path, r in results:
        try:
            r = r or detect(path, a.preset)
        except Exception as e:  # noqa: BLE001
            print(f"error\t{path}\t{e}", file=sys.stderr)
            continue
        hits += r.is_nailong
        if a.json:
            print(json.dumps(dict(path=path, is_nailong=r.is_nailong, score=r.score, features=r.features),
                             ensure_ascii=False))
        else:
            print(f"{r.score:6.1f}\t{'nailong' if r.is_nailong else '-'}\t{path}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
