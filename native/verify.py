"""校验 libnailong.so 与 Python 参考实现逐位一致，并计时。

python verify.py <图片列表文件，每行一个路径> [--threads 1,8,16] [--limit N]
- 正确性：每张图 20 个特征和全部预设的分数都要 ==；参考实现报错的图，C 也必须报错。
- 计时：同一批文件，Python 参考实现单进程 vs C（不同线程数），都已预热页缓存。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nailong_native as nn  # noqa: E402
import nailong  # noqa: E402
from nailong.features import FEATURE_NAMES, extract  # noqa: E402
from nailong.score import score  # noqa: E402


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    paths = [p for p in Path(sys.argv[1]).read_text().splitlines() if p][: int(arg("--limit", "1000000000"))]
    threads = [int(x) for x in arg("--threads", "1,8").split(",")]
    for p in paths:  # 预热页缓存
        with open(p, "rb") as fh:
            fh.read()
    gate = nailong.min_yellow(nailong.PRESETS["balanced"])

    t0 = time.perf_counter()
    ref = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                f = extract(fh.read(), min_yellow=gate)
            ref.append((f, [score(f, nailong.PRESETS[k]) for k in nn.PRESET_NAMES]))
        except Exception:  # noqa: BLE001
            ref.append(None)
    t_py = time.perf_counter() - t0

    feats, scores, status = nn.run(paths, threads=max(threads))
    bad = 0
    for i, r in enumerate(ref):
        if r is None:
            bad += status[i] == 0
            continue
        f, s = r
        if status[i] != 0 or any(feats[i, j] != f[k] for j, k in enumerate(FEATURE_NAMES)) or list(scores[i]) != s:
            bad += 1
    print(f"{len(paths)} 张图：与参考实现不一致 {bad} 张")
    print(f"Python 参考实现（单进程）：{len(paths) / t_py:.1f} 张/秒")
    for t in threads:
        t0 = time.perf_counter()
        nn.run(paths, threads=t)
        dt = time.perf_counter() - t0
        print(f"C {t:2d} 线程：{len(paths) / dt:.0f} 张/秒，加速比 {t_py / dt:.1f}x")


if __name__ == "__main__":
    main()
