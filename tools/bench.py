"""性能基准（内存有上限）。

用法: python tools/bench.py <图片目录或 glob> [--limit N] [--workers N] [--backend cpu|gpu] [--mem-mb M]

- 单进程计时：图片先读进内存（总量不超过 --mem-mb），计时不含读盘；
- 多进程吞吐：每个工作进程自己读文件，不复制共享数据；进程数按可用内存自动下调。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nailong  # noqa: E402

MARKER = b"qq-sticker-block"
PER_WORKER_MB = 150  # 单个工作进程的保守峰值估计（解码 + 特征）


def available_mb() -> int:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 2048  # 拿不到时按 2 GB 保守处理


def list_images(pattern: str, limit: int):
    p = Path(pattern).expanduser()
    files = sorted(str(f) for f in p.rglob("*") if f.is_file()) if p.is_dir() else sorted(glob.glob(pattern, recursive=True))
    out = []
    for f in files:
        if len(out) >= limit:
            break
        try:
            with open(f, "rb") as fh:
                head = fh.read(2048)
        except OSError:
            continue
        if (head[:4] in (b"GIF8", b"\x89PNG") or head[:2] == b"\xff\xd8" or head[:4] == b"RIFF") and MARKER not in head:
            out.append(f)
    return out


def load_blobs(paths, mem_mb):
    blobs, total = [], 0
    for f in paths:
        b = open(f, "rb").read()
        if total + len(b) > mem_mb * 1024 * 1024:
            break
        blobs.append(b)
        total += len(b)
    return blobs, total


def _run_path(path):
    try:
        with open(path, "rb") as fh:
            return nailong.detect(fh.read(), "aggressive").is_nailong
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--mem-mb", type=int, default=300, help="单进程测试最多读进内存多少 MB")
    a = ap.parse_args()
    paths = list_images(a.pattern, a.limit)
    blobs, total = load_blobs(paths, a.mem_mb)
    print(f"{len(paths)} 张图；单进程测试读入 {len(blobs)} 张（{total / 2**20:.0f} MB）")

    if a.backend == "gpu":
        from nailong import gpu
        det = gpu.Detector("aggressive")
        det.detect_many(blobs[:64])  # 预热：编译着色器
        t = time.perf_counter()
        res = det.detect_many(blobs)
        dt = time.perf_counter() - t
        print(f"GPU 批处理: {dt / len(blobs) * 1000:.3f} ms/张，{len(blobs) / dt:.0f} 张/秒，命中 {sum(r.is_nailong for r in res)}")
        return

    t = time.perf_counter()
    hits = sum(bool(nailong.detect(b, "aggressive").is_nailong) for b in blobs)
    dt = time.perf_counter() - t
    print(f"单进程: {dt / len(blobs) * 1000:.3f} ms/张，{len(blobs) / dt:.0f} 张/秒，命中 {hits}")
    del blobs

    budget = max(1, (available_mb() // 2) // PER_WORKER_MB)  # 最多用一半可用内存
    workers = max(1, min(a.workers, os.cpu_count() or 1, budget))
    if workers < a.workers:
        print(f"按可用内存限制，进程数 {a.workers} → {workers}")
    with ProcessPoolExecutor(workers) as ex:
        t = time.perf_counter()
        res = list(ex.map(_run_path, paths, chunksize=16))
        dt = time.perf_counter() - t
    print(f"{workers} 进程（含读盘）: {len(paths) / dt:.0f} 张/秒，命中 {sum(bool(r) for r in res)}")


if __name__ == "__main__":
    main()
