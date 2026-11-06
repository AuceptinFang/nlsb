"""奶龙分类器：DINOv2-small 图像特征 + 逻辑回归（比规则打分准得多，代价是 24 MB 模型和 ~170 MB 内存）。

骨干网络 DINOv2-small（Meta，Apache-2.0；Xenova 导出的 int8 ONNX）首次使用时下载并校验 SHA-256；
分类头 head.npz 只有 768 个权重 + 截距，随包分发。动图取 4 个均匀分布的帧，取最高分。

  python -m nailong.model 图片…     打分
  python -m nailong.model --fetch   只下载模型
  python -m nailong.model --serve   子进程模式：stdin 每行一个路径，stdout 每行回 "命中(0/1) 分数" 或 "err 原因"

环境变量：NAILONG_MODEL_DIR 模型存放目录；HF_ENDPOINT 换 Hugging Face 镜像（如 https://hf-mirror.com）；
NAILONG_THREADS 推理线程数（默认 2）。
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_FILE = "dinov2-small-int8.onnx"
MODEL_PATH = "Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"  # int8：准确度与 fp32 相同，内存少 100 MB
MODEL_SHA256 = "3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917"
HEAD = Path(__file__).with_name("head.npz")
SIZE = 224
SHRINK = 512  # 预处理前把长边缩到 SHRINK～2×SHRINK
FRAMES = 4


def model_dir():
    if os.environ.get("NAILONG_MODEL_DIR"):
        return Path(os.environ["NAILONG_MODEL_DIR"]).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/nailong-detector"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nailong-detector"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch():
    """确保模型文件在本地，返回路径。下载到临时文件，校验通过后再改名，不会留下半个文件。"""
    path = model_dir() / MODEL_FILE
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{MODEL_PATH}"
    tmp = path.with_suffix(".part")
    print(f"下载模型 {url}（24 MB）", file=sys.stderr, flush=True)

    def bar(n, block, total):
        if sys.stderr.isatty() and total > 0:
            done = min(n * block, total)
            print(f"\r  {'█' * (30 * done // total):<30} {done * 100 // total:3d}%", end="", file=sys.stderr, flush=True)

    urllib.request.urlretrieve(url, tmp, bar)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if _sha256(tmp) != MODEL_SHA256:
        tmp.unlink()
        raise RuntimeError(f"模型校验失败：{url}")
    tmp.rename(path)
    return path


class Classifier:
    def __init__(self, threads=None, tolerant=False):
        """tolerant：截断的图也尽量读（训练用）。默认读到截断就报错，因为那多半是文件还没下载完，调用方应稍后重试。"""
        import numpy as np
        self.tolerant = tolerant
        import onnxruntime as ort
        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("NAILONG_THREADS", 2))
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False  # 内存池会越长越大，关掉对速度几乎没影响
        self.sess = ort.InferenceSession(str(fetch()), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        h = np.load(HEAD)
        self.a, self.b, self.thr = h["a"], float(h["b"]), float(h["thr"])
        self.mean = np.array((0.485, 0.456, 0.406), np.float32)
        self.std = np.array((0.229, 0.224, 0.225), np.float32)

    def _prep(self, im):
        """大图先缩小 → 透明处垫白 → 白边补成正方形 → 224 双三次缩放 → ImageNet 归一化（与训练时完全一致）。"""
        from PIL import Image
        im = im.convert("RGBA")
        k = max(im.size) // SHRINK
        if k >= 2:  # 大图先整数倍缩小：省内存和时间，对分类几乎没影响（训练时也经过这一步）
            im = im.reduce(k)
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im).convert("RGB")
        w, h = im.size
        m = max(w, h)
        sq = Image.new("RGB", (m, m), (255,) * 3)
        sq.paste(im, ((m - w) // 2, (m - h) // 2))
        x = self.np.asarray(sq.resize((SIZE, SIZE), Image.BICUBIC), self.np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    def frames(self, path):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = self.tolerant
        im = Image.open(path)
        im.draft("RGB", (SHRINK, SHRINK))  # JPEG 解码时直接按 1/2、1/4、1/8 缩小，其他格式无效果
        n = getattr(im, "n_frames", 1)
        out = [self._prep(im)]
        for k in sorted({round(i * (n - 1) / (FRAMES - 1)) for i in range(1, FRAMES)} - {0}) if n > 1 else []:
            try:
                im.seek(k)
                out.append(self._prep(im))
            except Exception:  # noqa: BLE001
                if not self.tolerant:
                    raise
                break  # 训练时：动图后面的帧坏了就只用前面的
        return out

    def score(self, path):
        """返回 (是否奶龙, 分数 0~1)。"""
        np = self.np
        h = np.concatenate([self.sess.run(None, {self.inp: x[None]})[0] for x in self.frames(path)])
        e = np.concatenate([h[:, 0], h[:, 1:].mean(1)], 1)  # CLS + patch 平均，768 维
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        s = float((1 / (1 + np.exp(-(e @ self.a + self.b)))).max())
        return s >= self.thr, s


def serve():
    os.nice(10)
    clf = Classifier()
    print("ready", flush=True)
    for line in sys.stdin:
        try:
            hit, s = clf.score(line.rstrip("\n"))
            out = f"{int(hit)} {s:.4f}"
        except Exception as e:  # noqa: BLE001
            out = "err " + f"{type(e).__name__}: {e}".replace("\n", " ")
        print(out, flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--serve"]:
        return serve()
    if argv == ["--fetch"]:
        print(fetch())
        return 0
    if not argv:
        print(__doc__.strip())
        return 2
    clf, hits = Classifier(), 0
    for path in argv:
        try:
            hit, s = clf.score(path)
        except Exception as e:  # noqa: BLE001
            print(f"error\t{path}\t{e}", file=sys.stderr)
            continue
        hits += hit
        print(f"{s:6.3f}\t{'nailong' if hit else '-'}\t{path}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
