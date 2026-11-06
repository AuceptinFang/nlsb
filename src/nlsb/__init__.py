"""nlsb 核心：把 QQ 本地缓存里被屏蔽表情的图换成"已屏蔽"占位图，并自动识别、屏蔽新收到的奶龙。

命令行在 nlsb.cli；这里是名单、缓存替换、识别子进程和常驻监听。
数据目录：macOS ~/Library/Application Support/qq-sticker-block，Linux ~/.local/share/qq-sticker-block，
可用环境变量 NLSB_HOME 覆盖；QQ 数据目录可用 NLSB_QQ_ROOT 覆盖。
"""
import collections
import datetime
import fcntl
import io
import json
import os
import plistlib
import re
import select
import selectors
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

__version__ = "0.2.0"
MACOS = sys.platform == "darwin"


def _env(name):
    """NLSB_xxx，兼容改名前的 QSB_xxx。"""
    return os.environ.get(f"NLSB_{name}") or os.environ.get(f"QSB_{name}")


def _home():
    if _env("HOME"):
        return Path(_env("HOME")).expanduser()
    if MACOS:
        return Path.home() / "Library/Application Support/qq-sticker-block"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "qq-sticker-block"


HOME = _home()
BLOCKLIST = HOME / "blocklist.txt"
ALLOWLIST = HOME / "allowlist.txt"
AUTOLIST = HOME / "auto.txt"  # 自动识别加入的 md5（黑名单子集），不预置占位文件
BACKUP = HOME / "backup"
LOG = HOME / "nlsb.log"
SCORES = HOME / "scores.tsv"   # 每次识别的分数（供 nlsb review 找"可能漏了"的表情），自动截断
LABELS = HOME / "labels.tsv"   # nlsb review 里人工确认的结果，可用于重新训练
CONFIG_FILE = HOME / "config.json"
LOCK = HOME / "nlsb.lock"
HISTORY_DONE = HOME / "history-scanned"  # 第一次启动时在后台把已有缓存扫一遍，扫完留下这个标记
LOG_MAX = 2 << 20              # 日志和分数记录超过 2 MB 时只保留最后 1/4
DEFAULTS = {
    "backup": True,        # 屏蔽时备份原图。关掉省磁盘，代价是还原时只能删掉占位图、等 QQ 重新下载
    "detector": "model",   # model：DINOv2 分类器；rules：纯规则识别（不用下载模型，误伤多）
    "threshold": None,     # 分类器阈值，None 用模型自带的 0.5；调高误伤少、漏得多
}
QQ_ROOT = Path(_env("QQ_ROOT")) if _env("QQ_ROOT") else (
    Path.home() / "Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ" if MACOS
    else Path.home() / ".config/QQ")
LABEL = "qq-sticker-block"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"                 # macOS: launchd
UNIT = Path.home() / ".config/systemd/user" / f"{LABEL}.service"               # Linux: systemd --user

MARKER = b"qq-sticker-block"
FONTS = ("/System/Library/Fonts/Hiragino Sans GB.ttc",                   # macOS
         "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",             # Arch
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",        # Debian / Ubuntu
         "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",      # Fedora
         "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc")
BASES = ("Emoji/emoji-recv", "Pic")
SUBDIRS = ("Ori", "Thumb")
# 实际观察到的命名：Ori/<md5>.<ext>，Thumb/<md5>_<0|720>.<ext>
SEED_EXTS = ("jpg", "png", "gif", "jpeg", "webp")
SEED_THUMBS = ("0", "720")
MD5_RE = re.compile(r"^([0-9a-fA-F]{32})(?:_\d+)?(?:\.\w+)?$")
HEX32 = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])")
O_EVTONLY = 0x8000


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update({k: v for k, v in json.loads(CONFIG_FILE.read_text()).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    if os.environ.get("NLSB_BACKUP"):  # 环境变量优先，方便临时试验
        cfg["backup"] = os.environ["NLSB_BACKUP"] != "0"
    if os.environ.get("NLSB_DETECTOR"):
        cfg["detector"] = os.environ["NLSB_DETECTOR"]
    return cfg


def save_config(cfg):
    _atomic_write(CONFIG_FILE, json.dumps({k: cfg[k] for k in DEFAULTS}, ensure_ascii=False, indent=2) + "\n")


def _atomic_write(path, text):
    """先写临时文件再改名：写到一半被打断也不会留下半个名单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


CONFIG = load_config()
KEEP_BACKUP = CONFIG["backup"]
DETECTOR = CONFIG["detector"]


def _font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size), "已屏蔽"
    try:  # 让 fontconfig 找一个中文字体
        p = subprocess.run(["fc-match", "-f", "%{file}", ":lang=zh"], capture_output=True, text=True).stdout
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size), "已屏蔽"
    except OSError:
        pass
    return ImageFont.load_default(), "BLOCKED"


class _Kqueue:
    """macOS / BSD：目录写入（新建、删除、改名）和单个文件的改写 / 删除 / 被替换都由内核通知。"""

    def __init__(self):
        import resource
        # 常量只在有 kqueue 的平台上存在，所以放在这里而不是类属性
        self.DIR = select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND
        self.FILE = (select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND | select.KQ_NOTE_DELETE
                     | select.KQ_NOTE_RENAME | select.KQ_NOTE_ATTRIB)
        self.GONE = select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME
        self.kq = select.kqueue()
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # 每个被监听的文件占一个句柄
        want = 16384 if hard == resource.RLIM_INFINITY else min(hard, 16384)
        if soft < want:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            except (ValueError, OSError):
                pass

    def add(self, path, file=False):
        fd = os.open(path, os.O_RDONLY | O_EVTONLY)
        self.kq.control([select.kevent(fd, select.KQ_FILTER_VNODE, select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                       self.FILE if file else self.DIR)], 0)
        return fd

    def remove(self, fd):
        os.close(fd)  # 关闭句柄即注销事件

    def wait(self, timeout):
        """返回 [(句柄, 标志, 文件名或 None)]。kqueue 的目录事件不带文件名。"""
        return [(ev.ident, ev.fflags, None) for ev in self.kq.control(None, 256, timeout)]


class _Inotify:
    """Linux：用 ctypes 调 inotify，不需要额外依赖。目录监听本身就会报告其中文件的改写，事件带文件名。"""
    MASK = 0x2 | 0x8 | 0x80 | 0x100  # IN_MODIFY | IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
    GONE = 0

    def __init__(self):
        import ctypes
        import ctypes.util
        self.libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 失败")

    def add(self, path, file=False):
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(str(path)), self.MASK)
        if wd < 0:
            raise OSError(f"inotify_add_watch 失败: {path}")
        return wd

    def remove(self, fd):
        pass

    def wait(self, timeout):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return []
        out = []
        try:
            while True:
                buf = os.read(self.fd, 1 << 16)
                i = 0
                while i + 16 <= len(buf):  # struct inotify_event { int wd; u32 mask, cookie, len; char name[]; }
                    wd = int.from_bytes(buf[i:i + 4], sys.byteorder, signed=True)
                    mask = int.from_bytes(buf[i + 4:i + 8], sys.byteorder)
                    ln = int.from_bytes(buf[i + 12:i + 16], sys.byteorder)
                    name = buf[i + 16:i + 16 + ln].split(b"\0", 1)[0].decode(errors="surrogateescape")
                    out.append((wd, mask, name or None))
                    i += 16 + ln
        except BlockingIOError:
            pass
        return out


_log_file = None


def migrate():
    """改名前的日志叫 qsb.log：挪成 nlsb.log（打开着的进程按 inode 写，不受影响）。"""
    old = HOME / "qsb.log"
    if old.exists() and not LOG.exists():
        os.replace(old, LOG)


def log_to_file():
    """命令行用：日志写进 nlsb.log 而不是打印到终端（后台服务的 stdout 本来就是 nlsb.log）。"""
    global _log_file
    HOME.mkdir(parents=True, exist_ok=True)
    migrate()
    _log_file = open(LOG, "a")


def log(*args):
    print(time.strftime("%m-%d %H:%M:%S"), *args, flush=True, file=_log_file or sys.stdout)


def load_md5s(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if re.fullmatch(r"[0-9a-f]{32}", line):
            out.add(line)
    return out


def load_blocklist():
    return load_md5s(BLOCKLIST)


def _save_md5s(path, md5s):
    _atomic_write(path, "".join(f"{m}\n" for m in sorted(md5s)))


def save_blocklist(md5s):
    _save_md5s(BLOCKLIST, md5s)


def add_auto(md5s):
    _save_md5s(AUTOLIST, load_md5s(AUTOLIST) | md5s)


def save_allowlist(md5s):
    _save_md5s(ALLOWLIST, md5s)


def trim(path, keep=LOG_MAX // 4):
    """文件超过 LOG_MAX 时原地截成最后 keep 字节（launchd 用 O_APPEND 打开日志，原地截断是安全的）。"""
    try:
        if path.stat().st_size <= LOG_MAX:
            return
        with open(path, "r+b") as f:
            f.seek(-keep, os.SEEK_END)
            tail = f.read()
            tail = tail[tail.find(b"\n") + 1:]
            f.seek(0)
            f.truncate()
            f.write(tail)
    except OSError:
        pass


def record_score(md5, score, path):
    with open(SCORES, "a") as f:
        f.write(f"{int(time.time())}\t{md5}\t{score:.4f}\t{path}\n")


def load_scores():
    """{md5: (分数, 路径)}，同一个 md5 取最后一次。"""
    out = {}
    try:
        for line in SCORES.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                out[parts[1]] = (float(parts[2]), parts[3])
    except (OSError, ValueError):
        pass
    return out


class _Model:
    """分类器跑在按需启动的子进程里：加载后占 ~170 MB 且卸载也不归还，
    所以空闲 IDLE 秒就让子进程退出（重新启动只要 0.15 秒，所以 IDLE 可以很短），常驻的主进程不碰 numpy / onnxruntime。
    解码大图留下的堆碎片也不归还，所以每识别 MAX_JOBS 张就换一个新的子进程（启动只要 0.2 秒）。"""
    IDLE = 5
    MAX_JOBS = 20

    def __init__(self):
        self.proc = None
        self.last = 0.0
        self.jobs = 0

    def _cmd(self, *args):
        return [sys.executable, "-m", "nailong.model", *args]

    def _readline(self, timeout):
        # 不能用 select.select：服务开着上千个文件句柄，编号超过 1024 时它直接报错；selectors 在 macOS 上用 kqueue
        with selectors.DefaultSelector() as sel:
            sel.register(self.proc.stdout, selectors.EVENT_READ)
            ready = sel.select(timeout)
        if not ready:
            self.close()
            raise TimeoutError("识别子进程无响应")
        line = self.proc.stdout.readline().strip()
        if not line:
            self.close()
            raise RuntimeError("识别子进程退出了")
        return line

    def fetch(self):
        """首次运行时下载模型（24 MB），之后直接返回。"""
        subprocess.run(self._cmd("--fetch"), check=True, stdout=subprocess.DEVNULL)

    def score(self, path):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(self._cmd("--serve"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         text=True, bufsize=1)
            self._readline(600)  # "ready"：加载模型，第一次还要下载
            self.jobs = 0
        self.proc.stdin.write(f"{path}\n")
        self.proc.stdin.flush()
        line = self._readline(120)
        self.last = time.monotonic()
        self.jobs += 1
        if self.jobs >= self.MAX_JOBS:
            self.close()
        if line.startswith("err"):
            raise OSError(line[4:])
        hit, s = line.split()
        return hit == "1", float(s)

    def close_if_idle(self):
        if self.proc and time.monotonic() - self.last > self.IDLE:
            self.close()

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


MODEL = _Model()


def detect(path):
    """奶龙识别，返回 (是否命中, 分数)。读不了（比如还没下载完）会抛异常。"""
    if DETECTOR == "rules":
        import nailong
        r = nailong.detect(str(path), "aggressive")
        return r.is_nailong, r.score
    hit, s = MODEL.score(path)
    if CONFIG["threshold"] is not None:
        hit = s >= CONFIG["threshold"]
    return hit, s


def accounts():
    return sorted(QQ_ROOT.glob("nt_qq_*/nt_data"))


def months_to_watch():
    today = datetime.date.today()
    cur = today.strftime("%Y-%m")
    prev = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
    nxt = (today.replace(day=28) + datetime.timedelta(days=4)).strftime("%Y-%m")
    return [prev, cur, nxt] if today.day >= 25 else [prev, cur]


# ---------- 占位图 ----------

_ph_cache = {}


def placeholder(w, h):
    key = (w, h)
    if key not in _ph_cache:
        img = Image.new("RGB", (w, h), (200, 200, 205))
        d = ImageDraw.Draw(img)
        font, text = _font(max(12, min(w, h) // 7))
        box = d.textbbox((0, 0), text, font=font)
        d.text(((w - box[2]) / 2, (h - box[3]) / 2), text, fill=(90, 90, 100), font=font)
        buf = io.BytesIO()
        img.save(buf, format="GIF", comment=MARKER)
        _ph_cache[key] = buf.getvalue()
    return _ph_cache[key]


def is_placeholder(path):
    try:
        with open(path, "rb") as f:
            return MARKER in f.read(2048)
    except OSError:
        return False


def image_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def backup_path(md5, nt_data, path):
    return BACKUP / md5 / nt_data.parent.name / path.relative_to(nt_data)


def known_size(md5, nt_data):
    """从备份里找原图尺寸，用于预置占位图。"""
    for p in (BACKUP / md5 / nt_data.parent.name).rglob("*"):
        if p.is_file() and (s := image_size(p)):
            return s
    return (240, 240)


# ---------- 核心操作 ----------

def block_file(md5, nt_data, path):
    """把一个缓存文件替换成占位图。返回是否改动。"""
    if is_placeholder(path):
        return False
    size = image_size(path)
    if size is None:  # 可能还在下载
        return False
    bp = backup_path(md5, nt_data, path)
    if KEEP_BACKUP and not bp.exists():
        bp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bp)
    # 原地写入，保留权限和扩展属性
    with open(path, "r+b") as f:
        f.truncate(0)
        f.write(placeholder(*size))
    log("blocked", path.relative_to(nt_data))
    return True


def seed(md5, nt_data, month):
    """在指定月份目录预置占位文件，让 QQ 直接命中缓存不再下载。"""
    w, h = known_size(md5, nt_data)
    data = placeholder(w, h)
    for base in BASES:
        names = {"Ori": [f"{md5}.{e}" for e in SEED_EXTS],
                 "Thumb": [f"{md5}_{t}.{e}" for t in SEED_THUMBS for e in SEED_EXTS]}
        for sub, files in names.items():
            d = nt_data / base / month / sub
            d.mkdir(parents=True, exist_ok=True)
            for name in files:
                p = d / name
                if not p.exists():
                    p.write_bytes(data)
                    os.chmod(p, 0o600)


def iter_cache_dirs(nt_data, months=None):
    for base in BASES:
        root = nt_data / base
        if not root.is_dir():
            continue
        for mdir in sorted(root.iterdir()):
            if mdir.is_dir() and (months is None or mdir.name in months):
                for sub in mdir.iterdir():
                    if sub.is_dir():
                        yield sub


def scan_dir(d, nt_data, blocked):
    changed = []
    try:
        entries = list(os.scandir(d))
    except OSError:
        return changed
    for e in entries:
        m = MD5_RE.match(e.name)
        if m and m.group(1).lower() in blocked and e.is_file():
            p = Path(e.path)
            if block_file(m.group(1).lower(), nt_data, p):
                changed.append(p)
    return changed


def apply_all(blocked, months=None):
    auto = load_md5s(AUTOLIST)
    for nt in accounts():
        for d in iter_cache_dirs(nt, months):
            scan_dir(d, nt, blocked)
        for md5 in blocked - auto:
            for month in months_to_watch():
                seed(md5, nt, month)


def locate(md5s):
    """全盘搜索缓存目录，返回 {md5: [(nt_data, path)]}。"""
    found = {m: [] for m in md5s}
    for nt in accounts():
        for dirpath, _, files in os.walk(nt):
            for name in files:
                m = MD5_RE.match(name)
                if m and m.group(1).lower() in found:
                    found[m.group(1).lower()].append((nt, Path(dirpath) / name))
    return found


def unblock(md5):
    """还原一个表情，返回 (从备份写回的文件数, 删掉的占位文件数)。"""
    n_restored = n_removed = 0
    for nt in accounts():
        bdir = BACKUP / md5 / nt.parent.name
        restored = set()
        if bdir.is_dir():
            for bp in bdir.rglob("*"):
                if bp.is_file():
                    target = nt / bp.relative_to(bdir)
                    with open(target, "wb") as f:
                        f.write(bp.read_bytes())
                    restored.add(target)
                    n_restored += 1
                    log("restored", target.relative_to(nt))
        # 预置的占位文件没有原图，删掉让 QQ 重新下载
        for d in iter_cache_dirs(nt):
            for p in d.glob(f"{md5}*"):
                if p not in restored and is_placeholder(p):
                    p.unlink()
                    n_removed += 1
        if bdir.is_dir():
            shutil.rmtree(bdir)
    return n_restored, n_removed


def block_md5s(md5s):
    """手动屏蔽：加入黑名单（移出白名单），替换已有缓存，并给当前月份预置占位文件。返回 {md5: 找到的缓存份数}。"""
    md5s = set(md5s)
    save_blocklist(load_blocklist() | md5s)
    save_allowlist(load_md5s(ALLOWLIST) - md5s)
    _save_md5s(AUTOLIST, load_md5s(AUTOLIST) - md5s)
    found = {}
    for md5, hits in locate(md5s).items():
        changed = {p for nt, p in hits if block_file(md5, nt, p)}
        found[md5] = sum(1 for nt, p in hits if p in changed or backup_path(md5, nt, p).exists())
    apply_all(md5s)
    return found


def unblock_md5s(md5s, allow=True):
    """还原：移出黑名单，从备份写回原图（没有备份就删掉占位图让 QQ 重新下载）。allow=True 时加入白名单。"""
    md5s = set(md5s)
    save_blocklist(load_blocklist() - md5s)
    if allow:
        save_allowlist(load_md5s(ALLOWLIST) | md5s)
    _save_md5s(AUTOLIST, load_md5s(AUTOLIST) - md5s)
    return {m: unblock(m) for m in md5s}


# ---------- 常驻监听 ----------

class Watcher:
    """纯事件驱动：没有待办时一直阻塞等内核事件，只保留每 30 秒一次的轻量检查（黑名单、跨月）。"""
    SLOW = 30
    BACKLOG_GAP = 0.2

    def __init__(self):
        self.events = _Kqueue() if MACOS else _Inotify()
        self.fds = {}       # 目录 -> 句柄
        self.owner = {}     # 句柄 -> (nt_data, dir)
        self.names = {}     # 目录 -> 文件名集合（kqueue 目录事件不带文件名，靠差集找新文件）
        self.files = {}     # 已屏蔽文件 -> 句柄（仅 macOS：文件被改写 / 替换时内核通知）
        self.file_of = {}   # 句柄 -> 文件
        self.recheck = {}   # 文件 -> [到期时间]
        self.pending = {}   # 待识别的新文件 -> [到期时间]
        self.sizes = {}     # 待识别的新文件 -> 上次看到的大小
        self.blocked = set()
        self.allow = load_md5s(ALLOWLIST)
        self.bl_mtime = None
        self.known = set()  # 已见过的 md5，只对新出现的跑识别
        self.months = ()
        self.backlog = collections.deque()  # 历史扫描待办
        self.backlog_hits = 0
        self.next_backlog = 0.0

    # ---- 目录监听 ----
    def watch(self, path, nt_data):
        if path in self.fds or not path.is_dir():
            return
        names = set()
        if path.name in SUBDIRS:
            names = set(os.listdir(path))
            for n in names:
                if m := MD5_RE.match(n):
                    self.known.add(m.group(1).lower())
        fd = self.events.add(path)
        self.fds[path] = fd
        self.owner[fd] = (nt_data, path)
        self.names[path] = names

    def refresh_watches(self):
        for nt in accounts():
            for base in BASES:
                self.watch(nt / base, nt)  # 发现新建的月份目录
            for month in months_to_watch():
                for base in BASES:
                    self.watch(nt / base / month, nt)
                    for sub in SUBDIRS:
                        self.watch(nt / base / month / sub, nt)

    def watched_subdirs(self):
        return [d for d in self.fds if d.name in SUBDIRS]

    # ---- 已屏蔽文件的监听（macOS） ----
    def track(self, p):
        if MACOS and p not in self.files:
            try:
                fd = self.events.add(p, file=True)
            except OSError:
                return
            self.files[p] = fd
            self.file_of[fd] = p

    def untrack(self, p):
        fd = self.files.pop(p, None)
        if fd is not None:
            self.file_of.pop(fd, None)
            self.events.remove(fd)

    def track_md5s(self, md5s):
        if not MACOS or not md5s:
            return
        for d in self.watched_subdirs():
            for m in md5s:
                for p in d.glob(f"{m}*"):
                    self.track(p)

    # ---- 黑名单 ----
    def reload_blocklist(self):
        mt = BLOCKLIST.stat().st_mtime if BLOCKLIST.exists() else 0
        if mt == self.bl_mtime:
            return
        first = self.bl_mtime is None
        self.bl_mtime = mt
        new = load_blocklist()
        added, removed = new - self.blocked, self.blocked - new
        self.blocked = new
        log(f"blocklist: {len(new)} 项" + (f"，新增 {len(added)}" if added else "") + (f"，移除 {len(removed)}" if removed else ""))
        for p in [p for p in self.files if p.name[:32].lower() in removed]:
            self.untrack(p)
        if added and not first:  # 启动时由 loop 统一处理；命令行 block 已经全盘替换过，这里只补监听目录
            apply_all(added)
            self.track_md5s(added)

    # ---- 事件处理 ----
    def on_dir_event(self, nt_data, d, name):
        if d.name not in SUBDIRS:
            self.refresh_watches()
            return
        if name is not None:  # inotify：直接知道是哪个文件
            self.on_file(d / name)
            return
        try:
            cur = set(os.listdir(d))
        except OSError:
            return
        new = cur - self.names.get(d, set())
        self.names[d] = cur
        for n in new:
            self.on_file(d / n)

    def on_file(self, p):
        m = MD5_RE.match(p.name)
        if not m:
            return
        md5 = m.group(1).lower()
        now = time.monotonic()
        if md5 in self.blocked:  # 可能还没写完，稍后再查几次（对占位图是空操作）
            self.recheck[p] = [now + 0.3, now + 1, now + 3]
        elif md5 not in self.known:
            self.known.add(md5)
            if md5 not in self.allow:  # 大的动图可能要下载好几秒；读到半截会报错，按这个节奏重试
                self.pending[p] = [now + 0.3, now + 1, now + 3, now + 10, now + 30]

    def on_tracked_file_event(self, fd, flags):
        p = self.file_of.get(fd)
        if p is None:
            return
        if flags & self.events.GONE:  # 占位图被删或被同名文件替换：旧句柄作废，复查同名路径
            self.untrack(p)
        now = time.monotonic()
        self.recheck[p] = [now + 0.3, now + 1, now + 3]

    def nt_of(self, p):
        for nt in accounts():
            if nt in p.parents:
                return nt

    def run_detections(self):
        """对新出现的表情跑奶龙识别，命中就加入黑名单并立即屏蔽。"""
        now = time.monotonic()
        for p, dues in list(self.pending.items()):
            if not dues or dues[0] > now:
                continue
            dues.pop(0)
            md5 = p.name[:32].lower()
            try:
                size = p.stat().st_size
                if self.sizes.get(p) != size:  # 还在写入：等大小稳定了再识别（不消耗重试次数）
                    self.sizes[p] = size
                    dues.insert(0, now + 0.5)
                    continue
                hit, score = detect(p)
            except Exception as e:
                if not dues or isinstance(e, FileNotFoundError):  # 一直读不了就放弃
                    del self.pending[p]
                    self.sizes.pop(p, None)
                    if not isinstance(e, FileNotFoundError):  # 临时文件被 QQ 改名是常事，不记
                        log(f"识别失败，已放弃 {md5}: {type(e).__name__}: {e}")
                continue
            del self.pending[p]
            self.sizes.pop(p, None)
            self.judged(md5, hit, score, p)

    def judged(self, md5, hit, score, p):
        guarded(record_score, md5, score, p)
        if not hit:
            return False
        self.reload_blocklist()  # 先合并命令行刚做的改动，免得把它们覆盖掉
        self.allow = load_md5s(ALLOWLIST)
        if md5 in self.blocked or md5 in self.allow:
            return False
        log(f"auto 奶龙 score={score:.2f}", md5, p.parent.relative_to(self.nt_of(p) or p.parent))
        self.blocked.add(md5)
        add_auto({md5})
        save_blocklist(self.blocked)
        self.bl_mtime = BLOCKLIST.stat().st_mtime
        apply_all({md5})
        self.track_md5s({md5})
        return True

    def run_backlog(self):
        """第一次启动时的历史扫描：只在没有新表情要处理时做，每 BACKLOG_GAP 秒一张，不抢资源。"""
        if not self.backlog or self.pending or time.monotonic() < self.next_backlog:
            return
        p = self.backlog.popleft()
        try:
            hit, score = detect(p)
            self.backlog_hits += self.judged(p.name[:32].lower(), hit, score, p)
        except Exception:  # noqa: BLE001  读不了的旧文件跳过
            pass
        self.next_backlog = time.monotonic() + self.BACKLOG_GAP
        if not self.backlog:
            HISTORY_DONE.touch()
            log(f"历史缓存扫描完成，屏蔽了 {self.backlog_hits} 个")

    def run_rechecks(self):
        now = time.monotonic()
        if any(d and d[0] <= now for d in self.recheck.values()):
            self.reload_blocklist()  # 用户刚 remove 的表情不要再屏蔽回去
        for p, dues in list(self.recheck.items()):
            if not dues or dues[0] > now:
                continue
            dues.pop(0)
            m = MD5_RE.match(p.name)
            nt = self.nt_of(p)
            if not (m and nt and p.exists() and m.group(1).lower() in self.blocked):
                del self.recheck[p]
                continue
            if is_placeholder(p):
                self.track(p)
                del self.recheck[p]
            elif block_file(m.group(1).lower(), nt, p):
                self.track(p)
            if p in self.recheck and not self.recheck[p]:
                del self.recheck[p]

    def slow_tick(self):
        trim(LOG)
        trim(SCORES)
        self.allow = load_md5s(ALLOWLIST)
        self.reload_blocklist()
        self.refresh_watches()
        months = tuple(months_to_watch())
        if months != self.months:  # 跨月：给新月份预置占位文件、监听已屏蔽文件
            self.months = months
            manual = self.blocked - load_md5s(AUTOLIST)
            for md5 in manual:
                for nt in accounts():
                    for month in months:
                        seed(md5, nt, month)
            self.track_md5s(self.blocked)

    def loop(self):
        migrate()
        lock = open(LOCK, "w")
        try:  # 同一个数据目录只允许一个服务在跑，否则两边会互相覆盖名单
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("nlsb 已经在运行（同一个数据目录只能有一个服务）")
        log(f"started（识别器：{DETECTOR}，备份：{'开' if KEEP_BACKUP else '关'}）")
        if DETECTOR == "model":
            guarded(MODEL.fetch)
        self.refresh_watches()  # 先建监听，启动扫描期间到达的新文件也能收到事件
        self.reload_blocklist()
        guarded(apply_all, self.blocked, months_to_watch())
        guarded(self.slow_tick)
        if not HISTORY_DONE.exists():
            self.backlog.extend(guarded(history_todo) or [])
            log(f"开始在后台扫描已有缓存：{len(self.backlog)} 个表情（空闲时慢慢做）")
            if not self.backlog:
                HISTORY_DONE.touch()
        next_slow = time.monotonic() + self.SLOW
        while True:
            try:
                now = time.monotonic()
                dues = [d[0] for d in (*self.recheck.values(), *self.pending.values()) if d]
                if MODEL.proc:  # 到点叫醒自己，让识别子进程按时退出
                    dues.append(MODEL.last + MODEL.IDLE + 0.1)
                if self.backlog:
                    dues.append(self.next_backlog)
                timeout = max(0.0, min([next_slow - now] + [d - now for d in dues]))
                for ident, flags, name in self.events.wait(timeout):
                    if ident in self.owner:
                        self.on_dir_event(*self.owner[ident], name)
                    elif ident in self.file_of:
                        self.on_tracked_file_event(ident, flags)
                self.run_detections()
                self.run_rechecks()
                self.run_backlog()
                MODEL.close_if_idle()
                if time.monotonic() >= next_slow:
                    next_slow = time.monotonic() + self.SLOW
                    self.slow_tick()
            except Exception:  # 常驻服务不能因为一次偶发错误（如 EINTR、文件正被 QQ 改写）退出
                log("出错，1 秒后继续：\n" + traceback.format_exc().rstrip())
                time.sleep(1)


def guarded(fn, *args):
    try:
        return fn(*args)
    except Exception:
        log(f"{getattr(fn, '__name__', fn)} 出错（已跳过）：\n" + traceback.format_exc().rstrip())


# ---------- 命令行 ----------

def parse_md5s(args):
    out = []
    for a in args:
        m = HEX32.search(Path(a).name)
        if not m:
            raise ValueError(f"认不出表情 ID：{a}（需要 32 位的 MD5，或 QQ「另存为」得到的文件名）")
        out.append(m.group(1).lower())
    return out


def history_todo():
    """还没识别过的已有缓存：每个 md5 取最小的一份（通常是缩略图，识别快）。"""
    skip = load_blocklist() | load_md5s(ALLOWLIST) | set(load_scores())
    todo = {}
    for nt in accounts():
        for d in iter_cache_dirs(nt):
            if d.name not in SUBDIRS:
                continue
            for e in os.scandir(d):
                m = MD5_RE.match(e.name)
                if m and m.group(1).lower() not in skip:
                    md5 = m.group(1).lower()
                    size = e.stat().st_size
                    if md5 not in todo or size < todo[md5][0]:
                        todo[md5] = (size, Path(e.path))
    return [p for _, p in todo.values() if not is_placeholder(p)]


def _service_env():
    """后台服务继承这些环境变量（launchd / systemd 不会读你的 shell 配置）。"""
    keys = ("NLSB_QQ_ROOT", "NAILONG_MODEL_DIR", "NAILONG_THREADS")  # 其余设置在 config.json 里
    return {"NLSB_HOME": str(HOME), **{k: os.environ[k] for k in keys if os.environ.get(k)}}


# ---------- 后台服务（macOS: launchd，Linux: systemd --user） ----------

def _run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _service_home():
    """已安装的服务用的数据目录；没装返回 None。"""
    try:
        if MACOS:
            with open(PLIST, "rb") as f:
                env = plistlib.load(f)["EnvironmentVariables"]
            return Path(env.get("NLSB_HOME") or env["QSB_HOME"])  # 兼容改名前装的服务
        m = re.search(r"^Environment=(?:NLSB|QSB)_HOME=(.*)$", UNIT.read_text(), re.M)
        return Path(m.group(1)) if m else None
    except (OSError, KeyError, ValueError, plistlib.InvalidFileException):
        return None


def service_installed():
    """装了服务，并且它用的就是当前数据目录（换了 NLSB_HOME 时不去碰别的数据目录的服务）。"""
    home = _service_home()
    return home is not None and home.resolve() == HOME.resolve()


def service_pid():
    """后台服务的 pid；没装或没在运行返回 None。"""
    if not service_installed():
        return None
    if MACOS:
        out = _run("launchctl", "print", f"gui/{os.getuid()}/{LABEL}").stdout
        m = re.search(r"^\s*pid = (\d+)", out, re.M)
    else:
        m = re.fullmatch(r"(\d+)", _run("systemctl", "--user", "show", "-p", "MainPID", "--value", f"{LABEL}.service").stdout.strip())
    return int(m.group(1)) if m and m.group(1) != "0" else None


def service_install():
    HOME.mkdir(parents=True, exist_ok=True)
    if MACOS:
        plist = {
            "Label": LABEL,
            "ProgramArguments": [sys.executable, "-m", "nlsb", "run"],
            "EnvironmentVariables": _service_env(),
            "WorkingDirectory": "/",
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(LOG),
            "StandardErrorPath": str(LOG),
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "Nice": 10,
        }
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        target = f"gui/{os.getuid()}/{LABEL}"
        _run("launchctl", "bootout", target)
        for _ in range(50):  # bootout 是异步的：旧服务没退干净就 bootstrap 会报 "5: Input/output error"
            if _run("launchctl", "print", target).returncode:
                break
            time.sleep(0.1)
        with open(PLIST, "wb") as f:
            plistlib.dump(plist, f)
        for attempt in range(5):
            r = _run("launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST))
            if r.returncode == 0:
                return PLIST
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"launchctl bootstrap 失败：{r.stderr.strip()}")
    UNIT.parent.mkdir(parents=True, exist_ok=True)
    env = "".join(f"Environment={k}={v}\n" for k, v in _service_env().items())
    UNIT.write_text(f"""[Unit]
Description=QQ sticker blocker (replaces blocked stickers in the local cache)

[Service]
ExecStart={sys.executable} -m nlsb run
WorkingDirectory=/
{env}Restart=always
RestartSec=5
Nice=10
IOSchedulingClass=idle
StandardOutput=append:{LOG}
StandardError=append:{LOG}

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{LABEL}.service"], check=True)
    return UNIT


def service_uninstall():
    if not service_installed():
        return
    if MACOS:
        _run("launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}")
        PLIST.unlink(missing_ok=True)
    else:
        _run("systemctl", "--user", "disable", "--now", f"{LABEL}.service")
        UNIT.unlink(missing_ok=True)
        _run("systemctl", "--user", "daemon-reload")


def service_restart():
    if not service_installed():
        return
    if MACOS:
        _run("launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")
    else:
        _run("systemctl", "--user", "restart", f"{LABEL}.service")
