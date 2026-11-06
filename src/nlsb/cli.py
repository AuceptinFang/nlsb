"""nlsb 命令行。"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nlsb

TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code):
    return lambda s: f"\033[{code}m{s}\033[0m" if TTY else str(s)


bold, dim, red, green, yellow = _c("1"), _c("2"), _c("31"), _c("32"), _c("33")
OK, WARN, ERR = green("✓"), yellow("!"), red("✗")


def tilde(p):
    p = str(p)
    home = str(Path.home())
    return "~" + p[len(home):] if p.startswith(home) else p


def pad(s, width):
    """按显示宽度补空格（中文占两格）。"""
    w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)
    return s + " " * max(0, width - w)


def short(md5):
    return md5[:8] + "…"


def confirm(question, yes):
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{ERR} {question} 需要确认；非交互环境请加 --yes")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def size_of(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def _model_info():
    """不导入 nailong 包（它会加载 numpy），直接读 model.py 里的路径常量。"""
    spec = importlib.util.find_spec("nailong")
    path = Path(spec.submodule_search_locations[0]) / "model.py"
    s = importlib.util.spec_from_file_location("_nlsb_model_paths", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.model_dir() / m.MODEL_FILE


def warn_if_no_service():
    if not nlsb.service_pid():
        print(dim("  提示：后台服务没在运行，新收到的表情不会自动处理（nlsb install）"))


def parse_ids(ids):
    try:
        return nlsb.parse_md5s(ids)
    except ValueError as e:
        sys.exit(f"{ERR} {e}")


# ---------- 命令 ----------

def cmd_status(a):
    print(f"{bold('nlsb')} {nlsb.__version__}  {dim('数据目录 ' + tilde(nlsb.HOME))}\n")
    rows = []
    pid = nlsb.service_pid()
    kind = "launchd" if nlsb.MACOS else "systemd"
    if pid:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        rows.append(("服务", f"{green('● 运行中')}  pid {pid} · {kind} · 内存 {int(rss or 0) // 1024} MB"))
    elif nlsb.service_installed():
        rows.append(("服务", f"{red('● 没在运行')}  已安装但没起来，看看 {tilde(nlsb.LOG)}"))
    else:
        rows.append(("服务", f"{yellow('○ 未安装')}  运行 nlsb install 开始使用"))
    cfg = nlsb.CONFIG
    if cfg["detector"] == "model":
        mp = _model_info()
        thr = f"{cfg['threshold']:.2f}" if cfg["threshold"] is not None else "0.50（默认）"
        model = f"模型 {human(mp.stat().st_size)}" if mp.exists() else yellow("模型未下载（nlsb install 会下载）")
        rows.append(("识别", f"分类器 · 阈值 {thr} · {model}"))
    else:
        rows.append(("识别", "规则识别器（不用模型，误伤较多）"))
    blocked, auto, allow = nlsb.load_blocklist(), nlsb.load_md5s(nlsb.AUTOLIST), nlsb.load_md5s(nlsb.ALLOWLIST)
    rows.append(("名单", f"屏蔽 {len(blocked)}（自动 {len(blocked & auto)} · 手动 {len(blocked - auto)}）· 白名单 {len(allow)}"))
    if pid and not nlsb.HISTORY_DONE.exists():
        rows.append(("历史", "后台正在扫描已有缓存（空闲时做，不用管）"))
    bsize = size_of(nlsb.BACKUP) if nlsb.BACKUP.exists() else 0
    rows.append(("备份", ("开" if cfg["backup"] else "关") + (f" · 已占用 {human(bsize)}" if bsize else "")))
    accts = nlsb.accounts()
    rows.append(("QQ", f"{len(accts)} 个账号 · 监听 {'、'.join(nlsb.months_to_watch())}" if accts
                 else red(f"没找到 QQ 数据目录（{tilde(nlsb.QQ_ROOT)}）")))
    for k, v in rows:
        print(f"  {pad(k, 6)}{v}")
    recent, errors = [], []
    if nlsb.LOG.exists():
        with open(nlsb.LOG, "rb") as f:
            f.seek(max(0, nlsb.LOG.stat().st_size - (256 << 10)))
            lines = f.read().decode(errors="replace").splitlines()
        recent = [ln for ln in lines if " auto 奶龙 " in ln][-5:]
        start = max((i for i, ln in enumerate(lines) if " started" in ln), default=0)
        errors = [ln for ln in lines[start:] if "识别失败" in ln or "出错" in ln][-3:]  # 只看这次启动以后的
    if recent:
        print(f"\n  {bold('最近自动屏蔽')}")
        for ln in reversed(recent):
            m = re.match(r"(\S+ \S+) auto 奶龙 score=(\S+) ([0-9a-f]{32})", ln)
            if m:
                print(f"    {m.group(1)}  {short(m.group(3))}  {m.group(2)}")
    if errors:
        print(f"\n  {yellow('最近的错误')}（完整内容见 {tilde(nlsb.LOG)}）")
        for ln in errors:
            print(f"    {dim(ln[:110])}")
    return 0


def cmd_block(a):
    md5s = parse_ids(a.ids)
    found = nlsb.block_md5s(md5s)
    for m in md5s:
        n = found.get(m, 0)
        print(f"{OK} 已屏蔽 {short(m)}  " + (f"替换了 {n} 份缓存" if n else dim("本地还没有，以后收到会自动屏蔽")))
    print(dim("  QQ 里重新打开聊天窗口才能看到效果"))
    warn_if_no_service()
    return 0


def cmd_unblock(a):
    md5s = parse_ids(a.ids)
    blocked = nlsb.load_blocklist()
    res = nlsb.unblock_md5s(md5s)
    for m in md5s:
        restored, removed = res[m]
        if m not in blocked and not restored and not removed:
            print(f"{OK} {short(m)} 本来就没屏蔽，已加入白名单")
        elif restored:
            print(f"{OK} 已还原 {short(m)}  从备份写回 {restored} 份")
        else:
            print(f"{OK} 已放出 {short(m)}  " + dim("没有备份，已删掉占位图，QQ 下次显示时会重新下载"))
    print(dim("  已加入白名单，自动识别不会再屏蔽它们"))
    return 0


def cmd_install(a):
    cfg = nlsb.load_config()
    cfg["backup"] = not a.no_backup
    nlsb.save_config(cfg)
    if not nlsb.accounts():
        print(f"{WARN} 没在 {tilde(nlsb.QQ_ROOT)} 找到 QQ 数据。先登录一次 QQ（NT 版），或用 NLSB_QQ_ROOT 指定目录")
    if nlsb.DETECTOR == "model":
        r = subprocess.run([sys.executable, "-m", "nailong.model", "--fetch"], stdout=subprocess.DEVNULL)
        if r.returncode:
            print(f"{ERR} 模型下载失败。网络不好的话可以设 HF_ENDPOINT=https://hf-mirror.com 再试")
            return 1
    try:
        path = nlsb.service_install()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
        print(f"{ERR} 后台服务装不上：{e}\n  过几秒再运行一次 nlsb install 试试")
        return 1
    for _ in range(20):
        if nlsb.service_pid():
            break
        time.sleep(0.25)
    if not nlsb.service_pid():
        print(f"{ERR} 服务装上了但没启动，看看 {tilde(nlsb.LOG)}")
        return 1
    print(f"{OK} 后台服务已启动，开机自动运行（{tilde(path)}）")
    print(f"  原图备份：{'关（还原时等 QQ 重新下载）' if a.no_backup else '开'}")
    if not nlsb.HISTORY_DONE.exists():
        print(dim("  已有的缓存会在后台空闲时慢慢扫一遍（几千张约半小时到一小时），不用管它"))
    if nlsb.MACOS:
        print(dim("  如果 macOS 弹窗问「是否允许访问其他 App 的数据」，选允许"))
    print(dim("  随时运行 nlsb 查看状态"))
    return 0


def cmd_uninstall(a):
    if a.restore:
        blocked = nlsb.load_blocklist()
        if blocked and not confirm(f"还原全部 {len(blocked)} 个已屏蔽的表情？", a.yes):
            return 1
    nlsb.service_uninstall()
    print(f"{OK} 后台服务已卸载")
    if a.restore:
        nlsb.unblock_md5s(nlsb.load_blocklist(), allow=False)
        print(f"{OK} 已还原全部屏蔽的表情（名单已清空）")
    else:
        print(dim("  已屏蔽的表情仍显示为占位图；要全部还原：nlsb uninstall --restore"))
    print(dim(f"  名单、备份和日志都在 {tilde(nlsb.HOME)}，不需要可以整个删掉"))
    return 0


def cmd_run(a):
    nlsb.Watcher().loop()


# ---------- 入口 ----------

class _Fmt(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=30)

    def _format_action(self, action):  # 直接列出各个命令，不要 argparse 默认的 "<命令>" 那一行
        if isinstance(action, argparse._SubParsersAction):
            return "".join(self._format_action(a) for a in action._get_subactions())
        return super()._format_action(action)


def build_parser():
    p = argparse.ArgumentParser(
        prog="nlsb", formatter_class=_Fmt, usage="nlsb [<命令>] [参数]",
        description="在本机屏蔽 QQ 表情，并自动识别、屏蔽新收到的奶龙。\n只改写本机的 QQ 缓存文件，原图默认先备份，随时可以还原。",
        epilog="不带参数运行 nlsb 查看状态。\n表情 ID：在 QQ 里对表情「另存为」，把得到的文件名（或路径）给 nlsb 就行。")
    p.add_argument("-V", "--version", action="version", version=f"nlsb {nlsb.__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<命令>", title="命令")

    def add(name, fn, help):
        sp = sub.add_parser(name, help=help, description=help, formatter_class=_Fmt)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("install", cmd_install, "安装后台服务并开机自启（重复运行可更新设置）")
    sp.add_argument("--no-backup", action="store_true", help="屏蔽时不备份原图（省磁盘，还原时只能等 QQ 重新下载）")
    sp = add("block", cmd_block, "屏蔽指定的表情")
    sp.add_argument("ids", nargs="+", metavar="表情")
    sp = add("unblock", cmd_unblock, "放出指定的表情，以后也不会再自动屏蔽它")
    sp.add_argument("ids", nargs="+", metavar="表情")
    sp = add("uninstall", cmd_uninstall, "卸载后台服务")
    sp.add_argument("--restore", action="store_true", help="同时还原所有已屏蔽的表情")
    sp.add_argument("-y", "--yes", action="store_true", help="不用确认")
    for parser in (p, *sub.choices.values()):
        parser._optionals.title = "选项"
        parser._positionals.title = "参数"
        for act in parser._actions:
            if act.dest == "help":
                act.help = "显示帮助"
    p._actions[1].help = "显示版本"
    run = sub.add_parser("run")  # 后台服务的入口，不在帮助里列出
    run.set_defaults(fn=cmd_run)
    sub._choices_actions = [x for x in sub._choices_actions if x.dest != "run"]
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        a.fn = cmd_status
    if a.cmd != "run":
        nlsb.log_to_file()
    try:
        return a.fn(a) or 0
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
