"""检查自动屏蔽的结果：生成本地标注页（自动屏蔽了的、可能漏了的），再把标注结果应用回来。

  python tools/review.py              生成页面并在浏览器打开
  python tools/review.py --apply      应用页面「复制结果」的内容（读剪贴板；也可以给文件路径）

页面在数据目录的 review/ 下，图片用符号链接指向备份和 QQ 缓存，不复制、不联网。
标注会记进数据目录的 labels.tsv，可以拿去给 tools/train_head.py 重新训练。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from nlsb import (ALLOWLIST, AUTOLIST, BACKUP, HOME, LABELS, MD5_RE, accounts, block_md5s, is_placeholder,  # noqa: E402
                 iter_cache_dirs, load_blocklist, load_md5s, load_scores, log_to_file, unblock_md5s)

DIR = HOME / "review"
MAYBE_MIN = 0.15   # 没屏蔽、但分数不低于这个的，放进"可能漏了"
MAYBE_MAX = 300


def _preview(md5, hint=None):
    """找一份能显示的原图：优先备份，其次 QQ 缓存里不是占位图的那份。"""
    if hint and os.path.isfile(hint) and not is_placeholder(hint):
        return hint
    files = [p for p in (BACKUP / md5).rglob("*") if p.is_file()]
    if not files:
        files = [p for nt in accounts() for d in iter_cache_dirs(nt) for p in d.glob(f"{md5}*")
                 if p.is_file() and not is_placeholder(p)]
    files.sort(key=lambda p: (p.parent.name == "Ori", p.stat().st_size), reverse=True)
    return str(files[0]) if files else None


def build():
    """生成页面，返回 (页面路径, 屏蔽中张数, 可能漏了张数, 没有原图可看的张数)。"""
    img = DIR / "img"
    img.mkdir(parents=True, exist_ok=True)
    for e in os.scandir(img):  # 只清理自己建的符号链接
        if e.is_symlink():
            os.unlink(e.path)
    scores = load_scores()
    blocked, allow = load_blocklist(), load_md5s(ALLOWLIST)

    def item(md5, path):
        name = f"{md5}{Path(path).suffix or '.img'}"
        os.symlink(path, img / name)
        s = scores.get(md5, (None,))[0]
        return {"md5": md5, "src": f"img/{name}", "score": None if s is None else round(s, 2)}

    keep, missing = [], 0
    for md5 in sorted(load_md5s(AUTOLIST) & blocked):
        p = _preview(md5)
        if p:
            keep.append(item(md5, p))
        else:
            missing += 1
    keep.sort(key=lambda x: (x["score"] is None, x["score"] or 0))  # 分数低的最可能是误伤，排前面
    cand = sorted(((s, m, p) for m, (s, p) in scores.items()
                   if s >= MAYBE_MIN and m not in blocked and m not in allow), reverse=True)
    maybe = []
    for s, m, p in cand:
        if len(maybe) >= MAYBE_MAX:
            break
        if (p := _preview(m, p)):
            maybe.append(item(m, p))
    page = DIR / "index.html"
    page.write_text(PAGE.replace("__DATA__", json.dumps({"keep": keep, "maybe": maybe, "missing": missing},
                                                        ensure_ascii=False)))
    return page, len(keep), len(maybe), missing


def parse(text):
    """页面导出的文本 → (要放出的, 要屏蔽的)。"""
    release, block = set(), set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and MD5_RE.fullmatch(parts[1]):
            {"not": release, "nailong": block}.get(parts[0], set()).add(parts[1].lower())
    return release, block - release


def apply(release, block):
    """放出 release、屏蔽 block，并把人工结论记进 labels.tsv（以后可用来重新训练）。"""
    paths = {m: _preview(m) for m in release | block}
    release &= load_blocklist()
    if release:
        unblock_md5s(release)
    if block:
        block_md5s(block)
    with open(LABELS, "a") as f:
        now = int(time.time())
        for m in sorted(release):
            f.write(f"{now}\t{m}\t0\t{paths.get(m) or ''}\n")
        for m in sorted(block):
            f.write(f"{now}\t{m}\t1\t{paths.get(m) or ''}\n")
    return release, block


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>nlsb review</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1f;--card:#fff;--mute:#86868b;--line:#0000001a;--mark:#e5484d;--ok:#30a46c}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--card:#232326;--mute:#9a9aa0;--line:#ffffff1f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0 12px 0 0}
button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
button.primary{background:var(--ok);color:#fff;border-color:var(--ok)}
.hint{color:var(--mute);margin-top:6px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;padding:12px 16px 40px}
.c{position:relative;background:var(--card);border-radius:10px;aspect-ratio:1;cursor:pointer;border:3px solid transparent;overflow:hidden}
.c img{width:100%;height:100%;object-fit:contain}
.c span{position:absolute;left:5px;top:5px;font-size:11px;background:#000a;color:#fff;padding:0 5px;border-radius:4px}
.c.m{border-color:var(--mark)}.c.m::after{content:"✕";position:absolute;right:7px;top:1px;color:var(--mark);font:700 22px sans-serif}
.empty{grid-column:1/-1;color:var(--mute);padding:40px 0;text-align:center}
</style></head><body>
<header><div class="row"><h1>nlsb review</h1><button id="t-keep"></button><button id="t-maybe"></button>
<span style="flex:1"></span><button id="clear">清空标记</button><button id="exp" class="primary">复制结果</button></div>
<div class="hint" id="hint"></div></header><main id="g"></main><script>
const D=__DATA__,KEY="nlsb-review-marks",M={keep:new Set(),maybe:new Set()};
try{const s=JSON.parse(localStorage.getItem(KEY)||"{}");for(const k in M)(s[k]||[]).forEach(x=>M[k].add(x))}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify({keep:[...M.keep],maybe:[...M.maybe]}))}catch(e){}};
const HINT={keep:"自动屏蔽了的表情。点选其中【不是奶龙】的，会被放出。分数低的排在前面"+(D.missing?`（另有 ${D.missing} 个没有备份，看不到原图）`:""),
            maybe:"没屏蔽、但分数偏高的表情。点选其中【是奶龙】的，会被屏蔽"};
const NAME={keep:"屏蔽中",maybe:"可能漏了"};let tab="keep";
function head(){for(const k in NAME){const b=document.getElementById("t-"+k);
  b.textContent=`${NAME[k]} ${D[k].length}`+(M[k].size?`（已标 ${M[k].size}）`:"");b.className=k==tab?"on":""}
  document.getElementById("hint").textContent=HINT[tab]}
function show(){const g=document.getElementById("g");g.innerHTML="";
  if(!D[tab].length)g.innerHTML='<div class="empty">这里没有需要看的</div>';
  for(const it of D[tab]){const c=document.createElement("div");c.className="c"+(M[tab].has(it.md5)?" m":"");
    c.innerHTML=`<img loading="lazy" src="${it.src}">`+(it.score==null?"":`<span>${it.score.toFixed(2)}</span>`);
    c.onclick=()=>{M[tab].has(it.md5)?M[tab].delete(it.md5):M[tab].add(it.md5);c.classList.toggle("m");save();head()};g.appendChild(c)}
  head();scrollTo(0,0)}
for(const k in NAME)document.getElementById("t-"+k).onclick=()=>{tab=k;show()};
document.getElementById("clear").onclick=()=>{if(confirm("清空所有标记？")){M.keep.clear();M.maybe.clear();save();show()}};
document.getElementById("exp").onclick=async()=>{
  const t=[...[...M.keep].map(x=>"not "+x),...[...M.maybe].map(x=>"nailong "+x)].join("\n");
  if(!t){alert("还没有标记任何表情");return}
  try{await navigator.clipboard.writeText(t);alert(`已复制 ${M.keep.size+M.maybe.size} 条。\n回到终端运行：python tools/review.py --apply`)}
  catch(e){prompt("复制下面的内容，再在终端运行 python tools/review.py --apply：",t)}};
show();
</script></body></html>"""


def read_clipboard():
    for cmd in (["pbpaste"], ["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "-bo"]):
        if shutil.which(cmd[0]):
            return subprocess.run(cmd, capture_output=True, text=True).stdout
    return ""


def main(argv):
    log_to_file()
    if "--apply" not in argv:
        page, n_keep, n_maybe, missing = build()
        print(f"标注页：{page}\n  屏蔽中 {n_keep} 个（点出不是奶龙的），可能漏了 {n_maybe} 个（点出是奶龙的）")
        if missing:
            print(f"  另有 {missing} 个自动屏蔽的没有备份，页面里看不到")
        webbrowser.open(page.as_uri())
        return 0
    rest = argv[argv.index("--apply") + 1:]
    text = Path(rest[0]).read_text() if rest else read_clipboard()
    release, block = parse(text)
    if not release and not block:
        print("没读到标注结果：先在页面里点「复制结果」")
        return 1
    if input(f"放出 {len(release)} 个、屏蔽 {len(block)} 个，继续？[y/N] ").strip().lower() != "y":
        return 1
    release, block = apply(release, block)
    print(f"已放出 {len(release)} 个（加入白名单），屏蔽 {len(block)} 个；标注记在 {LABELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
