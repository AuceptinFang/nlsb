"""GPU 后端（wgpu：Linux/Windows 上走 Vulkan，macOS 上走 Metal）。

与 CPU 版逐位一致：GPU 上只有整数运算（掩码、积分图、方框计数、阈值的交叉相乘比较、分子取最大），
最后的除法在 CPU 上用 numpy float32 完成。

分工：CPU 解码 / 缩放 / HSV / 颜色特征（便宜），GPU 批量做眼睛检测和 256px 纹理（贵）。

    from nailong.gpu import Detector
    det = Detector("aggressive")
    results = det.detect_many(paths_or_bytes)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import features as F
from .score import PRESETS, score

S = F.EYE_SIZE           # 256
S1 = S + 1
IMG = S1 * S1            # 每张图的积分图元素数
NST = 40                 # 每张图的统计槽：4 个量 × 8 个尺度 + 纹理和 + 纹理数 + 候选数
CAP = 2048               # 每张图最多记录多少个绿眼峰候选（超出则该图回退到 CPU）
ENTRY = 5                # 每个候选：(y<<16|x) + 8 个尺度的绿虹膜分子（两两打包）

assert len(F.EYE_RADII) == 8


def _wgsl() -> str:
    rings = F.RINGS
    arr = lambda i: ", ".join(f"{t[i]}u" for t in rings)  # noqa: E731
    arri = lambda i: ", ".join(f"{t[i]}" for t in rings)  # noqa: E731
    return f"""
const S: u32 = {S}u;
const S1: u32 = {S1}u;
const IMG: u32 = {IMG}u;
const NST: u32 = {NST}u;
const CAP: u32 = {CAP}u;
const ENTRY: u32 = {ENTRY}u;
const MAXC: u32 = {F.MAX_CANDIDATES}u;
const R1 = array<i32, 8>({arri(0)});
const R2 = array<i32, 8>({arri(1)});
const R3 = array<i32, 8>({arri(2)});
const A1 = array<u32, 8>({arr(3)});
const RN = array<u32, 8>({arr(4)});
const ON = array<u32, 8>({arr(5)});

@group(0) @binding(0) var<storage, read> hsv: array<u32>;
@group(0) @binding(1) var<storage, read> info: array<u32>;        // 每张图 4 个：w, h, stride, 0
@group(0) @binding(2) var<storage, read_write> ii0: array<u32>;   // 深色 | 绿 << 16
@group(0) @binding(3) var<storage, read_write> ii1: array<u32>;   // 黄 | 白 << 16
@group(0) @binding(4) var<storage, read_write> ii2: array<u32>;   // 候选点
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cands: array<u32>;

var<workgroup> wtex: array<atomic<u32>, 2>;
var<workgroup> wmax: array<atomic<u32>, 32>;

fn hh(p: u32) -> u32 {{ return p & 255u; }}
fn ss(p: u32) -> u32 {{ return (p >> 8u) & 255u; }}
fn vv(p: u32) -> u32 {{ return (p >> 16u) & 255u; }}
fn is_yel(p: u32) -> bool {{ return hh(p) >= 22u && hh(p) <= 48u && ss(p) >= 60u && vv(p) >= 100u; }}
fn adiff(a: u32, b: u32) -> u32 {{ return select(b - a, a - b, a >= b); }}

// 1) 掩码写进积分图缓冲区（位置偏移 +1 行 +1 列），顺便累计 256px 纹理
@compute @workgroup_size(256)
fn masks(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    if (li < 2u) {{ atomicStore(&wtex[li], 0u); }}
    workgroupBarrier();
    let b = gid.y;
    let i = gid.x;
    let x = i % S;
    let y = i / S;
    let w = info[b * 4u];
    let h = info[b * 4u + 1u];
    let stride = info[b * 4u + 2u];
    if (x < w && y < h) {{
        let p = hsv[b * S * S + i];
        let d = vv(p) < 50u;
        let g = hh(p) >= 38u && hh(p) <= 110u && ss(p) >= 45u && vv(p) >= 40u && vv(p) <= 235u;
        let yl = is_yel(p);
        let wt = vv(p) >= 200u && ss(p) < 60u;
        let c = d && ((x + y) % stride == 0u);
        let o = b * IMG + (y + 1u) * S1 + x + 1u;
        ii0[o] = u32(d) | (u32(g) << 16u);
        ii1[o] = u32(yl) | (u32(wt) << 16u);
        ii2[o] = u32(c);
        if (yl) {{
            var sum = 0u;
            var cnt = 0u;
            if (x + 1u < w) {{
                let q = hsv[b * S * S + i + 1u];
                if (is_yel(q)) {{ sum += adiff(vv(p), vv(q)); cnt += 1u; }}
            }}
            if (y + 1u < h) {{
                let q = hsv[b * S * S + i + S];
                if (is_yel(q)) {{ sum += adiff(vv(p), vv(q)); cnt += 1u; }}
            }}
            if (cnt > 0u) {{ atomicAdd(&wtex[0], sum); atomicAdd(&wtex[1], cnt); }}
        }}
    }}
    workgroupBarrier();
    if (li < 2u) {{
        let v = atomicLoad(&wtex[li]);
        if (v > 0u) {{ atomicAdd(&stats[b * NST + 32u + li], v); }}
    }}
}}

// 2) 积分图：按行前缀和（u32 回绕加法，打包的两段各自独立）
@compute @workgroup_size(64)
fn scan_rows(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let b = gid.y;
    let y = gid.x;
    if (y >= info[b * 4u + 1u]) {{ return; }}
    let base = b * IMG + (y + 1u) * S1;
    let w = info[b * 4u];
    var a0 = 0u; var a1 = 0u; var a2 = 0u;
    for (var x = 1u; x <= w; x++) {{
        a0 += ii0[base + x]; ii0[base + x] = a0;
        a1 += ii1[base + x]; ii1[base + x] = a1;
        a2 += ii2[base + x]; ii2[base + x] = a2;
    }}
}}

// 3) 积分图：按列前缀和
@compute @workgroup_size(64)
fn scan_cols(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let b = gid.y;
    let x = gid.x + 1u;
    if (x > info[b * 4u]) {{ return; }}
    let h = info[b * 4u + 1u];
    var a0 = 0u; var a1 = 0u; var a2 = 0u;
    for (var y = 1u; y <= h; y++) {{
        let o = b * IMG + y * S1 + x;
        a0 += ii0[o]; ii0[o] = a0;
        a1 += ii1[o]; ii1[o] = a1;
        a2 += ii2[o]; ii2[o] = a2;
    }}
}}

fn box2(o: u32, y: i32, x: i32, r: i32, w: i32, h: i32) -> vec2<u32> {{
    let y0 = u32(clamp(y - r, 0, h)) * S1;
    let y1 = u32(clamp(y + r + 1, 0, h)) * S1;
    let x0 = u32(clamp(x - r, 0, w));
    let x1 = u32(clamp(x + r + 1, 0, w));
    let s0 = ii0[o + y1 + x1] - ii0[o + y0 + x1] - ii0[o + y1 + x0] + ii0[o + y0 + x0];
    let s1 = ii1[o + y1 + x1] - ii1[o + y0 + x1] - ii1[o + y1 + x0] + ii1[o + y0 + x0];
    return vec2<u32>(s0, s1);
}}

// 4) 中心-环绕滤波
@compute @workgroup_size(256)
fn eyes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    if (li < 32u) {{ atomicStore(&wmax[li], 0u); }}
    workgroupBarrier();
    let b = gid.y;
    let i = gid.x;
    let x = i % S;
    let y = i / S;
    let w = info[b * 4u];
    let h = info[b * 4u + 1u];
    let o = b * IMG;
    var live = x < w && y < h;
    if (live) {{
        let me = o + (y + 1u) * S1 + x + 1u;
        live = (ii2[me] - ii2[me - 1u] - ii2[me - S1] + ii2[me - S1 - 1u]) != 0u;
    }}
    if (live) {{
        let tot = ii2[o + h * S1 + w];
        if (tot > MAXC) {{  // 与 CPU 一致：按光栅顺序每 stp 个取一个
            let stp = tot / MAXC + 1u;
            let rnk = ii2[o + y * S1 + w] + ii2[o + (y + 1u) * S1 + x] - ii2[o + y * S1 + x];
            live = (rnk % stp) == 0u;
        }}
    }}
    if (live) {{
        let yi = i32(y); let xi = i32(x); let wi = i32(w); let hi = i32(h);
        var gnum: array<u32, 8>;
        var found = false;
        for (var k = 0u; k < 8u; k++) {{
            let c = box2(o, yi, xi, R1[k], wi, hi);
            let m = box2(o, yi, xi, R2[k], wi, hi);
            let q = box2(o, yi, xi, R3[k], wi, hi);
            let cd = c.x & 0xFFFFu; let cg = c.x >> 16u; let cy = c.y & 0xFFFFu; let cw = c.y >> 16u;
            let md = m.x & 0xFFFFu; let mg = m.x >> 16u; let my = m.y & 0xFFFFu; let mw = m.y >> 16u;
            let og = q.x >> 16u; let oy = q.y & 0xFFFFu;
            let a1 = A1[k]; let rn = RN[k]; let on = ON[k];
            let pupil = 20u * cd >= 11u * a1 && 2u * (md - cd) <= rn;
            let skin = 10u * (oy - my) >= 3u * on && 4u * (og - mg) <= on;
            let glint = 20u * cd >= 7u * a1 && 50u * cw >= a1 && 10u * cw <= 3u * a1
                        && 5u * (my - cy) >= 2u * rn && skin;
            gnum[k] = 0u;
            if (pupil && skin) {{
                gnum[k] = mg - cg;
                atomicMax(&wmax[k], mg - cg);
                if (5u * (mg - cg) >= rn) {{ found = true; }}
            }}
            if (pupil) {{
                atomicMax(&wmax[8u + k], my - cy);
                atomicMax(&wmax[16u + k], mw - cw);
            }}
            if (glint) {{ atomicMax(&wmax[24u + k], my - cy); }}
        }}
        if (found) {{  // 绿虹膜比例 ≥ 0.2 的候选，交给 CPU 做非极大值抑制
            let pos = atomicAdd(&stats[b * NST + 34u], 1u);
            if (pos < CAP) {{
                let e = (b * CAP + pos) * ENTRY;
                cands[e] = (y << 16u) | x;
                for (var k = 0u; k < 4u; k++) {{
                    cands[e + 1u + k] = gnum[2u * k] | (gnum[2u * k + 1u] << 16u);
                }}
            }}
        }}
    }}
    workgroupBarrier();
    if (li < 32u) {{
        let v = atomicLoad(&wmax[li]);
        if (v > 0u) {{ atomicMax(&stats[b * NST + li], v); }}
    }}
}}
"""


class Detector:
    """批量识别。batch：一次送 GPU 的图片数；workers：CPU 预处理线程数。"""

    def __init__(self, preset: str = "balanced", params: dict | None = None, batch: int = 64,
                 workers: int | None = None, power_preference: str = "high-performance"):
        import wgpu

        self.wgpu = wgpu
        self.p = params or PRESETS[preset]
        self.batch = batch
        self.workers = workers or min(8, os.cpu_count() or 1)
        adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self.adapter_info = adapter.info
        self.device = adapter.request_device_sync()
        module = self.device.create_shader_module(code=_wgsl())
        U = wgpu.BufferUsage
        st = wgpu.BufferBindingType
        layout = self.device.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": st.read_only_storage if i < 2 else st.storage}}
            for i in range(7)
        ])
        playout = self.device.create_pipeline_layout(bind_group_layouts=[layout])
        self.pipes = {name: self.device.create_compute_pipeline(
            layout=playout, compute={"module": module, "entry_point": name})
            for name in ("masks", "scan_rows", "scan_cols", "eyes")}
        B = batch
        self.buf = {
            "hsv": self.device.create_buffer(size=B * S * S * 4, usage=U.STORAGE | U.COPY_DST),
            "meta": self.device.create_buffer(size=B * 16, usage=U.STORAGE | U.COPY_DST),
            "ii0": self.device.create_buffer(size=B * IMG * 4, usage=U.STORAGE | U.COPY_DST),
            "ii1": self.device.create_buffer(size=B * IMG * 4, usage=U.STORAGE | U.COPY_DST),
            "ii2": self.device.create_buffer(size=B * IMG * 4, usage=U.STORAGE | U.COPY_DST),
            "stats": self.device.create_buffer(size=B * NST * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST),
            "cands": self.device.create_buffer(size=B * CAP * ENTRY * 4, usage=U.STORAGE | U.COPY_SRC),
        }
        order = ("hsv", "meta", "ii0", "ii1", "ii2", "stats", "cands")
        self.group = self.device.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": self.buf[k], "offset": 0, "size": self.buf[k].size}}
            for i, k in enumerate(order)
        ])

    # ---- CPU 预处理 ----
    def _prepare(self, src):
        big, orig = F._load(src, S, 0)
        small = big.resize((max(1, big.width // 2), max(1, big.height // 2)), F.Image.BILINEAR) \
            if not (F.FAST_SMALL and big.width >= 2 and big.height >= 2) else big.reduce(2)
        f = F.color_features(small, min(self.p["gate"], self.p["gate_lo"]))
        f["src"] = float(orig)
        if f.get("skipped"):  # 过不了颜色门槛，其余特征不影响分数
            f.update(dict.fromkeys(F.EYE_KEYS, 0.0))
            return f, None
        H, Sa, V = F.hsv(big)
        return f, (H, Sa, V)

    # ---- GPU 批处理 ----
    def _eyes_batch(self, hsvs):
        B = len(hsvs)
        packed = np.zeros((self.batch, S, S), np.uint32)
        meta = np.zeros((self.batch, 4), np.uint32)
        for j, (H, Sa, V) in enumerate(hsvs):
            h, w = V.shape
            packed[j, :h, :w] = H.astype(np.uint32) | (Sa.astype(np.uint32) << 8) | (V.astype(np.uint32) << 16)
            meta[j] = (w, h, F.CANDIDATE_STRIDE, 0)
        q = self.device.queue
        q.write_buffer(self.buf["hsv"], 0, packed)
        q.write_buffer(self.buf["meta"], 0, meta)
        enc = self.device.create_command_encoder()
        for k in ("ii0", "ii1", "ii2", "stats"):
            enc.clear_buffer(self.buf[k])
        cp = enc.begin_compute_pass()
        cp.set_bind_group(0, self.group)
        for name, groups in (("masks", (S * S // 256, B, 1)), ("scan_rows", (S // 64, B, 1)),
                             ("scan_cols", (S // 64, B, 1)), ("eyes", (S * S // 256, B, 1))):
            cp.set_pipeline(self.pipes[name])
            cp.dispatch_workgroups(*groups)
        cp.end()
        q.submit([enc.finish()])
        stats = np.frombuffer(q.read_buffer(self.buf["stats"], 0, B * NST * 4), np.uint32).reshape(B, NST)
        ncand = stats[:, 34]
        cands = None
        if ncand.any():
            cands = np.frombuffer(q.read_buffer(self.buf["cands"], 0, B * CAP * ENTRY * 4),
                                  np.uint32).reshape(B, CAP, ENTRY)
        rn = np.array([t[4] for t in F.RINGS], np.float32)
        radii = np.array([t[0] for t in F.RINGS], np.int16)
        out = []
        for j in range(B):
            if ncand[j] > CAP:  # 候选溢出：这张图回退到 CPU
                out.append(None)
                continue
            st = stats[j]
            frac = lambda a: float((st[a:a + 8].astype(np.float32) / rn).max())  # noqa: E731
            f = dict(green_best=frac(0), yellow_eye=frac(8), sclera=frac(16), glint=frac(24),
                     tex256=int(st[32]) / max(1, int(st[33])), green_pair=0.0)
            n = int(ncand[j])
            if n:
                e = cands[j, :n]
                order = np.argsort(e[:, 0], kind="stable")  # (y<<16|x) 排序 = 光栅顺序
                e = e[order]
                ys = (e[:, 0] >> 16).astype(np.int64)
                xs = (e[:, 0] & 0xFFFF).astype(np.int64)
                nums = np.empty((n, 8), np.uint32)
                nums[:, 0::2] = e[:, 1:] & 0xFFFF
                nums[:, 1::2] = e[:, 1:] >> 16
                g_best = np.zeros(n, np.float32)
                g_rad = np.zeros(n, np.int16)
                for k in range(8):  # 与 CPU 相同的更新顺序（严格大于才更新）
                    g = nums[:, k].astype(np.float32) / rn[k]
                    upd = g > g_best
                    g_best[upd] = g[upd]
                    g_rad[upd] = radii[k]
                f["green_pair"] = F.green_pair(g_best, g_rad, ys, xs)
            out.append(f)
        return out

    def extract_many(self, srcs):
        """返回每张图的特征 dict（与 nailong.features.extract(src, min_yellow=gate) 一致）。"""
        results = [None] * len(srcs)
        pending = []  # (index, color_feats, hsv)
        chunk = self.batch * 2  # 分块提交，内存只和批大小有关，不随图片总数增长
        with ThreadPoolExecutor(self.workers) as ex:
            for start in range(0, len(srcs), chunk):
                part = srcs[start:start + chunk]
                for i, (f, h) in enumerate(ex.map(self._prepare, part), start):
                    if h is None:
                        results[i] = f
                    else:
                        pending.append((i, f, h))
                    if len(pending) == self.batch:
                        self._flush(pending, results, srcs)
                        pending = []
            if pending:
                self._flush(pending, results, srcs)
        return results

    def _flush(self, pending, results, srcs):
        eyes = self._eyes_batch([h for _, _, h in pending])
        for (i, f, h), e in zip(pending, eyes):
            f.update(e if e is not None else F.eye_features_hsv(*h))  # 候选溢出时回退到 CPU
            results[i] = f

    def detect_many(self, srcs):
        from . import Result
        out = []
        for f in self.extract_many(srcs):
            s = score(f, self.p)
            out.append(Result(s >= self.p["thr"], s, self.p["thr"], f))
        return out
