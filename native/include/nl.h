/* nailong-hpc 模块间接口。三个模块（decode / prep / feat）各自实现，最后链接进 libnailong.so。
 *
 * 正确性基准：~/Programs/nailong-detector（Python 参考实现）。每一步的输出都必须与参考实现逐位相同。
 */
#ifndef NL_H
#define NL_H
#include <stddef.h>
#include <stdint.h>

#define NL_OK 0
#define NL_UNSUPPORTED 1   /* 本模块不处理这种输入，调用方回退到 Python 参考实现 */
#define NL_ERROR 2         /* 输入损坏（参考实现也会抛异常） */

#define NL_EYE_SIZE 256

/* ---------- A. decode ----------
 * 等价于参考实现 features._load 中直到 convert 为止的部分：
 *   im = Image.open(BytesIO(buf))
 *   if im.format == "JPEG": im.draft("RGB", (256, 256))        # FAST_JPEG=True 时 d = size = 256
 *   has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
 *   im = im.convert("RGBA" if has_alpha else "RGB")
 * 输出 px 与 np.asarray(im) 逐字节相同。ch = 4 当且仅当 has_alpha。px 由 malloc 分配，调用方 free。 */
typedef struct {
    int w, h, ch;
    uint8_t *px;
} nl_img;

int nl_decode(const uint8_t *buf, size_t len, nl_img *out);

/* ---------- B. prep ----------
 * 等价于 _load 剩余部分 + extract 里生成 small + 两次 hsv()：
 *   orig = max(w, h)
 *   f = max(w, h) // 256; if f >= 2: im = im.reduce(f)
 *   k = 256 / max(w, h)
 *   im = im.resize((max(1, round(w*k)), max(1, round(h*k))), Image.BILINEAR)   # Python round：四舍六入五成双
 *   if has_alpha: im = alpha_composite(白底 RGBA, im).convert("RGB")
 *   big = im
 *   small = big.resize((max(1, big.width // 2), max(1, big.height // 2)), Image.BILINEAR)   # FAST_SMALL=False
 *   bH,bS,bV = big.convert("HSV")；sH,sS,sV = small.convert("HSV")
 * 所有平面由 nl_prep_free 释放。 */
typedef struct {
    int orig;
    int bw, bh, sw, sh;
    uint8_t *bH, *bS, *bV;
    uint8_t *sH, *sS, *sV;
} nl_prep;

int nl_prepare(const nl_img *in, nl_prep *out);
void nl_prep_free(nl_prep *p);

/* ---------- C. feat ----------
 * 等价于 extract(src, min_yellow=gate) 在拿到 big/small 之后的全部计算：
 *   f = color_features(small, min_yellow); f["src"] = orig
 *   若 skipped：眼睛特征全为 0；否则 f.update(eye_features(big))
 * out 按 NL_FEATURE_NAMES 的顺序写 20 个 double，每个都必须与 Python float 逐位相同。
 * *skipped 返回是否在颜色门槛处提前退出。 */
#define NL_NFEAT 20
extern const char *const NL_FEATURE_NAMES[NL_NFEAT];
/* "yfrac","yhue","ysat","yaspect","yfill","outline","edges","ytex","shading","belly",
 * "red","ofrac","mouth","src","green_best","green_pair","yellow_eye","glint","sclera","tex256" */

int nl_features(const nl_prep *p, double min_yellow, double out[NL_NFEAT], int *skipped);

/* 与 nailong.score.score 逐位相同。params 按 NL_PARAM_NAMES 顺序（= list(nailong.PRESETS["balanced"])）：
 * "gate","gate_o","gate_lo","eye_gb","g0","g1","w_g","w_pair","gl","w_glint","ye","w_ye","sc","w_sc",
 * "hue","w_hue","asp","w_asp","red","w_red","out","w_out","tex","w_tex","e_lo","e_hi","w_e",
 * "belly","w_belly","shade","w_shade","orange","w_orange","mouth","w_mouth","tex2","w_tex2","small","w_small",
 * "gp","w_gp","thr"
 * 提前退出门槛 min_yellow = min(gate, gate_lo)（见 nailong.min_yellow）。 */
#define NL_NPARAM 42
extern const char *const NL_PARAM_NAMES[NL_NPARAM];
double nl_score(const double f[NL_NFEAT], const double params[NL_NPARAM]);

#endif
