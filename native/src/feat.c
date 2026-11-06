/* Module C: feature extraction + scoring, bit-exact with nailong-detector (numpy 2.5.4).
 *
 * Reference: src/nailong/features.py  color_features / eye_features_hsv, src/nailong/score.py.
 * All counts are computed with exact integer arithmetic; where the reference uses float32
 * (box_blur, std/mean with numpy pairwise summation, eye ratios) the same float32 operations
 * are performed in the same order.  Build with -ffp-contract=off (no FMA contraction) and
 * without -ffast-math.
 *
 * AVX2 is used when available (-march=native on the target); a portable scalar path computes
 * identical results otherwise.
 *
 * Thread safety: all scratch memory is per-thread (lazily malloc'ed, grown on demand, freed by a
 * pthread key destructor at thread exit).  No global mutable state besides pthread_once.
 */
#include "nl.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

const char *const NL_FEATURE_NAMES[NL_NFEAT] = {
    "yfrac", "yhue", "ysat", "yaspect", "yfill", "outline", "edges", "ytex", "shading", "belly",
    "red", "ofrac", "mouth", "src", "green_best", "green_pair", "yellow_eye", "glint", "sclera", "tex256",
};

const char *const NL_PARAM_NAMES[NL_NPARAM] = {
    "gate", "gate_o", "gate_lo", "eye_gb", "g0", "g1", "w_g", "w_pair", "gl", "w_glint", "ye", "w_ye",
    "sc", "w_sc", "hue", "w_hue", "asp", "w_asp", "red", "w_red", "out", "w_out", "tex", "w_tex",
    "e_lo", "e_hi", "w_e", "belly", "w_belly", "shade", "w_shade", "orange", "w_orange", "mouth",
    "w_mouth", "tex2", "w_tex2", "small", "w_small", "gp", "w_gp", "thr",
};

enum {
    F_YFRAC, F_YHUE, F_YSAT, F_YASPECT, F_YFILL, F_OUTLINE, F_EDGES, F_YTEX, F_SHADING, F_BELLY,
    F_RED, F_OFRAC, F_MOUTH, F_SRC, F_GREEN_BEST, F_GREEN_PAIR, F_YELLOW_EYE, F_GLINT, F_SCLERA,
    F_TEX256,
};

enum {
    P_GATE, P_GATE_O, P_GATE_LO, P_EYE_GB, P_G0, P_G1, P_W_G, P_W_PAIR, P_GL, P_W_GLINT, P_YE, P_W_YE,
    P_SC, P_W_SC, P_HUE, P_W_HUE, P_ASP, P_W_ASP, P_RED, P_W_RED, P_OUT, P_W_OUT, P_TEX, P_W_TEX,
    P_E_LO, P_E_HI, P_W_E, P_BELLY, P_W_BELLY, P_SHADE, P_W_SHADE, P_ORANGE, P_W_ORANGE, P_MOUTH,
    P_W_MOUTH, P_TEX2, P_W_TEX2, P_SMALL, P_W_SMALL, P_GP, P_W_GP, P_THR,
};

/* features.MAX_CANDIDATES / CANDIDATE_STRIDE=2 / RINGS (EYE_RADII = 1,2,3,4,6,8,11,15) */
#define MAX_CANDIDATES 12000
#define NRING 8
typedef struct { int r, r2, r3, a1, rn, on; } ring_t;
static const ring_t RINGS[NRING] = {
    {1, 3, 7, 9, 40, 176},       {2, 5, 10, 25, 96, 320},     {3, 8, 14, 49, 240, 552},
    {4, 10, 17, 81, 360, 784},   {6, 14, 25, 169, 672, 1760}, {8, 19, 32, 289, 1232, 2704},
    {11, 25, 43, 529, 2072, 4968}, {15, 34, 57, 961, 3800, 8464},
};

#ifdef NL_STATS /* test-only instrumentation */
#include <x86intrin.h>
long long nlt_stats[32];
#define STAT(i, v) (nlt_stats[i] += (v))
#define TICK(i) (nlt_stats[8 + (i)] += (long long)__rdtsc())
#define TOCK(i) (nlt_stats[8 + (i)] -= (long long)__rdtsc())
#else
#define STAT(i, v) ((void)0)
#define TICK(i) ((void)0)
#define TOCK(i) ((void)0)
#endif

/* ------------------------------------------------------------------ per-thread scratch */

typedef struct {
    void *p;
    size_t cap;
} buf_t;

enum { B_SMASK, B_ROW16, B_F32A, B_F32B, B_BMASK, B_II64, B_CAND, B_PEAK, B_HIST, NBUF };

typedef struct {
    buf_t b[NBUF];
} scratch_t;

static pthread_key_t g_key;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static __thread scratch_t *t_scr;
static uint8_t COMPRESS_LUT[256][8]; /* positions of the set bits of an 8-bit mask */
static int SKIN_T;                    /* min 7x7 skin count c with (float)c / 49.f >= 0.35f */

static void scr_free(void *v) {
    scratch_t *s = v;
    if (!s) return;
    for (int i = 0; i < NBUF; i++) free(s->b[i].p);
    free(s);
}
static void once_init(void) {
    pthread_key_create(&g_key, scr_free);
    for (int m = 0; m < 256; m++) {
        int k = 0;
        for (int j = 0; j < 8; j++)
            if (m >> j & 1) COMPRESS_LUT[m][k++] = (uint8_t)j;
        for (; k < 8; k++) COMPRESS_LUT[m][k] = 0;
    }
    SKIN_T = 50;
    for (int c = 49; c >= 0; c--)
        if ((float)c / 49.0f >= 0.35f) SKIN_T = c;
}

static scratch_t *scr_get(void) {
    if (t_scr) return t_scr;
    pthread_once(&g_once, once_init);
    scratch_t *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    pthread_setspecific(g_key, s);
    t_scr = s;
    return s;
}

static void *scr_buf(scratch_t *s, int i, size_t bytes) {
    buf_t *b = &s->b[i];
    if (b->cap < bytes) {
        size_t c = bytes + bytes / 4 + 256;
        void *p = NULL;
        if (posix_memalign(&p, 64, c)) return NULL;
        free(b->p);
        b->p = p;
        b->cap = c;
    }
    return b->p;
}

/* ------------------------------------------------------------------ SIMD helpers */

#if defined(__AVX2__)
typedef __m256i v8;
static inline v8 ld8(const void *p) { return _mm256_loadu_si256((const v8 *)p); }
static inline void st8(void *p, v8 v) { _mm256_storeu_si256((v8 *)p, v); }
static inline v8 k8(int c) { return _mm256_set1_epi8((char)c); }
static inline v8 ge8(v8 x, int c) { return _mm256_cmpeq_epi8(_mm256_max_epu8(x, k8(c)), x); }
static inline v8 le8(v8 x, int c) { return _mm256_cmpeq_epi8(_mm256_min_epu8(x, k8(c)), x); }
static inline v8 in8(v8 x, int lo, int hi) { return le8(_mm256_sub_epi8(x, k8(lo)), hi - lo); }
static inline v8 and8(v8 a, v8 b) { return _mm256_and_si256(a, b); }
static inline v8 or8(v8 a, v8 b) { return _mm256_or_si256(a, b); }
static inline unsigned mm8(v8 m) { return (unsigned)_mm256_movemask_epi8(m); }
static inline v8 absd8(v8 a, v8 b) { return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)); }
static inline int64_t hsum64(v8 s) {
    return _mm256_extract_epi64(s, 0) + _mm256_extract_epi64(s, 1) + _mm256_extract_epi64(s, 2) +
           _mm256_extract_epi64(s, 3);
}
/* lanes j >= ov set (ov = number of leading lanes already processed by the previous block) */
static const uint8_t TAILMASK[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
static inline v8 keepv(int ov) { return ld8(TAILMASK + 32 - ov); }
/* Row loop over n >= 32 elements in 32-lane blocks; the last block is moved back to end at n
 * and its first `ov` lanes (already done) must be masked out of every count. */
#define BLOCKS32(x, ov, n) \
    for (int x = 0, ov = 0; x < (n); x += 32, ov = 0) \
        if ((x + 32 > (n)) ? (ov = x + 32 - (n), x = (n) - 32, 1) : 1)
#endif

/* pairs i in [0,n) with (ma[i] & mb[i] & bit): count and sum |va[i] - vb[i]| */
static inline void pair_tex(const uint8_t *ma, const uint8_t *mb, const uint8_t *va, const uint8_t *vb,
                            int n, int bit, int64_t *tot, int64_t *cnt) {
    int64_t t = 0, c = 0;
#if defined(__AVX2__)
    if (n >= 32) {
        v8 acc = _mm256_setzero_si256(), kb = k8(bit);
        BLOCKS32(i, ov, n) {
            v8 both = and8(_mm256_cmpeq_epi8(and8(and8(ld8(ma + i), ld8(mb + i)), kb), kb), keepv(ov));
            v8 d = and8(absd8(ld8(va + i), ld8(vb + i)), both);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(d, _mm256_setzero_si256()));
            c += __builtin_popcount(mm8(both));
        }
        *tot += hsum64(acc);
        *cnt += c;
        return;
    }
#endif
    for (int i = 0; i < n; i++)
        if (ma[i] & mb[i] & bit) { t += abs(va[i] - vb[i]); c++; }
    *tot += t;
    *cnt += c;
}

/* ------------------------------------------------------------------ numpy helpers */

/* numpy FLOAT_pairwise_sum (loops_utils.h.src), PW_BLOCKSIZE 128, 8 accumulators.  The 8
 * accumulators are exactly the 8 lanes of one AVX vector. */
static float pw_sum(const float *a, ptrdiff_t n) {
    if (n < 8) {
        float res = 0.f;
        for (ptrdiff_t i = 0; i < n; i++) res += a[i];
        return res;
    } else if (n <= 128) {
        float r[8];
        ptrdiff_t i;
#if defined(__AVX2__)
        __m256 acc = _mm256_loadu_ps(a);
        for (i = 8; i < n - (n % 8); i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
        _mm256_storeu_ps(r, acc);
#else
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
#endif
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    } else {
        ptrdiff_t n2 = n / 2;
        n2 -= n2 % 8;
        return pw_sum(a, n2) + pw_sum(a + n2, n - n2);
    }
}
/* np.add.reduce over a contiguous 1-D float32 array: identity 0 + one pairwise call
 * (verified against numpy 2.5.4 for lengths up to 70000). */
static inline float np_sum_f32(const float *a, ptrdiff_t n) { return 0.f + pw_sum(a, n); }

/* value at sorted position k of a multiset given as histogram */
static int hist_kth(const int32_t *hist, int len, int64_t k) {
    int64_t c = 0;
    for (int v = 0; v < len; v++) {
        c += hist[v];
        if (c > k) return v;
    }
    return len - 1;
}

/* np.percentile(a, q*100) (method linear, numpy _lerp) followed by .astype(int); a given as a
 * histogram of n >= 1 values */
static int64_t hist_percentile_int(const int32_t *hist, int len, int64_t n, double q) {
    double v = (double)(n - 1) * q;
    if (v >= (double)(n - 1)) return hist_kth(hist, len, n - 1);
    int64_t pi = (int64_t)floor(v);
    double g = v - (double)pi;
    int64_t a = hist_kth(hist, len, pi), b = hist_kth(hist, len, pi + 1);
    double d = (double)(b - a);
    double res = (g >= 0.5) ? (double)b - d * (1.0 - g) : (double)a + d * g;
    return (int64_t)res;
}

/* np.median of an integer multiset given as histogram */
static double hist_median(const int32_t *hist, int len, int64_t n) {
    if (n % 2 == 1) return (double)hist_kth(hist, len, (n - 1) / 2);
    double a = (double)hist_kth(hist, len, n / 2 - 1), b = (double)hist_kth(hist, len, n / 2);
    return (0.0 + a + b) / 2.0;
}

/* ------------------------------------------------------------------ colour gate */

static void count_yo(const uint8_t *H, const uint8_t *S, const uint8_t *V, size_t n, int64_t *ny,
                     int64_t *no) {
    size_t i = 0;
    int64_t cy = 0, co = 0;
#if defined(__AVX2__)
    const v8 zero = _mm256_setzero_si256();
    while (i + 32 <= n) {
        v8 ay = zero, ao = zero; /* uint8 counters, <= 255 iterations per block */
        size_t end = n - i >= 32 * 255 ? i + 32 * 255 : i + ((n - i) / 32) * 32;
        for (; i < end; i += 32) {
            v8 h = ld8(H + i), s = ld8(S + i), v = ld8(V + i);
            v8 my = and8(and8(in8(h, 22, 48), ge8(s, 70)), ge8(v, 110));
            v8 mo = and8(and8(in8(h, 12, 21), ge8(s, 90)), ge8(v, 100));
            ay = _mm256_sub_epi8(ay, my);
            ao = _mm256_sub_epi8(ao, mo);
        }
        cy += hsum64(_mm256_sad_epu8(ay, zero));
        co += hsum64(_mm256_sad_epu8(ao, zero));
    }
    if (i < n && n >= 32) { /* overlapped last block */
        const int ov = (int)(i + 32 - n);
        i = n - 32;
        v8 h = ld8(H + i), s = ld8(S + i), v = ld8(V + i), kp = keepv(ov);
        cy += __builtin_popcount(mm8(and8(and8(and8(in8(h, 22, 48), ge8(s, 70)), ge8(v, 110)), kp)));
        co += __builtin_popcount(mm8(and8(and8(and8(in8(h, 12, 21), ge8(s, 90)), ge8(v, 100)), kp)));
        i = n;
    }
#endif
    for (; i < n; i++) {
        unsigned h = H[i], s = S[i], v = V[i];
        cy += (h - 22u <= 26u) & (s >= 70) & (v >= 110);
        co += (h - 12u <= 9u) & (s >= 90) & (v >= 100);
    }
    *ny = cy;
    *no = co;
}

/* ------------------------------------------------------------------ colour features (128 px) */

enum { M_Y = 1, M_O = 2, M_C = 4, M_R = 8, M_M = 16, M_D = 32 };

static inline unsigned cflags1(unsigned hh, unsigned ss, unsigned vv) {
    unsigned yl = (hh - 22u <= 26u) & (ss >= 70) & (vv >= 110);
    unsigned og = (hh - 12u <= 9u) & (ss >= 90) & (vv >= 100);
    unsigned cr = (hh - 18u <= 30u) & (ss - 25u < 45u) & (vv >= 170);
    unsigned rd = ((hh <= 12) | (hh >= 235)) & (ss >= 120) & (vv >= 120);
    unsigned mo = (vv - 25u < 95u) & (ss >= 70) & ((hh <= 16) | (hh >= 235));
    unsigned dk = vv < 90;
    return yl | og << 1 | cr << 2 | rd << 3 | mo << 4 | dk << 5;
}

__attribute__((noinline)) static int color_full(scratch_t *sc, const nl_prep *p, int64_t n_yel,
                                                double out[NL_NFEAT]) {
    const int w = p->sw, h = p->sh;
    const size_t np_ = (size_t)w * h;
    const uint8_t *H = p->sH, *S = p->sS, *V = p->sV;
    const int W4 = w + 4;
    /* mask with a 2-px wrap-around border (np.roll), plus slack for vector over-reads */
    uint8_t *P = scr_buf(sc, B_SMASK, (size_t)W4 * (h + 4) + 64);
    int32_t *hist = scr_buf(sc, B_HIST, sizeof(int32_t) * (2 * 256 + (size_t)w + 2 * (size_t)h));
    /* row buffers: 16-bit sums with zero padding */
    uint16_t *rb = scr_buf(sc, B_ROW16, sizeof(uint16_t) * 4 * ((size_t)w + 64));
    if (!P || !hist || !rb) return NL_UNSUPPORTED;
    int32_t *hH = hist, *hS = hist + 256, *hrow = hist + 512, *hcol = hrow + h, *hmrow = hcol + w;
    memset(hist, 0, sizeof(int32_t) * (512 + (size_t)w + 2 * (size_t)h));
#define MROW(y) (P + (size_t)((y) + 2) * W4 + 2)

    TOCK(4);
    int64_t n_red = 0, n_skin = 0, n_mouth = 0;
    for (int y = 0; y < h; y++) {
        const uint8_t *Hr = H + (size_t)y * w, *Sr = S + (size_t)y * w, *Vr = V + (size_t)y * w;
        uint8_t *mr = MROW(y);
        int rowc = 0;
#if defined(__AVX2__)
        if (w >= 32) {
            BLOCKS32(x, ov, w) {
                v8 hv = ld8(Hr + x), sv = ld8(Sr + x), vv = ld8(Vr + x);
                v8 yl = and8(and8(in8(hv, 22, 48), ge8(sv, 70)), ge8(vv, 110));
                v8 og = and8(and8(in8(hv, 12, 21), ge8(sv, 90)), ge8(vv, 100));
                v8 cr = and8(and8(in8(hv, 18, 48), in8(sv, 25, 69)), ge8(vv, 170));
                v8 hr = or8(le8(hv, 12), ge8(hv, 235));
                v8 rd = and8(and8(hr, ge8(sv, 120)), ge8(vv, 120));
                v8 mo = and8(and8(in8(vv, 25, 119), ge8(sv, 70)), or8(le8(hv, 16), ge8(hv, 235)));
                v8 dk = le8(vv, 89);
                v8 b = or8(or8(and8(yl, k8(M_Y)), and8(og, k8(M_O))), or8(and8(cr, k8(M_C)), and8(rd, k8(M_R))));
                b = or8(b, or8(and8(mo, k8(M_M)), and8(dk, k8(M_D))));
                st8(mr + x, b);
                const unsigned keep = ~0u << ov;
                const unsigned ym = mm8(yl) & keep;
                rowc += __builtin_popcount(ym);
                n_red += __builtin_popcount(mm8(rd) & keep);
                n_skin += __builtin_popcount(mm8(or8(yl, og)) & keep);
                const int nm = __builtin_popcount(mm8(mo) & keep);
                n_mouth += nm;
                hmrow[y] += nm;
                if (n_yel >= 20) {
                    unsigned bits = ym;
                    while (bits) {
                        const int j = x + __builtin_ctz(bits);
                        hH[Hr[j]]++;
                        hS[Sr[j]]++;
                        hcol[j]++;
                        bits &= bits - 1;
                    }
                }
            }
        } else
#endif
        {
            for (int x = 0; x < w; x++) {
                unsigned b = cflags1(Hr[x], Sr[x], Vr[x]);
                mr[x] = (uint8_t)b;
                rowc += b & 1;
                n_red += (b >> 3) & 1;
                n_skin += (b & 3) != 0;
                n_mouth += (b >> 4) & 1;
                hmrow[y] += (b >> 4) & 1;
                if (b & 1) {
                    hH[Hr[x]]++;
                    hS[Sr[x]]++;
                    hcol[x]++;
                }
            }
        }
        hrow[y] = rowc;
    }
    out[F_RED] = (double)n_red / (double)np_;
    TICK(4);
    TOCK(5);

    /* mouth: dark-red pixels whose 7x7 box (zero padded) is >= 35% skin (float32 box_blur r=3
     * compared with float32(0.35) == integer threshold SKIN_T); separable: per row that has a
     * mouth pixel, 7-row column sums, then a 7-tap sum at the mouth pixels. */
    int64_t mcount = 0;
    if (n_mouth) {
        uint8_t *col = (uint8_t *)rb; /* [3 zeros][w column 7-sums][zeros] */
        memset(col, 0, (size_t)w + 64);
        uint8_t *cs = col + 3;
#define SKIN01(r, x) ((MROW(r)[x] & (M_Y | M_O)) != 0)
        for (int r = 0; r <= 3 && r < h; r++)
            for (int x = 0; x < w; x++) cs[x] += SKIN01(r, x);
        for (int y = 0; y < h; y++) {
            if (y > 0) { /* slide the 7-row window: add row y+3, drop row y-4 */
                const uint8_t *ma = y + 3 < h ? MROW(y + 3) : NULL, *md = y - 4 >= 0 ? MROW(y - 4) : NULL;
                int x = 0;
#if defined(__AVX2__)
                const v8 k3 = k8(M_Y | M_O), k1 = k8(1);
                for (; x + 32 <= w; x += 32) {
                    v8 c = ld8(cs + x);
                    if (ma) c = _mm256_add_epi8(c, _mm256_min_epu8(and8(ld8(ma + x), k3), k1));
                    if (md) c = _mm256_sub_epi8(c, _mm256_min_epu8(and8(ld8(md + x), k3), k1));
                    st8(cs + x, c);
                }
#endif
                for (; x < w; x++) cs[x] = (uint8_t)(cs[x] + (ma ? (ma[x] & (M_Y | M_O)) != 0 : 0) -
                                                     (md ? (md[x] & (M_Y | M_O)) != 0 : 0));
            }
            if (!hmrow[y]) continue;
            const uint8_t *mr = MROW(y);
#if defined(__AVX2__)
            if (w >= 32) {
                const v8 kt = k8(SKIN_T);
                BLOCKS32(x, ov, w) {
                    v8 s7 = ld8(cs + x - 3);
                    for (int d = -2; d <= 3; d++) s7 = _mm256_add_epi8(s7, ld8(cs + x + d));
                    v8 ok = _mm256_cmpeq_epi8(_mm256_max_epu8(s7, kt), s7); /* s7 >= SKIN_T */
                    const unsigned mm = mm8(_mm256_slli_epi16(ld8(mr + x), 3)); /* mouth bit 4 -> 7 */
                    mcount += __builtin_popcount(mm & mm8(ok) & (~0u << ov));
                }
                continue;
            }
#endif
            for (int x = 0; x < w; x++) {
                if (!(mr[x] & M_M)) continue;
                const int c = cs[x - 3] + cs[x - 2] + cs[x - 1] + cs[x] + cs[x + 1] + cs[x + 2] + cs[x + 3];
                mcount += c >= SKIN_T;
            }
        }
#undef SKIN01
    }
    out[F_MOUTH] = (double)mcount / (double)(n_skin > 1 ? n_skin : 1);
    TICK(5);

    if (n_yel < 20) return NL_OK; /* yhue..belly stay 0 */
    const int64_t n = n_yel;

    TOCK(6);
    out[F_YHUE] = hist_median(hH, 256, n);
    out[F_YSAT] = hist_median(hS, 256, n);

    const double q2 = 2.0 / 100.0, q98 = 98.0 / 100.0;
    int y0 = (int)hist_percentile_int(hrow, h, n, q2), y1 = (int)hist_percentile_int(hrow, h, n, q98);
    int x0 = (int)hist_percentile_int(hcol, w, n, q2), x1 = (int)hist_percentile_int(hcol, w, n, q98);
    const int bh = y1 - y0 + 1, bw = x1 - x0 + 1;
    out[F_YASPECT] = (double)bh / (double)bw;
    TICK(6);
    TOCK(7);

    /* yellow / cream counts inside the box */
    {
        int64_t cy = 0, cc = 0;
        for (int y = y0; y <= y1; y++) {
            const uint8_t *mr = MROW(y) + x0;
#if defined(__AVX2__)
            if (bw >= 32) {
                BLOCKS32(x, ov, bw) {
                    v8 b = ld8(mr + x);
                    const unsigned keep = ~0u << ov;
                    cy += __builtin_popcount(mm8(_mm256_slli_epi16(b, 7)) & keep);
                    cc += __builtin_popcount(mm8(_mm256_slli_epi16(b, 5)) & keep);
                }
                continue;
            }
#endif
            for (int x = 0; x < bw; x++) {
                cy += mr[x] & M_Y;
                cc += (mr[x] >> 2) & 1;
            }
        }
        out[F_YFILL] = (double)cy / (double)((int64_t)bh * bw);
        out[F_BELLY] = (double)cc / (double)n;
    }

    /* outline: np.roll wrap-around neighbours at distance 1 and 2 -> fill the wrap padding */
    for (int py = 0; py < h + 4; py++) {
        const int sy = ((py - 2) % h + h) % h;
        uint8_t *dst = P + (size_t)py * W4;
        const uint8_t *src = MROW(sy);
        if (py < 2 || py >= h + 2) memcpy(dst + 2, src, (size_t)w);
        dst[0] = src[((-2) % w + w) % w];
        dst[1] = src[((-1) % w + w) % w];
        dst[w + 2] = src[w % w];
        dst[w + 3] = src[(w + 1) % w];
    }
    {
        int64_t border = 0, hits = 0;
        const int offs[4] = {1, -1, W4, -W4};
        for (int k = 0; k < 4; k++) {
            const int o = offs[k];
            for (int y = 0; y < h; y++) {
                const uint8_t *c = MROW(y), *n1 = c + o, *n2 = c + 2 * o;
#if defined(__AVX2__)
                if (w >= 32) {
                    BLOCKS32(x, ov, w) {
                        v8 a = ld8(c + x), b1 = ld8(n1 + x), b2 = ld8(n2 + x);
                        v8 e = _mm256_slli_epi16(_mm256_andnot_si256(b1, a), 7); /* Y & ~Y(nb) -> bit 7 */
                        v8 d = _mm256_slli_epi16(or8(b1, b2), 2);                   /* dark bit 5 -> bit 7 */
                        const unsigned me = mm8(e) & (~0u << ov), md = mm8(d);
                        border += __builtin_popcount(me);
                        hits += __builtin_popcount(me & md);
                    }
                    continue;
                }
#endif
                for (int x = 0; x < w; x++) {
                    unsigned e = c[x] & ~n1[x] & M_Y;
                    border += e;
                    hits += e & (((n1[x] | n2[x]) & M_D) != 0);
                }
            }
        }
        out[F_OUTLINE] = border ? (double)hits / (double)border : 0.0;
    }
    TICK(7);
    TOCK(8);

    /* hard-edge density inside the box */
    if (bh > 1 && bw > 1) {
        int64_t edges = 0;
        const int nx = bw - 1;
        for (int y = y0; y < y1; y++) {
            const uint8_t *v0 = V + (size_t)y * w + x0, *v1 = v0 + w;
#if defined(__AVX2__)
            if (nx >= 32) {
                BLOCKS32(x, ov, nx) {
                    v8 a = ld8(v0 + x);
                    v8 d = _mm256_max_epu8(absd8(ld8(v0 + x + 1), a), absd8(ld8(v1 + x), a));
                    const unsigned le = mm8(_mm256_cmpeq_epi8(_mm256_subs_epu8(d, k8(40)), _mm256_setzero_si256()));
                    edges += __builtin_popcount(~le & (~0u << ov));
                }
                continue;
            }
#endif
            for (int x = 0; x < nx; x++) {
                int a = v0[x];
                edges += (abs(v0[x + 1] - a) > 40) | (abs(v1[x] - a) > 40);
            }
        }
        out[F_EDGES] = (double)edges / (double)((int64_t)(bh - 1) * (bw - 1));
    } else {
        out[F_EDGES] = 0.0;
    }

    /* ytex: exact integer sums (the float32 sums in the reference are exact, < 2^24) */
    {
        int64_t tot = 0, cnt = 0;
        for (int y = 0; y < h; y++) {
            const uint8_t *mr = MROW(y), *vr = V + (size_t)y * w;
            pair_tex(mr, mr + 1, vr, vr + 1, w - 1, M_Y, &tot, &cnt);
            if (y + 1 < h) pair_tex(mr, mr + W4, vr, vr + w, w, M_Y, &tot, &cnt);
        }
        out[F_YTEX] = (double)tot / (double)(cnt > 1 ? cnt : 1);
    }
    TICK(8);
    TOCK(9);

    /* shading: blur = box_blur(V, 4) (float32: exact integer 9x9 sum / 81.f); std / mean over the
     * yellow pixels with numpy pairwise sums.  Separable 16-bit box sums (max 81*255 < 2^16). */
    {
        float *fb = scr_buf(sc, B_F32A, sizeof(float) * ((size_t)n + 16));
        float *fr = scr_buf(sc, B_F32B, sizeof(float) * ((size_t)n + 16));
        if (!fb || !fr) return NL_UNSUPPORTED;
        uint16_t *cs = rb + 4;                    /* column 9-sums, >= 4 zeros each side */
        uint16_t *bs = rb + 2 * ((size_t)w + 64); /* 9x9 sums of the current row */
        memset(rb, 0, sizeof(uint16_t) * ((size_t)w + 64));
        for (int x = 0; x < w; x++) {
            int s = 0;
            for (int r = 0; r <= 4 && r < h; r++) s += V[(size_t)r * w + x];
            cs[x] = (uint16_t)s;
        }
        int64_t k = 0;
        for (int y = 0; y < h; y++) {
            if (y > 0) { /* slide the 9-row window: add row y+4, drop row y-5 */
                const uint8_t *va = y + 4 < h ? V + (size_t)(y + 4) * w : NULL;
                const uint8_t *vd = y - 5 >= 0 ? V + (size_t)(y - 5) * w : NULL;
                int x = 0;
#if defined(__AVX2__)
                for (; x + 16 <= w; x += 16) {
                    __m256i c = ld8(cs + x);
                    if (va) c = _mm256_add_epi16(c, _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(va + x))));
                    if (vd) c = _mm256_sub_epi16(c, _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(vd + x))));
                    st8(cs + x, c);
                }
#endif
                for (; x < w; x++) cs[x] = (uint16_t)(cs[x] + (va ? va[x] : 0) - (vd ? vd[x] : 0));
            }
            const uint8_t *mr = MROW(y), *vr = V + (size_t)y * w;
            int x = 0;
#if defined(__AVX2__)
            for (; x + 16 <= w; x += 16) {
                __m256i s = ld8(cs + x - 4);
                for (int d = -3; d <= 4; d++) s = _mm256_add_epi16(s, ld8(cs + x + d));
                st8(bs + x, s);
            }
#endif
            for (; x < w; x++) {
                int s = 0;
                for (int d = -4; d <= 4; d++) s += cs[x + d];
                bs[x] = (uint16_t)s;
            }
            x = 0;
#if defined(__AVX2__)
            const __m256 k81 = _mm256_set1_ps(81.0f);
            const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            for (; x + 8 <= w; x += 8) {
                const unsigned ym =
                    (unsigned)_mm_movemask_epi8(_mm_slli_epi16(_mm_loadl_epi64((const __m128i *)(mr + x)), 7)) & 0xFF;
                if (!ym) continue;
                __m256 s = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(bs + x))));
                __m256 b = _mm256_div_ps(s, k81);
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(vr + x))));
                __m256 r = _mm256_and_ps(_mm256_sub_ps(vf, b), absm);
                const __m256i perm = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)COMPRESS_LUT[ym]));
                _mm256_storeu_ps(fb + k, _mm256_permutevar8x32_ps(b, perm));
                _mm256_storeu_ps(fr + k, _mm256_permutevar8x32_ps(r, perm));
                k += __builtin_popcount(ym);
            }
#endif
            for (; x < w; x++) {
                if (!(mr[x] & M_Y)) continue;
                const float b = (float)bs[x] / 81.0f;
                fb[k] = b;
                fr[k] = fabsf((float)vr[x] - b);
                k++;
            }
        }
        const float mean = (float)((double)np_sum_f32(fb, n) / (double)n);
        int64_t i = 0;
#if defined(__AVX2__)
        const __m256 mv = _mm256_set1_ps(mean);
        for (; i + 8 <= n; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(fb + i), mv);
            _mm256_storeu_ps(fb + i, _mm256_mul_ps(d, d));
        }
#endif
        for (; i < n; i++) {
            const float d = fb[i] - mean;
            fb[i] = d * d;
        }
        const float var = (float)((double)np_sum_f32(fb, n) / (double)n);
        const float low = sqrtf(var);
        const float resid = (float)((double)np_sum_f32(fr, n) / (double)n);
        out[F_SHADING] = (double)low / ((double)resid + 1.0);
    }
    TICK(9);
#undef MROW
    return NL_OK;
}

/* ------------------------------------------------------------------ eye features (256 px) */

typedef struct {
    float v;
    int32_t idx;
} peak_t;

static int peak_cmp(const void *a, const void *b) {
    const peak_t *p = a, *q = b;
    if (p->v > q->v) return -1;
    if (p->v < q->v) return 1;
    return (p->idx > q->idx) - (p->idx < q->idx);
}

#if !defined(__AVX2__)
static const uint64_t PACK[16] = {
#define PK(b) (((uint64_t)((b) & 1)) | ((uint64_t)(((b) >> 1) & 1) << 16) | \
               ((uint64_t)(((b) >> 2) & 1) << 32) | ((uint64_t)(((b) >> 3) & 1) << 48))
    PK(0), PK(1), PK(2), PK(3), PK(4), PK(5), PK(6), PK(7),
    PK(8), PK(9), PK(10), PK(11), PK(12), PK(13), PK(14), PK(15),
#undef PK
};
#endif

enum { E_D = 1, E_G = 2, E_Y = 4, E_W = 8 };

static inline unsigned eflags1(unsigned hh, unsigned ss, unsigned vv) {
    unsigned dk = vv < 50;
    unsigned gr = (hh - 38u <= 72u) & (ss >= 45) & (vv - 40u <= 195u);
    unsigned yl = (hh - 22u <= 26u) & (ss >= 60) & (vv >= 100);
    unsigned wh = (vv >= 200) & (ss < 60);
    return dk | gr << 1 | yl << 2 | wh << 3;
}

/* Distinct box radii used by RINGS (r, r2, r3), ascending. */
#define NU 19
static const int UR[NU] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 17, 19, 25, 32, 34, 43, 57};
/* The packed integral image (uint64 = 4 x 16-bit channel counts dark/green/yellow/white, sums
 * mod 2^64 exactly as in the reference) has padded columns so that unclamped column indices
 * reproduce np.clip: left pad = column 0 (zeros), right pad = column w.  Rows are clamped
 * through per-row offsets. */
#define PADL 57
#define PADR 57

/* box sum from the 4 integral-image corners.  AVX2 build: the integral is kept per 16-bit lane
 * (mod 2^16, no carries between channels), so the box is computed per lane as well; the true
 * per-channel counts are < 2^16, hence exact.  Scalar build: uint64 with carries, as numpy. */
static inline uint64_t box4(uint64_t b1, uint64_t a1, uint64_t b0, uint64_t a0) {
#if defined(__AVX2__)
    const __m128i r = _mm_add_epi16(
        _mm_sub_epi16(_mm_sub_epi16(_mm_cvtsi64_si128((long long)b1), _mm_cvtsi64_si128((long long)a1)),
                      _mm_cvtsi64_si128((long long)b0)),
        _mm_cvtsi64_si128((long long)a0));
    return (uint64_t)_mm_cvtsi128_si64(r);
#else
    return b1 - a1 - b0 + a0;
#endif
}

static inline void boxes_at(const uint64_t *ii, const ptrdiff_t *offA, const ptrdiff_t *offB, int x,
                            uint64_t *bx) {
#define B1(u, r) \
    bx[u] = box4(ii[offB[u] + x + (r) + 1], ii[offA[u] + x + (r) + 1], ii[offB[u] + x - (r)], ii[offA[u] + x - (r)]);
    B1(0, 1) B1(1, 2) B1(2, 3) B1(3, 4) B1(4, 5) B1(5, 6) B1(6, 7) B1(7, 8) B1(8, 10) B1(9, 11)
    B1(10, 14) B1(11, 15) B1(12, 17) B1(13, 19) B1(14, 25) B1(15, 32) B1(16, 34) B1(17, 43) B1(18, 57)
#undef B1
}

static inline void row_offsets(int y, int h, ptrdiff_t SW, ptrdiff_t *offA, ptrdiff_t *offB) {
    for (int u = 0; u < NU; u++) {
        int ya = y - UR[u], yb = y + UR[u] + 1;
        if (ya < 0) ya = 0;
        if (yb > h) yb = h;
        offA[u] = ya * SW + PADL;
        offB[u] = yb * SW + PADL;
    }
}

/* ring k: (k, r, ring_n, out_n, a1, index of r, r2, r3 in UR) */
#define RING_LIST(X)                     \
    X(0, 1, 40, 176, 9, 0, 2, 6)         \
    X(1, 2, 96, 320, 25, 1, 4, 8)        \
    X(2, 3, 240, 552, 49, 2, 7, 10)      \
    X(3, 4, 360, 784, 81, 3, 8, 12)      \
    X(4, 6, 672, 1760, 169, 5, 10, 14)   \
    X(5, 8, 1232, 2704, 289, 7, 13, 15)  \
    X(6, 11, 2072, 4968, 529, 9, 14, 17) \
    X(7, 15, 3800, 8464, 961, 11, 16, 18)

#define M16 0xFFFFull

/* green-iris ratio with the radius of the last strict improvement (g_rad) */
#define RING_G(k, R, RN, ON, A1, IA, IB, IC)                                                       \
    {                                                                                              \
        const uint64_t C = bx[IA], Mb = bx[IB], O = bx[IC];                                        \
        const int cd = (int)(C & M16), cg = (int)(C >> 16 & M16);                                  \
        const int md = (int)(Mb & M16), mg = (int)(Mb >> 16 & M16), my = (int)(Mb >> 32 & M16);    \
        const int og = (int)(O >> 16 & M16), oy = (int)(O >> 32 & M16);                            \
        const int pupil = (20 * cd >= 11 * (A1)) & (2 * (md - cd) <= (RN));                        \
        const int skin = (10 * (oy - my) >= 3 * (ON)) & (4 * (og - mg) <= (ON));                   \
        if (pupil & skin) {                                                                        \
            const float g = (float)(mg - cg) / (float)(RN);                                        \
            if (g > gb) { gb = g; grad = (R); }                                                    \
        }                                                                                          \
    }

#if !defined(__AVX2__)
#define RING_FULL(k, R, RN, ON, A1, IA, IB, IC)                                                    \
    {                                                                                              \
        const uint64_t C = bx[IA], Mb = bx[IB], O = bx[IC];                                        \
        const int cd = (int)(C & M16), cg = (int)(C >> 16 & M16), cy = (int)(C >> 32 & M16),      \
                  cw = (int)(C >> 48);                                                             \
        const int md = (int)(Mb & M16), mg = (int)(Mb >> 16 & M16), my = (int)(Mb >> 32 & M16),    \
                  mw = (int)(Mb >> 48);                                                            \
        const int og = (int)(O >> 16 & M16), oy = (int)(O >> 32 & M16);                            \
        const int ryn = my - cy;                                                                   \
        const int pupil = (20 * cd >= 11 * (A1)) & (2 * (md - cd) <= (RN));                        \
        const int skin = (10 * (oy - my) >= 3 * (ON)) & (4 * (og - mg) <= (ON));                   \
        const int glint = (20 * cd >= 7 * (A1)) & (50 * cw >= (A1)) & (10 * cw <= 3 * (A1)) &       \
                          (5 * ryn >= 2 * (RN)) & skin;                                            \
        const int tye = pupil ? ryn : -1, tsc = pupil ? mw - cw : -1, tgl = glint ? ryn : -1;      \
        maxye[k] = maxye[k] > tye ? maxye[k] : tye;                                                \
        maxsc[k] = maxsc[k] > tsc ? maxsc[k] : tsc;                                                \
        maxgl[k] = maxgl[k] > tgl ? maxgl[k] : tgl;                                                \
        if (pupil & skin) {                                                                        \
            const float g = (float)(mg - cg) / (float)(RN);                                        \
            if (g > gb) gb = g;                                                                    \
        }                                                                                          \
    }
#endif

#if defined(__AVX2__)
/* Ring logic for 4 rings at once.  A box vector holds 4 rings x 4 channels as 16-bit lanes
 * (lane 4q+c: ring 4G+q, channel c = dark/green/yellow/white; all counts <= 115^2 < 2^15).
 * Every integer condition of the reference (e.g. 20*cd >= 11*a1) is rewritten as the exact
 * threshold on a count (cd >= ceil(11*a1/20)); lanes a condition does not look at get thresholds
 * that always hold, so a condition is true for a ring iff its 64-bit lane group is all-ones. */
typedef struct {
    __m256i kB, kA, kC, kD, kE, kF, kG;
} ringk_t;

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

static void ring_consts(ringk_t *K) {
    for (int G = 0; G < 2; G++) {
        int16_t B[16], A[16], Cc[16], D[16], E[16], F[16], Gg[16];
        for (int q = 0; q < 4; q++) {
            const ring_t R = RINGS[4 * G + q];
            for (int c = 0; c < 4; c++) {
                const int L = 4 * q + c;
                B[L] = A[L] = E[L] = F[L] = -1;
                Cc[L] = D[L] = Gg[L] = 32767;
            }
            B[4 * q + 0] = (int16_t)(cdiv(11 * R.a1, 20) - 1); /* pupil: cd >= ceil(11 a1 / 20) */
            D[4 * q + 0] = (int16_t)(R.rn / 2 + 1);            /* pupil: md - cd <= rn / 2 */
            F[4 * q + 2] = (int16_t)(cdiv(3 * R.on, 10) - 1);  /* skin: oy - my >= ceil(3 on / 10) */
            Gg[4 * q + 1] = (int16_t)(R.on / 4 + 1);           /* skin: og - mg <= on / 4 */
            A[4 * q + 0] = (int16_t)(cdiv(7 * R.a1, 20) - 1);  /* glint: cd >= ceil(7 a1 / 20) */
            A[4 * q + 3] = (int16_t)(cdiv(R.a1, 50) - 1);      /* glint: cw >= ceil(a1 / 50) */
            Cc[4 * q + 3] = (int16_t)(3 * R.a1 / 10 + 1);      /* glint: cw <= 3 a1 / 10 */
            E[4 * q + 2] = (int16_t)(cdiv(2 * R.rn, 5) - 1);   /* glint: my - cy >= ceil(2 rn / 5) */
        }
        K[G].kB = ld8(B); K[G].kA = ld8(A); K[G].kC = ld8(Cc); K[G].kD = ld8(D);
        K[G].kE = ld8(E); K[G].kF = ld8(F); K[G].kG = ld8(Gg);
    }
}
#endif

__attribute__((noinline)) static int eye_full(scratch_t *sc, const nl_prep *p, double out[NL_NFEAT]) {
    const int w = p->bw, h = p->bh;
    const ptrdiff_t SW = (ptrdiff_t)w + 1 + PADL + PADR;
    const size_t np_ = (size_t)w * h;
    const uint8_t *H = p->bH, *S = p->bS, *V = p->bV;
    uint8_t *m = scr_buf(sc, B_BMASK, np_ + 64);
    uint64_t *ii = scr_buf(sc, B_II64, sizeof(uint64_t) * (size_t)SW * (h + 1));
    int32_t *cand = scr_buf(sc, B_CAND, sizeof(int32_t) * (np_ / 2 + 64));
    if (!m || !ii || !cand) return NL_UNSUPPORTED;

    /* masks + packed integral image + texture + all checkerboard candidates, one pass */
    TOCK(2);
    int64_t tex_tot = 0, tex_cnt = 0, nall = 0;
    memset(ii, 0, sizeof(uint64_t) * SW);
    for (int y = 0; y < h; y++) {
        const uint8_t *Hr = H + (size_t)y * w, *Sr = S + (size_t)y * w, *Vr = V + (size_t)y * w;
        uint8_t *mr = m + (size_t)y * w;
#if defined(__AVX2__)
        if (w >= 32) {
            const v8 par = _mm256_set1_epi16((y & 1) ? (short)0xFF00 : (short)0x00FF); /* (x+y) even, x even */
            BLOCKS32(x, ov, w) {
                v8 hv = ld8(Hr + x), sv = ld8(Sr + x), vv = ld8(Vr + x);
                v8 dk = le8(vv, 49);
                v8 gr = and8(and8(in8(hv, 38, 110), ge8(sv, 45)), in8(vv, 40, 235));
                v8 yl = and8(and8(in8(hv, 22, 48), ge8(sv, 60)), ge8(vv, 100));
                v8 wh = and8(ge8(vv, 200), le8(sv, 59));
                v8 b = or8(or8(and8(dk, k8(E_D)), and8(gr, k8(E_G))), or8(and8(yl, k8(E_Y)), and8(wh, k8(E_W))));
                st8(mr + x, b);
                /* the overlapped last block may start at an odd x: flip the parity pattern */
                unsigned cm = mm8(and8(dk, (x & 1) ? _mm256_xor_si256(par, k8(-1)) : par)) & (~0u << ov);
                while (cm) {
                    cand[nall++] = y << 16 | (x + __builtin_ctz(cm));
                    cm &= cm - 1;
                }
            }
        } else
#endif
        {
            for (int x = 0; x < w; x++) {
                unsigned b = eflags1(Hr[x], Sr[x], Vr[x]);
                mr[x] = (uint8_t)b;
                if ((b & E_D) && !((x + y) & 1)) cand[nall++] = y << 16 | x;
            }
        }
        uint64_t *row = ii + (size_t)(y + 1) * SW + PADL;
        const uint64_t *prev = row - SW;
        for (int x = -PADL; x <= 0; x++) row[x] = 0;
#if defined(__AVX2__)
        {
            /* 4 pixels per vector, 16-bit lane per channel; blocks past w only write the right pad
             * (overwritten below) and read the slack after the mask / the pad of prev. */
            const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
            const __m256i kbit = _mm256_set1_epi64x(0x0008000400020001ll), k1 = _mm256_set1_epi16(1);
            __m256i carry = _mm256_setzero_si256();
            for (int x = 0; x < w; x += 4) {
                int32_t m4;
                memcpy(&m4, mr + x, 4);
                __m256i v = _mm256_min_epu16(and8(_mm256_shuffle_epi8(_mm256_set1_epi32(m4), shuf), kbit), k1);
                v = _mm256_add_epi16(v, _mm256_slli_si256(v, 8));
                v = _mm256_add_epi16(v, _mm256_blend_epi32(_mm256_permute4x64_epi64(v, 0x50), _mm256_setzero_si256(), 0x0F));
                const __m256i o = _mm256_add_epi16(v, carry);
                carry = _mm256_add_epi16(carry, _mm256_permute4x64_epi64(v, 0xFF));
                st8(row + x + 1, _mm256_add_epi16(o, ld8(prev + x + 1)));
            }
        }
#else
        {
            uint64_t acc = 0;
            for (int x = 0; x < w; x++) {
                acc += PACK[mr[x]];
                row[x + 1] = prev[x + 1] + acc;
            }
        }
#endif
        for (int x = w + 1; x <= w + PADR; x++) row[x] = row[w];
        pair_tex(mr, mr + 1, Vr, Vr + 1, w - 1, E_Y, &tex_tot, &tex_cnt);
        if (y > 0) pair_tex(mr - w, mr, Vr - w, Vr, w, E_Y, &tex_tot, &tex_cnt);
    }
    TICK(2);
    out[F_TEX256] = (double)tex_tot / (double)(tex_cnt > 1 ? tex_cnt : 1);
    out[F_GREEN_BEST] = out[F_GREEN_PAIR] = out[F_YELLOW_EYE] = out[F_GLINT] = out[F_SCLERA] = 0.0;
    if (nall == 0) return NL_OK;

    /* features.candidates: more than MAX_CANDIDATES -> every step-th in raster order */
    int64_t nc = nall;
    if (nall > MAX_CANDIDATES) {
        const int64_t step = nall / MAX_CANDIDATES + 1;
        nc = 0;
        for (int64_t k = 0; k < nall; k += step) cand[nc++] = cand[k];
    }
    peak_t *peaks = scr_buf(sc, B_PEAK, sizeof(peak_t) * (size_t)nc);
    if (!peaks) return NL_UNSUPPORTED;
    STAT(3, nc);
    STAT(4, 1);

    int maxye[NRING], maxsc[NRING], maxgl[NRING];
    float gbest_all = 0.f;
    int64_t npk = 0;
    ptrdiff_t offA[NU], offB[NU];
    uint64_t bx[NU];
    TOCK(3);
#if defined(__AVX2__)
    {
        ringk_t K[2];
        ring_consts(K);
        const __m256i ones = _mm256_set1_epi64x(-1);
        __m256i RYS[2] = {ones, ones}, RGL[2] = {ones, ones}; /* running maxima, -1 = none */
        __m256i IDX[2][3][4];                                 /* gather indices of the current row */
        const long long *base = (const long long *)ii;
        int cur_y = -1;
        for (int64_t i = 0; i < nc; i++) {
            const int y = cand[i] >> 16, x = cand[i] & 0xFFFF;
            if (y != cur_y) {
                cur_y = y;
                for (int G = 0; G < 2; G++)
                    for (int T = 0; T < 3; T++) {
                        int64_t b1[4], a1[4], b0[4], a0[4];
                        for (int q = 0; q < 4; q++) {
                            const ring_t R = RINGS[4 * G + q];
                            const int rr = T == 0 ? R.r : T == 1 ? R.r2 : R.r3;
                            int ya = y - rr, yb = y + rr + 1;
                            if (ya < 0) ya = 0;
                            if (yb > h) yb = h;
                            b1[q] = yb * SW + PADL + rr + 1;
                            a1[q] = ya * SW + PADL + rr + 1;
                            b0[q] = yb * SW + PADL - rr;
                            a0[q] = ya * SW + PADL - rr;
                        }
                        IDX[G][T][0] = ld8(b1); IDX[G][T][1] = ld8(a1);
                        IDX[G][T][2] = ld8(b0); IDX[G][T][3] = ld8(a0);
                    }
            }
            const __m256i xv = _mm256_set1_epi64x(x);
            float gb = 0.f;
            for (int G = 0; G < 2; G++) {
#define GBOX(T)                                                                                   \
    _mm256_add_epi16(                                                                             \
        _mm256_sub_epi16(                                                                         \
            _mm256_sub_epi16(_mm256_i64gather_epi64(base, _mm256_add_epi64(IDX[G][T][0], xv), 8), \
                             _mm256_i64gather_epi64(base, _mm256_add_epi64(IDX[G][T][1], xv), 8)), \
            _mm256_i64gather_epi64(base, _mm256_add_epi64(IDX[G][T][2], xv), 8)),                 \
        _mm256_i64gather_epi64(base, _mm256_add_epi64(IDX[G][T][3], xv), 8))
                const __m256i C = GBOX(0), Mv = GBOX(1);
                const __m256i MC = _mm256_sub_epi16(Mv, C);
                const __m256i pupil = and8(_mm256_cmpgt_epi16(C, K[G].kB), _mm256_cmpgt_epi16(K[G].kD, MC));
                const __m256i gpre = and8(and8(_mm256_cmpgt_epi16(C, K[G].kA), _mm256_cmpgt_epi16(K[G].kC, C)),
                                          _mm256_cmpgt_epi16(MC, K[G].kE));
                const __m256i pq = _mm256_cmpeq_epi64(pupil, ones), gpq = _mm256_cmpeq_epi64(gpre, ones);
                RYS[G] = _mm256_max_epi16(RYS[G], _mm256_blendv_epi8(ones, MC, pq));
                if (_mm256_testz_si256(or8(pq, gpq), or8(pq, gpq))) continue; /* skin irrelevant */
                const __m256i OM = _mm256_sub_epi16(GBOX(2), Mv);
#undef GBOX
                const __m256i skin = and8(_mm256_cmpgt_epi16(OM, K[G].kF), _mm256_cmpgt_epi16(K[G].kG, OM));
                const __m256i sq = _mm256_cmpeq_epi64(skin, ones);
                const __m256i gq = and8(gpq, sq);
                RGL[G] = _mm256_max_epi16(RGL[G], _mm256_blendv_epi8(ones, MC, gq));
                const unsigned ps = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(and8(pq, sq)));
                if (__builtin_expect(ps != 0, 0)) {
                    int16_t mc[16];
                    st8(mc, MC);
                    for (int q = 0; q < 4; q++)
                        if (ps >> q & 1) {
                            const float g = (float)mc[4 * q + 1] / (float)RINGS[4 * G + q].rn;
                            if (g > gb) gb = g;
                        }
                }
            }
            gbest_all = gb > gbest_all ? gb : gbest_all;
            if ((double)gb >= 0.2) {
                peaks[npk].v = gb;
                peaks[npk].idx = (int32_t)i;
                npk++;
            }
        }
        for (int G = 0; G < 2; G++) {
            int16_t a[16], b[16];
            st8(a, RYS[G]);
            st8(b, RGL[G]);
            for (int q = 0; q < 4; q++) {
                maxye[4 * G + q] = a[4 * q + 2];
                maxsc[4 * G + q] = a[4 * q + 3];
                maxgl[4 * G + q] = b[4 * q + 2];
            }
        }
    }
#else
    for (int k = 0; k < NRING; k++) maxye[k] = maxsc[k] = maxgl[k] = -1;
    {
        int cur_y = -1;
        for (int64_t i = 0; i < nc; i++) {
            const int y = cand[i] >> 16, x = cand[i] & 0xFFFF;
            if (y != cur_y) {
                row_offsets(y, h, SW, offA, offB);
                cur_y = y;
            }
            boxes_at(ii, offA, offB, x, bx);
            float gb = 0.f;
            RING_LIST(RING_FULL)
            gbest_all = gb > gbest_all ? gb : gbest_all;
            if ((double)gb >= 0.2) {
                peaks[npk].v = gb;
                peaks[npk].idx = (int32_t)i;
                npk++;
            }
        }
    }
#endif
    TICK(3);

    out[F_GREEN_BEST] = (double)gbest_all;
    double ye = 0.0, sc_ = 0.0, gl = 0.0;
    for (int k = 0; k < NRING; k++) {
        const float rn = (float)RINGS[k].rn;
        if (maxye[k] >= 0) {
            double a = (double)((float)maxye[k] / rn), b = (double)((float)maxsc[k] / rn);
            if (a > ye) ye = a;
            if (b > sc_) sc_ = b;
        }
        if (maxgl[k] >= 0) {
            double a = (double)((float)maxgl[k] / rn);
            if (a > gl) gl = a;
        }
    }
    out[F_YELLOW_EYE] = ye;
    out[F_SCLERA] = sc_;
    out[F_GLINT] = gl;

    /* green_pair: stable-argsort NMS over g_best >= 0.2, up to 4 peaks */
    if (npk >= 2) {
        qsort(peaks, (size_t)npk, sizeof(peak_t), peak_cmp);
        int pr[4], py[4], px[4], np4 = 0;
        for (int64_t t = 0; t < npk && np4 < 4; t++) {
            const int i = peaks[t].idx;
            const int y = cand[i] >> 16, x = cand[i] & 0xFFFF;
            int grad = 0;
            float gb = 0.f;
            row_offsets(y, h, SW, offA, offB);
            boxes_at(ii, offA, offB, x, bx);
            RING_LIST(RING_G)
            int ok = 1;
            for (int q = 0; q < np4; q++)
                if (!(abs(y - py[q]) > 3 * pr[q] + 2 || abs(x - px[q]) > 3 * pr[q] + 2)) { ok = 0; break; }
            if (ok) { pr[np4] = grad; py[np4] = y; px[np4] = x; np4++; }
        }
        for (int a = 0; a < np4; a++)
            for (int b = a + 1; b < np4; b++) {
                double q = (double)pr[a] / (double)pr[b];
                int mx = pr[a] > pr[b] ? pr[a] : pr[b];
                if (0.5 <= q && q <= 2 && abs(py[a] - py[b]) <= 3 * mx + 2) {
                    out[F_GREEN_PAIR] = 1.0;
                    return NL_OK;
                }
            }
    }
    return NL_OK;
}

/* ------------------------------------------------------------------ public API */

int nl_features(const nl_prep *p, double min_yellow, double out[NL_NFEAT], int *skipped) {
    if (!p || p->sw <= 0 || p->sh <= 0 || p->bw <= 0 || p->bh <= 0 || !p->sH || !p->sS || !p->sV)
        return NL_ERROR;
    if (p->bw > 0x7FFF || p->bh > 0x7FFF) return NL_UNSUPPORTED;
    for (int i = 0; i < NL_NFEAT; i++) out[i] = 0.0;
    const size_t ns = (size_t)p->sw * p->sh;
    int64_t ny, no;
    count_yo(p->sH, p->sS, p->sV, ns, &ny, &no);
    out[F_YFRAC] = (double)ny / (double)ns;
    out[F_OFRAC] = (double)no / (double)ns;
    out[F_SRC] = (double)p->orig;
    if (min_yellow != 0.0 && out[F_YFRAC] + out[F_OFRAC] < min_yellow) {
        if (skipped) *skipped = 1;
        return NL_OK;
    }
    if (skipped) *skipped = 0;
    if (!p->bH || !p->bS || !p->bV) return NL_ERROR;
    scratch_t *sc = scr_get();
    if (!sc) return NL_UNSUPPORTED;
    TOCK(0);
    int rc = color_full(sc, p, ny, out);
    TICK(0);
    if (rc != NL_OK) return rc;
    TOCK(1);
    rc = eye_full(sc, p, out);
    TICK(1);
    return rc;
}

double nl_score(const double f[NL_NFEAT], const double p[NL_NPARAM]) {
    if (f[F_YFRAC] + p[P_GATE_O] * f[F_OFRAC] < p[P_GATE]) {
        if (!(f[F_YFRAC] + f[F_OFRAC] >= p[P_GATE_LO] && f[F_GREEN_PAIR] != 0.0 &&
              f[F_GREEN_BEST] >= p[P_EYE_GB]))
            return -10.0;
    }
    /* Python: max(0.0, z) returns z only if z > 0.0; min(1.0, m) returns m only if m < 1.0 */
    double z = (f[F_GREEN_BEST] - p[P_G0]) / (p[P_G1] - p[P_G0]);
    double mz = z > 0.0 ? z : 0.0;
    mz = mz < 1.0 ? mz : 1.0;
    double s = p[P_W_G] * mz;
    s += p[P_W_PAIR] * f[F_GREEN_PAIR];
    s += p[P_W_GLINT] * (double)(f[F_GLINT] >= p[P_GL]);
    s += p[P_W_YE] * (double)(f[F_YELLOW_EYE] >= p[P_YE]);
    s -= p[P_W_SC] * (double)(f[F_SCLERA] >= p[P_SC]);
    s += p[P_W_HUE] * (double)(f[F_YHUE] <= p[P_HUE]);
    s += p[P_W_ASP] * (double)(f[F_YASPECT] >= p[P_ASP]);
    s -= p[P_W_RED] * (double)(f[F_RED] >= p[P_RED]);
    s -= p[P_W_OUT] * (double)(f[F_OUTLINE] >= p[P_OUT]);
    s -= p[P_W_TEX] * (double)(f[F_YTEX] >= p[P_TEX]);
    s -= p[P_W_TEX2] * (double)(f[F_TEX256] >= p[P_TEX2]);
    s += p[P_W_E] * (double)(f[F_EDGES] <= p[P_E_LO]);
    s -= p[P_W_E] * (double)(f[F_EDGES] >= p[P_E_HI]);
    s += p[P_W_BELLY] * (double)(f[F_BELLY] >= p[P_BELLY]);
    s += p[P_W_SHADE] * (double)(f[F_SHADING] >= p[P_SHADE]);
    s += p[P_W_ORANGE] * (double)(f[F_OFRAC] >= p[P_ORANGE]);
    s += p[P_W_MOUTH] * (double)(f[F_MOUTH] >= p[P_MOUTH]);
    s -= p[P_W_SMALL] * (double)(f[F_SRC] < p[P_SMALL]);
    /* Python: w_gp * (green_pair and green_best >= gp) —— 值等于 w_gp * 0 或 w_gp * 1 */
    s += p[P_W_GP] * (double)(f[F_GREEN_PAIR] != 0.0 && f[F_GREEN_BEST] >= p[P_GP]);
    return s;
}
