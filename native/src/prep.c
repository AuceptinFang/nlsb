/* nailong-hpc module B: reduce + bilinear resize + alpha composite + RGB->HSV.
 *
 * Bit-exact re-implementation of the Pillow 13.0.0 operations used by
 * nailong.features._load / extract (derivations in the comments below):
 *   Image.reduce(f)            -> Reduce.c (RGBA goes through RGBa premultiplied)
 *   Image.resize(BILINEAR)     -> Resample.c fixed point (RGBA goes through RGBa)
 *   Image.alpha_composite(white, im).convert("RGB")  -> AlphaComposite.c
 *   convert("HSV")             -> Convert.c rgb2hsv_row
 *
 * Internal images are tightly packed, pixel stride ps = 3 (caller RGB) or 4
 * (RGBX / RGBA / RGBa).  Thread safety: the only process-wide state is three
 * read-only 64 KB LUTs built once under pthread_once; scratch memory (bump
 * arena) and a small coefficient cache are per thread and are freed by a
 * pthread key destructor at thread exit (or explicitly by nlp_release_scratch).
 *
 * Needs AVX2 + FMA; uses AVX-VNNI (vpdpwssd) when compiled for a CPU that has it.
 * Build: gcc -O3 -march=native -ffp-contract=off -fPIC -c prep.c   (+ -lm -lpthread)
 */
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../include/nl.h"
#include "prep_internal.h"

#define PRECISION_BITS (32 - 8 - 2)

/* ------------------------------------------------------------------ */
/* per-thread scratch arena                                             */
/* ------------------------------------------------------------------ */

#define ARENA_SLACK 128            /* every block gets this much over-read slack */
#define ARENA_MAX_RETAIN (64u << 20)
#define ARENA_MAX_EXTRA 128
#define ARENA_INIT (4u << 20)

typedef struct {
    uint8_t *base;
    size_t cap, used;
    void *extra[ARENA_MAX_EXTRA];
    size_t extra_bytes;
    int nextra;
} arena_t;

static __thread arena_t g_arena;
static pthread_key_t g_tls_key; /* destructor frees the thread's scratch at thread exit */
static void arena_first_use(void);

static void *aalloc(size_t n) {
    arena_t *A = &g_arena;
    if (!A->base && !A->nextra) arena_first_use();
    n = (n + ARENA_SLACK + 63) & ~(size_t)63;
    if (A->used + n <= A->cap) {
        void *p = A->base + A->used;
        A->used += n;
        return p;
    }
    if (A->nextra >= ARENA_MAX_EXTRA) return NULL;
    void *p = aligned_alloc(64, n);
    if (!p) return NULL;
    A->extra[A->nextra++] = p;
    A->extra_bytes += n;
    return p;
}

static void areset(void) {
    arena_t *A = &g_arena;
    size_t need = A->used + A->extra_bytes;
    for (int i = 0; i < A->nextra; i++) free(A->extra[i]);
    A->nextra = 0;
    A->extra_bytes = 0;
    A->used = 0;
    if (need > A->cap) {
        free(A->base);
        A->base = NULL;
        A->cap = 0;
        if (need <= ARENA_MAX_RETAIN) {
            size_t c = (need + need / 4 + 4095) & ~(size_t)4095;
            A->base = aligned_alloc(64, c);
            if (A->base) A->cap = c;
        }
    }
}

static void cc_release(void);
void nlp_release_scratch(void) {
    cc_release();
    areset();
    free(g_arena.base);
    g_arena.base = NULL;
    g_arena.cap = 0;
}

static void tls_destructor(void *unused) {
    (void)unused;
    nlp_release_scratch();
}

static void arena_first_use(void) {
    arena_t *A = &g_arena;
    A->base = aligned_alloc(64, ARENA_INIT);
    A->cap = A->base ? ARENA_INIT : 0;
    A->used = 0;
    pthread_setspecific(g_tls_key, (void *)1); /* arm the thread-exit destructor */
}

/* ------------------------------------------------------------------ */
/* lookup tables (read-only after init)                                 */
/* ------------------------------------------------------------------ */

#define SHIFTFORDIV255(a) ((((a) >> 8) + a) >> 8)
#define MULDIV255(a, b, tmp) (tmp = (a) * (b) + 128, SHIFTFORDIV255(tmp))
#define CLIP8(v) ((v) <= 0 ? 0 : (v) < 256 ? (v) : 255)

/* index [a << 8 | c]; +4 bytes so that 32-bit gathers at any index stay in bounds */
static uint8_t LUT_UNPRE_PRE[65536 + 4];  /* RGBa->RGBA (rgba2rgbA) then RGBA->RGBa (rgbA2rgba) */
static uint8_t LUT_UNPRE_COMP[65536 + 4]; /* RGBa->RGBA then composite over opaque white */
static uint8_t LUT_COMP[65536 + 4];       /* composite RGBA over opaque white */
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static unsigned unpremul(unsigned c, unsigned a) {
    if (a == 255 || a == 0) return c;
    unsigned v = (255 * c) / a;
    return CLIP8(v);
}

static unsigned comp_white(unsigned c, unsigned a) {
    /* AlphaComposite.c with dst = (255,255,255,255), src = (c,.,.,a) */
    if (a == 0) return 255;
    const unsigned PB = 7;
    uint32_t dsta = 255, dstc = 255;
    uint32_t blend = dsta * (255 - a);
    uint32_t outa255 = a * 255 + blend;
    uint32_t coef1 = (uint32_t)(a * 255 * 255 * (1 << PB)) / outa255;
    uint32_t coef2 = 255 * (1 << PB) - coef1;
    uint32_t tmp = c * coef1 + dstc * coef2;
    return (SHIFTFORDIV255(tmp + (0x80 << PB))) >> PB;
}

static void init_tables(void) {
    pthread_key_create(&g_tls_key, tls_destructor);
    for (unsigned a = 0; a < 256; a++)
        for (unsigned c = 0; c < 256; c++) {
            unsigned tmp, u = unpremul(c, a);
            LUT_UNPRE_PRE[a << 8 | c] = (uint8_t)MULDIV255(u, a, tmp);
            LUT_UNPRE_COMP[a << 8 | c] = (uint8_t)comp_white(u, a);
            LUT_COMP[a << 8 | c] = (uint8_t)comp_white(c, a);
        }
}

/* ------------------------------------------------------------------ */
/* small image descriptor                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *p;
    int w, h, ps; /* rows are tightly packed: stride = w * ps */
} img_t;

static inline size_t img_bytes(int w, int h, int ps) { return (size_t)w * h * ps; }

/* ------------------------------------------------------------------ */
/* Reduce.c                                                             */
/* ------------------------------------------------------------------ */

static uint32_t division_UINT32(int divider, int result_bits) {
    uint32_t max_dividend = (1 << result_bits) * divider;
    float max_int = (1 << 30) * 4.0;
    return (uint32_t)(max_int / max_dividend);
}

/* premultiply 4 pixels (16 bytes) -> 16 x u16 (rgbA2rgba, MULDIV255) */
static inline __m256i premul16(__m128i px) {
    const __m256i abc = _mm256_setr_epi8(6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15,
                                         6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15);
    __m256i v = _mm256_cvtepu8_epi16(px);
    __m256i a = _mm256_shuffle_epi8(v, abc);
    a = _mm256_blend_epi16(a, _mm256_set1_epi16(255), 0x88);
    __m256i t = _mm256_add_epi16(_mm256_mullo_epi16(v, a), _mm256_set1_epi16(128));
    return _mm256_srli_epi16(_mm256_add_epi16(t, _mm256_srli_epi16(t, 8)), 8);
}

static void apply_alpha_lut(const uint8_t *in, uint8_t *out, size_t npix, const uint8_t *lut);

/* acc (4 x u16 per pixel) (=|+=) one or two source rows of w pixels.  ps 4: RGBA
 * (premultiplied on the fly if premul) or RGBX; ps 3: RGB expanded to RGB0. */
static inline void accum_rows(uint16_t *acc, const uint8_t *ra, const uint8_t *rb2, int w, int ps, int premul, int init) {
    int x = 0;
    if (ps == 4) {
        for (; x + 4 <= w; x += 4) {
            __m128i pa = _mm_loadu_si128((const __m128i *)(ra + x * 4));
            __m256i v = premul ? premul16(pa) : _mm256_cvtepu8_epi16(pa);
            if (rb2) {
                __m128i pb = _mm_loadu_si128((const __m128i *)(rb2 + x * 4));
                v = _mm256_add_epi16(v, premul ? premul16(pb) : _mm256_cvtepu8_epi16(pb));
            }
            if (!init) v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i *)(acc + x * 4)));
            _mm256_storeu_si256((__m256i *)(acc + x * 4), v);
        }
    } else {
        const __m128i ex = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        for (; x + 4 <= w && (size_t)x * 3 + 16 <= (size_t)w * 3; x += 4) {
            __m256i v = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(ra + x * 3)), ex));
            if (rb2)
                v = _mm256_add_epi16(v, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(rb2 + x * 3)), ex)));
            if (!init) v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i *)(acc + x * 4)));
            _mm256_storeu_si256((__m256i *)(acc + x * 4), v);
        }
    }
    for (; x < w; x++) {
        unsigned tmp;
        for (int k = 0; k < 4; k++) {
            unsigned v;
            if (ps == 3) {
                v = k < 3 ? ra[x * 3 + k] + (rb2 ? rb2[x * 3 + k] : 0) : 0;
            } else if (premul) {
                v = k == 3 ? ra[x * 4 + 3] : MULDIV255(ra[x * 4 + k], ra[x * 4 + 3], tmp);
                if (rb2) v += k == 3 ? rb2[x * 4 + 3] : MULDIV255(rb2[x * 4 + k], rb2[x * 4 + 3], tmp);
            } else {
                v = ra[x * 4 + k] + (rb2 ? rb2[x * 4 + k] : 0);
            }
            acc[x * 4 + k] = (uint16_t)(init ? v : acc[x * 4 + k] + v);
        }
    }
}

/* one output cell from cw source pixels of acc (4 x u16 each); Pillow rounding */
static inline void cell_out(const uint16_t *q, int cw, uint32_t am, uint32_t m, uint8_t *o) {
    __m128i s0 = _mm_setzero_si128(), s1 = s0;
    int c = 0;
    for (; c + 2 <= cw; c += 2) {
        s0 = _mm_add_epi32(s0, _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q + c * 4))));
        s1 = _mm_add_epi32(s1, _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q + c * 4 + 4))));
    }
    if (c < cw) s0 = _mm_add_epi32(s0, _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q + c * 4))));
    __m128i v = _mm_add_epi32(_mm_add_epi32(s0, s1), _mm_set1_epi32((int)am));
    v = _mm_srli_epi32(_mm_mullo_epi32(v, _mm_set1_epi32((int)m)), 24); /* uint32 wraparound as in C */
    v = _mm_packus_epi16(_mm_packus_epi32(v, v), v);
    int32_t o4 = _mm_cvtsi128_si32(v);
    memcpy(o, &o4, 4);
}

/* full-width cells for constant F (lets the compiler unroll the pixel loop) */
#define CELLS_F(F)                                                                                         \
    for (; ox < fullw; ox++) {                                                                             \
        const uint16_t *q = acc + (size_t)ox * (F) * 4;                                                    \
        __m128i s = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q));                              \
        for (int c = 1; c < (F); c++) s = _mm_add_epi32(s, _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q + c * 4)))); \
        __m128i v = _mm_srli_epi32(_mm_mullo_epi32(_mm_add_epi32(s, AM), M), 24);                         \
        v = _mm_packus_epi16(_mm_packus_epi32(v, v), v);                                                   \
        int32_t o4 = _mm_cvtsi128_si32(v);                                                                 \
        memcpy(orow + ox * 4, &o4, 4);                                                                     \
    }

/* horizontal f-sums of one accumulated row (nr source rows) -> w1 output pixels */
static void hsum_row(const uint16_t *acc, int w, int f, int nr, uint8_t *orow) {
    const int fullw = w / f, remw = w % f;
    const uint32_t cnt = (uint32_t)f * nr, m = division_UINT32(cnt, 8), am = cnt / 2;
    int ox = 0;
    if (f == 2) {
        /* cnt in {2, 4}: ((s + am) * m) >> 24 == (s + am) >> log2(cnt), all in u16 */
        const __m256i AM = _mm256_set1_epi16((short)am);
        const __m128i SH = _mm_cvtsi32_si128(cnt == 4 ? 2 : 1);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (; ox + 4 <= fullw; ox += 4) {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(acc + ox * 8));
            __m256i v1 = _mm256_loadu_si256((const __m256i *)(acc + ox * 8 + 16));
            v0 = _mm256_add_epi16(v0, _mm256_srli_si256(v0, 8)); /* [P0+P1, . | P2+P3, .] */
            v1 = _mm256_add_epi16(v1, _mm256_srli_si256(v1, 8));
            __m256i sm = _mm256_unpacklo_epi64(v0, v1);         /* [O0, O2 | O1, O3] */
            sm = _mm256_srl_epi16(_mm256_add_epi16(sm, AM), SH);
            __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(sm, sm), perm);
            _mm_storeu_si128((__m128i *)(orow + ox * 4), _mm256_castsi256_si128(p8));
        }
    }
    {
        const __m128i AM = _mm_set1_epi32((int)am), M = _mm_set1_epi32((int)m);
        switch (f) {
        case 2: CELLS_F(2) break;
        case 3: CELLS_F(3) break;
        case 4: CELLS_F(4) break;
        case 5: CELLS_F(5) break;
        case 6: CELLS_F(6) break;
        default:
            for (; ox < fullw; ox++) cell_out(acc + (size_t)ox * f * 4, f, am, m, orow + ox * 4);
        }
    }
    if (remw) {
        const uint32_t cr = (uint32_t)remw * nr;
        cell_out(acc + (size_t)fullw * f * 4, remw, cr / 2, division_UINT32(cr, 8), orow + fullw * 4);
    }
}
#undef CELLS_F

/* Box-reduce `in` by f (f >= 2) into `out` (ps 4).  premul: input is RGBA,
 * premultiply on load (Image.reduce converts RGBA->RGBa first).  lut (may be
 * NULL) is applied to channels 0..2 using channel 3 as alpha.
 * Every output cell = ((sum + n/2) * division_UINT32(n, 8)) >> 24 with n the
 * number of source pixels in the cell; this equals Pillow's specialised 2x2,
 * 3x3, 4x4, 5x5, NxN and ReduceCorners formulas. */
static int reduce_box(const img_t *in, int f, int premul, const uint8_t *lut, img_t *out) {
    const int w = in->w, h = in->h, ps = in->ps;
    const int w1 = (w - 1) / f + 1, h1 = (h - 1) / f + 1;
    const int nch = premul ? 4 : 3;
    const size_t rb = (size_t)w * ps;
    out->w = w1;
    out->h = h1;
    out->ps = 4;
    out->p = aalloc(img_bytes(w1, h1, 4));
    if (!out->p) return -1;
    /* u16 sums: column sums need nr*255 <= 65535 and 16-bit f-sums f*f*255 (f == 2 only) */
    const int wide = f > 257;
    void *accv = aalloc((size_t)w * 4 * (wide ? 4 : 2) + 64);
    if (!accv) return -1;
    uint16_t *acc16 = accv;
    uint32_t *acc32 = accv;
    const int fullw = w / f, remw = w % f;

    for (int oy = 0; oy < h1; oy++) {
        const int r0 = oy * f, r1 = r0 + f < h ? r0 + f : h, nr = r1 - r0;
        uint8_t *orow = out->p + (size_t)oy * w1 * 4;
        if (!wide) {
            int r = r0, init = 1;
            for (; r + 2 <= r1; r += 2, init = 0)
                accum_rows(acc16, in->p + (size_t)r * rb, in->p + (size_t)(r + 1) * rb, w, ps, premul, init);
            if (r < r1) accum_rows(acc16, in->p + (size_t)r * rb, NULL, w, ps, premul, init);
            hsum_row(acc16, w, f, nr, orow);
        } else {
            memset(acc32, 0, rb * 4);
            for (int r = r0; r < r1; r++) {
                const uint8_t *row = in->p + (size_t)r * rb;
                if (premul) {
                    for (size_t i = 0; i < rb; i += 4) {
                        unsigned al = row[i + 3], tmp;
                        acc32[i + 0] += MULDIV255(row[i + 0], al, tmp);
                        acc32[i + 1] += MULDIV255(row[i + 1], al, tmp);
                        acc32[i + 2] += MULDIV255(row[i + 2], al, tmp);
                        acc32[i + 3] += al;
                    }
                } else {
                    for (size_t i = 0; i < rb; i++) acc32[i] += row[i];
                }
            }
            const uint32_t cnt_full = (uint32_t)f * nr, cnt_rem = (uint32_t)remw * nr;
            const uint32_t m_full = division_UINT32(cnt_full, 8), a_full = cnt_full / 2;
            const uint32_t m_rem = remw ? division_UINT32(cnt_rem, 8) : 0, a_rem = cnt_rem / 2;
            for (int ox = 0; ox < w1; ox++) {
                const int c0 = ox * f, c1 = ox < fullw ? c0 + f : w;
                const uint32_t m = ox < fullw ? m_full : m_rem, am = ox < fullw ? a_full : a_rem;
                uint32_t sm[4] = {0, 0, 0, 0};
                for (int c = c0; c < c1; c++)
                    for (int k = 0; k < nch; k++) sm[k] += acc32[(size_t)c * ps + k];
                for (int k = 0; k < 4; k++) orow[ox * 4 + k] = (uint8_t)(((sm[k] + am) * m) >> 24);
            }
        }
        if (lut) apply_alpha_lut(orow, orow, (size_t)w1, lut);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Resample.c (bilinear, 8bpc)                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    int ksize;
    int *bounds;   /* [out*2]: xmin, count */
    int32_t *kk;   /* [out*ksize] fixed point */
    int kp;        /* pairs per output = (ksize + 1) / 2 */
    uint32_t *ph;  /* [out*kp] packed (k[2j] >> 11) | (k[2j+1] >> 11) << 16 */
    uint32_t *pl;  /* [out*kp] packed (k[2j] & 2047) | (k[2j+1] & 2047) << 16 */
} coeffs_t;

static inline double bilinear_filter(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

static int precompute_coeffs(int inSize, float in0, float in1, int outSize, coeffs_t *c) {
    double support, scale, filterscale;
    int xx, x, ksize, xmin, xmax;
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    support = 1.0 * filterscale;
    ksize = (int)ceil(support) * 2 + 1;
    double *k = aalloc(sizeof(double) * ksize);
    int32_t *kk = aalloc(sizeof(int32_t) * (size_t)outSize * ksize);
    int *bounds = aalloc(sizeof(int) * 2 * (size_t)outSize);
    if (!k || !kk || !bounds) return -1;
    double inv_filterscale = 1.0 / filterscale;
    for (xx = 0; xx < outSize; xx++) {
        double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        for (x = 0; x < xmax; x++) {
            double w = bilinear_filter((x + xmin - center + 0.5) * inv_filterscale);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0)
            for (x = 0; x < xmax; x++) k[x] /= ww;
        for (; x < ksize; x++) k[x] = 0;
        for (x = 0; x < ksize; x++) {
            int32_t v;
            if (k[x] < 0) v = (int)(-0.5 + k[x] * (1 << PRECISION_BITS));
            else v = (int)(0.5 + k[x] * (1 << PRECISION_BITS));
            kk[(size_t)xx * ksize + x] = v;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    c->ksize = ksize;
    c->bounds = bounds;
    c->kk = kk;
    /* 16-bit split for pmaddwd: k = kh * 2048 + kl, 0 <= kh <= 2048, 0 <= kl < 2048
     * (bilinear coefficients are never negative). */
    const int kp = (ksize + 1) / 2;
    uint32_t *ph = aalloc(sizeof(uint32_t) * (size_t)outSize * kp);
    uint32_t *pl = aalloc(sizeof(uint32_t) * (size_t)outSize * kp);
    if (!ph || !pl) return -1;
    for (xx = 0; xx < outSize; xx++) {
        const int32_t *kx = kk + (size_t)xx * ksize;
        for (int j = 0; j < kp; j++) {
            uint32_t k0 = (uint32_t)kx[2 * j], k1 = 2 * j + 1 < ksize ? (uint32_t)kx[2 * j + 1] : 0;
            ph[(size_t)xx * kp + j] = (k0 >> 11) | (k1 >> 11) << 16;
            pl[(size_t)xx * kp + j] = (k0 & 2047) | (k1 & 2047) << 16;
        }
    }
    c->kp = kp;
    c->ph = ph;
    c->pl = pl;
    return 0;
}

/* per-thread LRU cache of coefficient sets keyed by (inSize, outSize): image sizes
 * repeat a lot (e.g. every 256-wide big image needs the same 256 -> 128 filter). */
#define CC_N 8
typedef struct {
    int in, out;
    unsigned stamp;
    void *mem;
    coeffs_t c;
} cc_entry;
static __thread cc_entry g_cc[CC_N];
static __thread unsigned g_cc_clock;

static void cc_release(void) {
    for (int i = 0; i < CC_N; i++) {
        free(g_cc[i].mem);
        g_cc[i].mem = NULL;
    }
}

static int get_coeffs(int inSize, int outSize, coeffs_t *c) {
    int lru = 0;
    for (int i = 0; i < CC_N; i++) {
        if (g_cc[i].mem && g_cc[i].in == inSize && g_cc[i].out == outSize) {
            g_cc[i].stamp = ++g_cc_clock;
            *c = g_cc[i].c;
            return 0;
        }
        if (!g_cc[i].mem || (g_cc[lru].mem && g_cc[i].stamp < g_cc[lru].stamp)) lru = i;
    }
    coeffs_t t;
    if (precompute_coeffs(inSize, 0.0f, (float)inSize, outSize, &t)) return -1;
    const size_t nb = sizeof(int) * 2 * (size_t)outSize, nk = sizeof(int32_t) * (size_t)outSize * t.ksize,
                 np = sizeof(uint32_t) * (size_t)outSize * t.kp;
    uint8_t *m = malloc(nb + nk + 2 * np);
    if (!m) { *c = t; return 0; } /* arena copy is valid for this call */
    memcpy(m, t.bounds, nb);
    memcpy(m + nb, t.kk, nk);
    memcpy(m + nb + nk, t.ph, np);
    memcpy(m + nb + nk + np, t.pl, np);
    cc_entry *e = &g_cc[lru];
    free(e->mem);
    e->mem = m;
    e->in = inSize;
    e->out = outSize;
    e->stamp = ++g_cc_clock;
    e->c = t;
    e->c.bounds = (int *)m;
    e->c.kk = (int32_t *)(m + nb);
    e->c.ph = (uint32_t *)(m + nb + nk);
    e->c.pl = (uint32_t *)(m + nb + nk + np);
    *c = e->c;
    return 0;
}

static inline uint8_t clip8(int in) {
    int v = in >> PRECISION_BITS;
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

#ifdef __AVXVNNI__
#define DPW(acc, x, k) acc = _mm256_dpwssd_avx_epi32(acc, x, k)
#else
#define DPW(acc, x, k) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, k))
#endif

/* Sum over n taps of src(j) * k[j] (+ rounding half) for 32 bytes, src(j) = base + j*step,
 * clip8(>> 22) -> 32 bytes.  Coefficients are split k = kh * 2048 + kl (both < 2^15) so
 * that pmaddwd / vpdpwssd can process two taps per 16-bit pair exactly.
 * Accumulator lane order is {0-3,16-19}, {4-7,20-23}, {8-11,24-27}, {12-15,28-31},
 * which the final packus sequence puts back in order. */
static inline __m256i taps32(const uint8_t *base, size_t step, int n, const uint32_t *ph, const uint32_t *pl) {
    const __m256i z = _mm256_setzero_si256();
    const __m256i half = _mm256_set1_epi32(1 << (PRECISION_BITS - 1));
    __m256i h0 = z, h1 = z, h2 = z, h3 = z, l0 = half, l1 = half, l2 = half, l3 = half;
#define PAIR(A, B, jj)                                                                                      \
    do {                                                                                                    \
        __m256i KH = _mm256_set1_epi32((int)ph[jj]), KL = _mm256_set1_epi32((int)pl[jj]);                  \
        __m256i a = _mm256_loadu_si256((const __m256i *)(A)), b = _mm256_loadu_si256((const __m256i *)(B)); \
        __m256i lo = _mm256_unpacklo_epi8(a, b), hi = _mm256_unpackhi_epi8(a, b), w;                       \
        w = _mm256_unpacklo_epi8(lo, z); DPW(h0, w, KH); DPW(l0, w, KL);                                    \
        w = _mm256_unpackhi_epi8(lo, z); DPW(h1, w, KH); DPW(l1, w, KL);                                    \
        w = _mm256_unpacklo_epi8(hi, z); DPW(h2, w, KH); DPW(l2, w, KL);                                    \
        w = _mm256_unpackhi_epi8(hi, z); DPW(h3, w, KH); DPW(l3, w, KL);                                    \
    } while (0)
    /* odd tap count: the last pair uses the same source twice; its partner coefficient is 0 */
    const uint8_t *s0 = base, *s1 = base + step, *s2 = base + 2 * step, *s3 = base + 3 * step, *s4 = base + 4 * step;
    switch (n) {
    case 1: PAIR(s0, s0, 0); break;
    case 2: PAIR(s0, s1, 0); break;
    case 3: PAIR(s0, s1, 0); PAIR(s2, s2, 1); break;
    case 4: PAIR(s0, s1, 0); PAIR(s2, s3, 1); break;
    case 5: PAIR(s0, s1, 0); PAIR(s2, s3, 1); PAIR(s4, s4, 2); break;
    default:
        for (int j = 0; j < n; j += 2) {
            const uint8_t *A = base + (size_t)j * step;
            PAIR(A, j + 1 < n ? A + step : A, j / 2);
        }
    }
#undef PAIR
    /* (h * 2048 + l) >> 22 == (h + (l >> 11)) >> 11 */
#define FIN(h, l) _mm256_srai_epi32(_mm256_add_epi32(h, _mm256_srai_epi32(l, 11)), 11)
    __m256i r0 = FIN(h0, l0), r1 = FIN(h1, l1), r2 = FIN(h2, l2), r3 = FIN(h3, l3);
#undef FIN
    return _mm256_packus_epi16(_mm256_packus_epi32(r0, r1), _mm256_packus_epi32(r2, r3));
}

/* 8x8 transpose of 32-bit elements: r[i] = row i -> r[j] = column j */
static inline void tr8x8(__m256i *r) {
    __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

/* 8 pixels -> 8 x 32-bit (RGBX); ps 3 reads exactly 24 bytes */
static inline __m256i load8px(const uint8_t *q, int ps) {
    if (ps == 4) return _mm256_loadu_si256((const __m256i *)q);
    const __m256i sh = _mm256_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
                                        4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1);
    __m256i v = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(q + 8)), _mm_loadu_si128((const __m128i *)q));
    return _mm256_shuffle_epi8(v, sh);
}

/* Horizontal pass over rows [y0, y0+nrows) of in (ps 3/4) -> out (ps 4).
 * Blocks of 8 rows are transposed into pixel-major order (32 bytes per input
 * column = 8 rows x RGBX), so each tap is one broadcast coefficient times 32
 * bytes, exactly like the vertical pass; 8 output columns are transposed back.
 * Never reads outside the input rows. */
static int hpass(const img_t *in, int y0, int nrows, const coeffs_t *c, img_t *out) {
    const int inW = in->w, outW = out->w, ps = in->ps, ks = c->ksize;
    const size_t irb = (size_t)inW * ps, orb = (size_t)outW * 4;
    uint8_t *T = aalloc((size_t)inW * 32);
    if (!T) return -1;
    for (int rb = 0; rb < nrows; rb += 8) {
        const int nr = nrows - rb < 8 ? nrows - rb : 8;
        const uint8_t *rows[8];
        for (int i = 0; i < 8; i++) rows[i] = in->p + (size_t)(y0 + rb + (i < nr ? i : nr - 1)) * irb;
        int x = 0;
        for (; x + 8 <= inW; x += 8) {
            __m256i v[8];
            for (int i = 0; i < 8; i++) v[i] = load8px(rows[i] + (size_t)x * ps, ps);
            tr8x8(v);
            for (int j = 0; j < 8; j++) _mm256_store_si256((__m256i *)(T + (size_t)(x + j) * 32), v[j]);
        }
        for (; x < inW; x++)
            for (int i = 0; i < 8; i++) {
                uint8_t *d = T + (size_t)x * 32 + i * 4;
                const uint8_t *s = rows[i] + (size_t)x * ps;
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = ps == 4 ? s[3] : 0;
            }
        __m256i col[8];
        uint8_t *orow0 = out->p + (size_t)rb * orb;
        for (int xx = 0; xx < outW; xx++) {
            const int xmin = c->bounds[xx * 2], n = c->bounds[xx * 2 + 1];
            const int32_t *k = c->kk + (size_t)xx * ks;
            (void)k;
            col[xx & 7] = taps32(T + (size_t)xmin * 32, 32, n, c->ph + (size_t)xx * c->kp, c->pl + (size_t)xx * c->kp);
            if ((xx & 7) == 7 || xx == outW - 1) {
                const int x0 = xx & ~7, cnt = xx - x0 + 1;
                tr8x8(col);
                for (int i = 0; i < nr; i++) {
                    uint8_t *d = orow0 + (size_t)i * orb + (size_t)x0 * 4;
                    if (cnt == 8) _mm256_storeu_si256((__m256i *)d, col[i]);
                    else {
                        uint8_t tmp[32];
                        _mm256_storeu_si256((__m256i *)tmp, col[i]);
                        memcpy(d, tmp, (size_t)cnt * 4);
                    }
                }
            }
        }
    }
    return 0;
}

/* vertical pass: layout agnostic (row bytes = w*ps); in rows offset by y0 */
static void vpass(const img_t *in, int y0, const coeffs_t *c, img_t *out) {
    const size_t rb = (size_t)in->w * in->ps;
    const int ks = c->ksize;
    for (int yy = 0; yy < out->h; yy++) {
        const int ymin = c->bounds[yy * 2] + y0, n = c->bounds[yy * 2 + 1];
        const int32_t *k = c->kk + (size_t)yy * ks;
        const uint8_t *base = in->p + (size_t)ymin * rb;
        uint8_t *dst = out->p + (size_t)yy * rb;
        size_t x = 0;
        for (; x + 32 <= rb; x += 32) {
            _mm256_storeu_si256((__m256i *)(dst + x), taps32(base + x, rb, n, c->ph + (size_t)yy * c->kp, c->pl + (size_t)yy * c->kp));
        }
        for (; x < rb; x++) {
            int s = 1 << (PRECISION_BITS - 1);
            for (int y = 0; y < n; y++) s += base[(size_t)y * rb + x] * k[y];
            dst[x] = clip8(s);
        }
    }
}

/* ImagingResampleInner with BILINEAR.  in: ps 3 or 4.  Result ps: 4 if a
 * horizontal pass ran, else in->ps.  Caller guarantees size differs. */
static int resample(const img_t *in, int xsize, int ysize, img_t *out) {
    const int need_h = xsize != in->w;
    const int need_v = ysize != in->h;
    const int hfirst = !((in->h - ysize) > 0 && (in->h - ysize) > (in->w - xsize) * 2);
    coeffs_t cv = {0}, ch = {0};
    int ybox_first = 0, ybox_last = in->h;
    if ((need_h && hfirst) || need_v)
        if (get_coeffs(in->h, ysize, &cv)) return -1;
    if (need_h) {
        if (hfirst) {
            ybox_first = cv.bounds[0];
            ybox_last = cv.bounds[ysize * 2 - 2] + cv.bounds[ysize * 2 - 1];
            /* bounds shift is applied as a y0 offset in vpass instead */
        }
        if (get_coeffs(in->w, xsize, &ch)) return -1;
    }
    img_t tmp;
    if (hfirst) {
        const img_t *cur = in;
        int cur_y0 = 0;
        if (need_h) {
            tmp.w = xsize;
            tmp.h = ybox_last - ybox_first;
            tmp.ps = 4;
            tmp.p = aalloc(img_bytes(tmp.w, tmp.h, 4));
            if (!tmp.p) return -1;
            if (hpass(in, ybox_first, tmp.h, &ch, &tmp)) return -1;
            cur = &tmp;
            cur_y0 = -ybox_first; /* vertical bounds are relative to source rows */
        }
        if (need_v) {
            out->w = xsize;
            out->h = ysize;
            out->ps = cur->ps;
            out->p = aalloc(img_bytes(xsize, ysize, cur->ps));
            if (!out->p) return -1;
            vpass(cur, cur_y0, &cv, out);
        } else {
            *out = *cur;
        }
    } else {
        const img_t *cur = in;
        if (need_v) {
            tmp.w = in->w;
            tmp.h = ysize;
            tmp.ps = in->ps;
            tmp.p = aalloc(img_bytes(tmp.w, tmp.h, tmp.ps));
            if (!tmp.p) return -1;
            vpass(in, 0, &cv, &tmp);
            cur = &tmp;
        }
        if (need_h) {
            out->w = xsize;
            out->h = ysize;
            out->ps = 4;
            out->p = aalloc(img_bytes(xsize, ysize, 4));
            if (!out->p) return -1;
            if (hpass(cur, 0, ysize, &ch, out)) return -1;
        } else {
            *out = *cur;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* per-pixel LUT stage (unpremultiply / composite)                      */
/* ------------------------------------------------------------------ */

/* out[c] = lut[a << 8 | in[c]] for c = 0..2, alpha copied (in == out allowed) */
static void apply_alpha_lut(const uint8_t *in, uint8_t *out, size_t npix, const uint8_t *lut) {
    const __m256i m8 = _mm256_set1_epi32(0xff);
    size_t i = 0;
    for (; i + 8 <= npix; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(in + i * 4));
        __m256i ab = _mm256_slli_epi32(_mm256_srli_epi32(v, 24), 8);
        __m256i ir = _mm256_or_si256(ab, _mm256_and_si256(v, m8));
        __m256i ig = _mm256_or_si256(ab, _mm256_and_si256(_mm256_srli_epi32(v, 8), m8));
        __m256i ib = _mm256_or_si256(ab, _mm256_and_si256(_mm256_srli_epi32(v, 16), m8));
        __m256i r = _mm256_and_si256(_mm256_i32gather_epi32((const int *)lut, ir, 1), m8);
        __m256i g = _mm256_and_si256(_mm256_i32gather_epi32((const int *)lut, ig, 1), m8);
        __m256i b = _mm256_and_si256(_mm256_i32gather_epi32((const int *)lut, ib, 1), m8);
        __m256i o = _mm256_or_si256(_mm256_or_si256(r, _mm256_slli_epi32(g, 8)),
                                    _mm256_or_si256(_mm256_slli_epi32(b, 16), _mm256_andnot_si256(_mm256_set1_epi32(0xffffff), v)));
        _mm256_storeu_si256((__m256i *)(out + i * 4), o);
    }
    for (; i < npix; i++) {
        const uint8_t *L = lut + ((unsigned)in[i * 4 + 3] << 8);
        out[i * 4 + 0] = L[in[i * 4 + 0]];
        out[i * 4 + 1] = L[in[i * 4 + 1]];
        out[i * 4 + 2] = L[in[i * 4 + 2]];
        out[i * 4 + 3] = in[i * 4 + 3];
    }
}

static void premultiply(const uint8_t *in, uint8_t *out, size_t npix) {
    size_t i = 0, nb = npix * 4;
    for (; i + 16 <= nb; i += 16) {
        __m256i v = premul16(_mm_loadu_si128((const __m128i *)(in + i)));
        __m128i p = _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storeu_si128((__m128i *)(out + i), p);
    }
    for (; i < nb; i += 4) {
        unsigned a = in[i + 3], tmp;
        out[i + 0] = MULDIV255(in[i + 0], a, tmp);
        out[i + 1] = MULDIV255(in[i + 1], a, tmp);
        out[i + 2] = MULDIV255(in[i + 2], a, tmp);
        out[i + 3] = a;
    }
}

/* ------------------------------------------------------------------ */
/* Convert.c rgb2hsv                                                     */
/* ------------------------------------------------------------------ */

void nlp_rgb2hsv_scalar(uint8_t *out, const uint8_t *in) {
    float h, s, rc, gc, bc, cr;
    uint8_t maxc, minc;
    uint8_t r, g, b;
    uint8_t uh, us, uv;
    r = in[0];
    g = in[1];
    b = in[2];
    maxc = r > g ? r : g;
    maxc = maxc > b ? maxc : b;
    minc = r < g ? r : g;
    minc = minc < b ? minc : b;
    uv = maxc;
    if (minc == maxc) {
        uh = 0;
        us = 0;
    } else {
        cr = (float)(maxc - minc);
        s = cr / (float)maxc;
        rc = ((float)(maxc - r)) / cr;
        gc = ((float)(maxc - g)) / cr;
        bc = ((float)(maxc - b)) / cr;
        if (r == maxc) {
            h = bc - gc;
        } else if (g == maxc) {
            h = 2.0 + rc - bc;
        } else {
            h = 4.0 + gc - rc;
        }
        h = fmod((h / 6.0 + 1.0), 1.0);
        uh = (uint8_t)CLIP8((int)(h * 255.0));
        us = (uint8_t)CLIP8((int)(s * 255.0));
    }
    out[0] = uh;
    out[1] = us;
    out[2] = uv;
}

/* (int)((double)v * 255.0) for 0 <= v <= 1 (float lanes), computed in float:
 * p = round(v*255) can only cross an integer upwards, and fma gives the sign of the
 * exact v*255 - floor(p). */
static inline __m256i trunc255(__m256 v) {
    const __m256 k255 = _mm256_set1_ps(255.0f);
    __m256 p = _mm256_mul_ps(v, k255);
    __m256i t = _mm256_cvttps_epi32(p);              /* floor(p) for p >= 0 */
    __m256 e = _mm256_fmsub_ps(v, k255, _mm256_cvtepi32_ps(t));
    __m256i lt = _mm256_castps_si256(_mm256_cmp_ps(e, _mm256_setzero_ps(), _CMP_LT_OQ));
    return _mm256_add_epi32(t, lt); /* lt = -1 when exact v*255 < floor(p) */
}


/* Pillow rgb2hsv_row for 8 pixels (r,g,b as int32 lanes 0..255). */
static inline void hsv8(__m256i r, __m256i g, __m256i b, uint8_t *H, uint8_t *S, uint8_t *V) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i maxc = _mm256_max_epi32(r, _mm256_max_epi32(g, b));
    __m256i minc = _mm256_min_epi32(r, _mm256_min_epi32(g, b));
    __m256i cri = _mm256_sub_epi32(maxc, minc);
    __m256i flat = _mm256_cmpeq_epi32(cri, zero);
    /* flat pixels (maxc == minc) get cr = 1, max = max+1 to avoid 0/0; masked at the end */
    __m256 crf = _mm256_cvtepi32_ps(_mm256_sub_epi32(cri, flat));
    __m256 maxf = _mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, flat));
#ifndef NLP_HSV_FMA
    __m256 s = _mm256_div_ps(crf, maxf);
#else
    /* NLP_HSV_FMA variant: one division for both denominators, inv = 1/(cr*max)
     * (the product is exact), then one FMA residual correction per quotient.
     * Exhaustively verified exact as well, but measured ~20% slower (the kernel is
     * bound by vector-ALU uops, not by the divider). */
    __m256 inv = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(crf, maxf));
    __m256 rcr = _mm256_mul_ps(maxf, inv), rmx = _mm256_mul_ps(crf, inv);
    __m256 s0 = _mm256_mul_ps(crf, rmx);
    __m256 s = _mm256_fmadd_ps(_mm256_fnmadd_ps(s0, maxf, crf), rmx, s0);
#endif
    /* Of rc, gc, bc only the middle channel needs a division (max -> 0/cr = 0,
     * min -> cr/cr = 1, both exact).  With q = (max - mid) / cr, Pillow's
     *   r max: h = bc - gc        = g>=b ? 1 - q : q - 1
     *   g max: h = 2.0 + rc - bc  = r>=b ? 1 + q : 3 - q
     *   b max: h = 4.0 + gc - rc  = g>=r ? 3 + q : 5 - q
     * where the double expressions are exact before the final float rounding,
     * so a single float add/sub reproduces them.  The case is looked up from the
     * strict comparisons P = g>r, Q = b>g, R = b>r (idx = 4P + 2Q + R):
     *   0: r max, b min: 1 - q      2: r max, g min: q - 1     3: b max, g min: 5 - q
     *   4: g max, b min: 1 + q      5: g max, r min: 3 - q     7: b max, r min: 3 + q
     * (1 and 6 are impossible).  Ties pick Pillow's branch except idx 3 with r == g,
     * where Pillow computes 3 + q with q == 1: both give exactly 4.0f. */
    __m256i mid = _mm256_sub_epi32(_mm256_add_epi32(r, _mm256_add_epi32(g, b)), _mm256_add_epi32(maxc, minc));
    __m256 dmf = _mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, mid));
#ifndef NLP_HSV_FMA
    __m256 q = _mm256_div_ps(dmf, crf);
#else
    __m256 q0 = _mm256_mul_ps(dmf, rcr);
    __m256 q = _mm256_fmadd_ps(_mm256_fnmadd_ps(q0, crf, dmf), rcr, q0);
#endif
    __m256i idx = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(_mm256_cmpgt_epi32(g, r), _mm256_set1_epi32(4)),
                                                  _mm256_and_si256(_mm256_cmpgt_epi32(b, g), _mm256_set1_epi32(2))),
                                  _mm256_and_si256(_mm256_cmpgt_epi32(b, r), _mm256_set1_epi32(1)));
    const __m256 tabA = _mm256_setr_ps(1.0f, 0.0f, -1.0f, 5.0f, 1.0f, 3.0f, 0.0f, 3.0f);
    const __m256 tabS = _mm256_castsi256_ps(_mm256_setr_epi32((int)0x80000000u, 0, 0, (int)0x80000000u, 0, (int)0x80000000u, 0, 0));
    __m256 h = _mm256_add_ps(_mm256_permutevar8x32_ps(tabA, idx),
                             _mm256_xor_ps(q, _mm256_permutevar8x32_ps(tabS, idx)));
    /* Pillow: h = fmod(h / 6.0 + 1.0, 1.0) in double, stored to float.  Float shortcut
     * (h/6 in float, +1 if h < 0) verified identical for every RGB triple. */
#ifndef NLP_HSV_FMA
    __m256 y = _mm256_div_ps(h, _mm256_set1_ps(6.0f));
#else
    /* h / 6 via reciprocal + one FMA residual correction (exhaustively verified) */
    /* (plain h * (1/6) is NOT exact: 20586 colours differ) */
    const __m256 c6 = _mm256_set1_ps(1.0f / 6.0f);
    __m256 y0 = _mm256_mul_ps(h, c6);
    __m256 y = _mm256_fmadd_ps(_mm256_fnmadd_ps(y0, _mm256_set1_ps(6.0f), h), c6, y0);
#endif
    /* + 1.0f where h < 0 (h is never -0.0f) */
    __m256 hf = _mm256_add_ps(y, _mm256_and_ps(_mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(h), 31)),
                                               _mm256_set1_ps(1.0f)));
    __m256i uh = _mm256_andnot_si256(flat, trunc255(hf));
    __m256i us = _mm256_andnot_si256(flat, trunc255(s));
    /* CLIP8 is a no-op here: 0 <= uh, us <= 255 */
    __m256i hs = _mm256_packus_epi32(uh, us);           /* [h0-3 s0-3 | h4-7 s4-7] u16 */
    __m256i vv = _mm256_packus_epi32(maxc, maxc);       /* [v0-3 v0-3 | v4-7 v4-7] */
    __m256i p8 = _mm256_packus_epi16(hs, vv);           /* [h0-3 s0-3 v0-3 v0-3 | h4-7 s4-7 v4-7 v4-7] */
    p8 = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    __m128i lo = _mm256_castsi256_si128(p8), hi = _mm256_extracti128_si256(p8, 1);
    _mm_storel_epi64((__m128i *)H, lo);
    _mm_storel_epi64((__m128i *)S, _mm_unpackhi_epi64(lo, lo));
    _mm_storel_epi64((__m128i *)V, hi);
}

static void hsv_planes(const img_t *im, uint8_t *H, uint8_t *S, uint8_t *V) {
    const size_t n = (size_t)im->w * im->h;
    const uint8_t *p = im->p;
    const __m256i m8 = _mm256_set1_epi32(0xff);
    size_t i = 0;
    if (im->ps == 4) {
        for (; i + 8 <= n; i += 8) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(p + i * 4));
            hsv8(_mm256_and_si256(v, m8), _mm256_and_si256(_mm256_srli_epi32(v, 8), m8),
                 _mm256_and_si256(_mm256_srli_epi32(v, 16), m8), H + i, S + i, V + i);
        }
    } else {
        const __m256i sh = _mm256_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
                                            4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1);
        for (; i + 8 <= n; i += 8) {
            const uint8_t *q = p + i * 3;
            __m256i v = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(q + 8)),
                                         _mm_loadu_si128((const __m128i *)q));
            v = _mm256_shuffle_epi8(v, sh);
            hsv8(_mm256_and_si256(v, m8), _mm256_and_si256(_mm256_srli_epi32(v, 8), m8),
                 _mm256_and_si256(_mm256_srli_epi32(v, 16), m8), H + i, S + i, V + i);
        }
    }
    for (; i < n; i++) {
        uint8_t o[3];
        nlp_rgb2hsv_scalar(o, p + i * im->ps);
        H[i] = o[0];
        S[i] = o[1];
        V[i] = o[2];
    }
}

/* exhaustive self-check of the vector HSV vs the Pillow scalar code.
 * returns number of mismatching RGB triples (0 expected). */
long nlp_hsv_selftest(void) {
    long bad = 0;
    uint8_t *buf = malloc(256 * 256 * 4);
    uint8_t *H = malloc(65536), *S = malloc(65536), *V = malloc(65536);
    for (int r = 0; r < 256; r++) {
        for (int g = 0; g < 256; g++)
            for (int b = 0; b < 256; b++) {
                uint8_t *q = buf + (g * 256 + b) * 4;
                q[0] = r; q[1] = g; q[2] = b; q[3] = 0;
            }
        img_t im = {buf, 256, 256, 4};
        hsv_planes(&im, H, S, V);
        for (int i = 0; i < 65536; i++) {
            uint8_t o[3];
            nlp_rgb2hsv_scalar(o, buf + i * 4);
            if (o[0] != H[i] || o[1] != S[i] || o[2] != V[i]) bad++;
        }
    }
    free(buf); free(H); free(S); free(V);
    return bad;
}

/* ------------------------------------------------------------------ */
/* public entry                                                          */
/* ------------------------------------------------------------------ */

#ifdef NLP_PROF
#include <x86intrin.h>
static __thread uint64_t g_prof[8];
void nlp_prof_get(uint64_t *o) { memcpy(o, g_prof, sizeof g_prof); memset(g_prof, 0, sizeof g_prof); }
#define PROF(i) do { uint64_t t_ = __rdtsc(); g_prof[i] += t_ - t_prev; t_prev = t_; } while (0)
#define PROF_START uint64_t t_prev = __rdtsc()
#else
#define PROF(i) do { } while (0)
#define PROF_START do { } while (0)
#endif

static int target_size(int w, int h, int *nw, int *nh) {
    const int m = w > h ? w : h;
    const double k = 256.0 / m;
    double a = rint(w * k), b = rint(h * k); /* Python round(): half to even */
    *nw = a < 1 ? 1 : (int)a;
    *nh = b < 1 ? 1 : (int)b;
    return 0;
}

int nl_prepare(const nl_img *in, nl_prep *out) {
    memset(out, 0, sizeof *out);
    if (!in || !in->px || in->w < 1 || in->h < 1 || (in->ch != 3 && in->ch != 4)) return NL_ERROR;
    pthread_once(&g_once, init_tables);
    PROF_START;
    const int alpha = in->ch == 4;
    int rc = NL_ERROR;
    img_t src = {in->px, in->w, in->h, in->ch};
    out->orig = in->w > in->h ? in->w : in->h;
    const int f = out->orig / NL_EYE_SIZE;

    img_t cur = src, big;
    int w1, h1, nw, nh;
    if (f >= 2) {
        w1 = (in->w - 1) / f + 1;
        h1 = (in->h - 1) / f + 1;
    } else {
        w1 = in->w;
        h1 = in->h;
    }
    target_size(w1, h1, &nw, &nh);
    const int need_resize = nw != w1 || nh != h1;

    if (f >= 2) {
        const uint8_t *lut = alpha ? (need_resize ? LUT_UNPRE_PRE : LUT_UNPRE_COMP) : NULL;
        if (reduce_box(&src, f, alpha, lut, &cur)) goto done;
        /* cur: RGBa (alpha && need_resize), RGB-composited (alpha && !need_resize), or RGBX */
    } else if (alpha && need_resize) {
        img_t pm = {aalloc(img_bytes(w1, h1, 4)), w1, h1, 4};
        if (!pm.p) goto done;
        premultiply(src.p, pm.p, (size_t)w1 * h1);
        cur = pm;
    } else if (alpha) {
        img_t cm = {aalloc(img_bytes(w1, h1, 4)), w1, h1, 4};
        if (!cm.p) goto done;
        apply_alpha_lut(src.p, cm.p, (size_t)w1 * h1, LUT_COMP);
        cur = cm;
    }
    PROF(0);

    if (need_resize) {
        if (resample(&cur, nw, nh, &big)) goto done;
        if (alpha) apply_alpha_lut(big.p, big.p, (size_t)nw * nh, LUT_UNPRE_COMP);
    } else {
        big = cur;
    }
    PROF(1);

    const int sw = big.w / 2 > 1 ? big.w / 2 : 1, sh = big.h / 2 > 1 ? big.h / 2 : 1;
    img_t small;
    if (sw == big.w && sh == big.h) {
        small = big;
    } else {
        if (resample(&big, sw, sh, &small)) goto done;
    }

    PROF(2);
    const size_t nb = (size_t)big.w * big.h, ns = (size_t)sw * sh;
    uint8_t *planes = malloc(3 * nb + 3 * ns + 32);
    if (!planes) goto done;
    out->bw = big.w;
    out->bh = big.h;
    out->sw = sw;
    out->sh = sh;
    out->bH = planes;
    out->bS = planes + nb;
    out->bV = planes + 2 * nb;
    out->sH = planes + 3 * nb;
    out->sS = planes + 3 * nb + ns;
    out->sV = planes + 3 * nb + 2 * ns;
    hsv_planes(&big, out->bH, out->bS, out->bV);
    hsv_planes(&small, out->sH, out->sS, out->sV);
    PROF(3);
    rc = NL_OK;
done:
    areset();
    if (rc != NL_OK) memset(out, 0, sizeof *out);
    return rc;
}

void nl_prep_free(nl_prep *p) {
    if (!p) return;
    free(p->bH); /* single allocation holds all six planes */
    memset(p, 0, sizeof *p);
}

/* test export: HSV of n packed RGB pixels (ps 3) */
void nlp_hsv(const uint8_t *rgb, size_t n, uint8_t *H, uint8_t *S, uint8_t *V) {
    img_t im = {(uint8_t *)rgb, (int)n, 1, 3};
    hsv_planes(&im, H, S, V);
}
