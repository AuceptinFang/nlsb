/* Fast Huffman (entropy) decoding plugged into libjpeg-turbo.
 *
 * libjpeg-turbo keeps doing everything else (markers, table validation, coefficient
 * buffering, block smoothing, IDCT, upsampling, colour conversion).  We only replace
 * cinfo->entropy->decode_mcu for each scan, with a decoder that has the exact semantics
 * of jdhuff.c / jdphuff.c on well-formed data (including the extended natural-order
 * table used for out-of-range coefficient indices).  On anything unusual (bad code,
 * running past the end of the segment, unexpected marker, FF fill bytes, refinement
 * symbol size != 1, ...) the decoder sets ->abort and returns FALSE ("suspend"); the
 * caller then discards the result and redoes the image with plain libjpeg-turbo.
 *
 * The hook is installed from the progress monitor, which libjpeg calls before every
 * consume_input() of the preload loop (multi-scan files) and at every
 * jpeg_read_scanlines() (single-scan files) - i.e. always after a scan's start_pass
 * and before its first MCU is decoded.
 *
 * Included from decode.c (single translation unit).
 */
#include <jpegint.h>

#define JFB 11 /* lookup bits */

/* LUT entry: see EK/EL/ER/ES/EV below
 * kind 1: code+value fit: consume len, run r, coefficient value
 * kind 2: code fits, value bits do not: consume len (code only), then s bits
 * kind 3: s == 0 symbol (EOB/EOBr/ZRL): consume len, run r
 * kind 0: code longer than JFB bits -> canonical slow path */
/* layout: [5:0] len  [7:6] kind  [11:8] r  [15:12] s  [31:16] value (int16).
 * len in the low bits lets the shift use the entry directly (x86 masks shift counts). */
/* kind bits: 0x40 fused (kind 1), 0x80 s==0 symbol (kind 3), neither with len>0 (kind 2),
 * entry 0 (kind 0) */
#define EK(e) (((e) & 0x40) ? 1u : ((e) & 0x80) ? 3u : ((e) & 63) ? 2u : 0u)
#define EL(e) ((e) & 63)
#define ER(e) (((e) >> 8) & 15)
#define ES(e) (((e) >> 12) & 15)
#define EV(e) ((int)((int32_t)(e) >> 16))
#define MKE(k, l, r, s, v) (((uint32_t)(l)) | ((k) == 1 ? 0x40u : (k) == 3 ? 0x80u : 0u) | ((uint32_t)(r) << 8) | ((uint32_t)(s) << 12) | ((uint32_t)(v) << 16))

typedef struct {
    uint32_t lut[1 << JFB];
    int32_t maxcode[18];
    int32_t valoff[18];
    uint8_t huffval[256];
    const JHUFF_TBL *src; /* table this was built from */
    uint8_t bits[17];     /* copy, to detect redefinition */
    uint8_t vals[256];
} jh_tab;

static const int nl_natural[64 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

static inline int hext(int v, int s) { return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v; }

/* returns 0 ok, -1 bad table */
static int jh_build(jh_tab *t, const JHUFF_TBL *h, int isDC) {
    uint8_t hsize[257];
    unsigned hcode[257];
    int p = 0;
    for (int l = 1; l <= 16; l++) {
        int i = h->bits[l];
        if (p + i > 256) return -1;
        while (i--) hsize[p++] = (uint8_t)l;
    }
    hsize[p] = 0;
    int nsym = p;
    unsigned code = 0;
    int si = hsize[0];
    p = 0;
    while (hsize[p]) {
        while (hsize[p] == si) { hcode[p++] = code; code++; }
        if (code >= (1u << si)) return -1;
        code <<= 1;
        si++;
    }
    p = 0;
    for (int l = 1; l <= 16; l++) {
        if (h->bits[l]) {
            t->valoff[l] = p - (int)hcode[p];
            p += h->bits[l];
            t->maxcode[l] = (int32_t)hcode[p - 1];
        } else {
            t->maxcode[l] = -1;
        }
    }
    t->maxcode[17] = 0x7FFFFFFF;
    memcpy(t->huffval, h->huffval, 256);
    if (isDC)
        for (int i = 0; i < nsym; i++)
            if (h->huffval[i] > 15) return -1;
    memset(t->lut, 0, sizeof t->lut);
    p = 0;
    for (int l = 1; l <= 16; l++) {
        for (int i = 0; i < h->bits[l]; i++, p++) {
            if (l > JFB) continue;
            int sym = h->huffval[p];
            int r = isDC ? 0 : sym >> 4, s = isDC ? sym : sym & 15;
            unsigned base = hcode[p] << (JFB - l), n = 1u << (JFB - l);
            for (unsigned j = 0; j < n; j++) {
                uint32_t e;
                if (s == 0) e = MKE(3, l, r, 0, 0);
                else if (l + s <= JFB) {
                    int v = (int)((j >> (JFB - l - s)) & ((1u << s) - 1));
                    e = MKE(1, l + s, r, s, hext(v, s));
                } else e = MKE(2, l, r, s, 0);
                t->lut[base + j] = e;
            }
        }
    }
    t->src = h;
    memcpy(t->bits, h->bits, 17);
    memcpy(t->vals, h->huffval, 256);
    return 0;
}

typedef struct {
    /* file */
    const uint8_t *buf;
    size_t len;
    /* destuffed segment of the current scan */
    uint8_t *d;
    size_t dcap, dn;
    size_t *ivl_end;     /* end byte offset (in d) of each restart interval */
    int *ivl_rst;        /* RST number that terminated interval i (-1: none / scan end) */
    size_t nivl, ivlcap;
    size_t ivl;          /* current interval */
    const uint8_t *seg_end; /* position of the marker that ends the scan's data */
    /* bit reader */
    const uint8_t *p;
    uint64_t acc;
    int cnt;
    /* scan state */
    int last_dc[4];
    int eobrun;
    unsigned rtg;        /* restarts to go */
    unsigned rst_expect;
    long mcu_done, mcu_total;
    int abort;
    int disabled;
    int installed_scans;
    /* tables */
    jh_tab tabs[8];      /* 0..3 DC, 4..7 AC */
    int tab_ok[8];
    const jh_tab *dct[D_MAX_BLOCKS_IN_MCU], *act[D_MAX_BLOCKS_IN_MCU];
    int memb[D_MAX_BLOCKS_IN_MCU];
} nlj_ctx;

#define BR_LOAD(X) const uint8_t *p = (X)->p; uint64_t acc = (X)->acc; int cnt = (X)->cnt
#define BR_SAVE(X) do { (X)->p = p; (X)->acc = acc; (X)->cnt = cnt; } while (0)
/* branchless refill: afterwards cnt >= 56 (bits beyond cnt in acc are the true next
 * bits, so re-OR-ing them on the next refill is harmless) */
static inline uint64_t be64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return __builtin_bswap64(v); }
#define REFILL() do { acc |= be64(p) >> cnt; p += (63 - cnt) >> 3; cnt |= 56; } while (0)
#define CONSUME(n) do { acc <<= (n); cnt -= (n); } while (0)
#define PEEK(n) ((unsigned)(acc >> (64 - (n))))

static inline size_t br_pos_bits(const nlj_ctx *X, const uint8_t *p, int cnt) {
    return (size_t)(p - X->d) * 8 - (size_t)cnt;
}

/* canonical decode of a code longer than JFB bits; returns symbol or -1; *len set */
static inline int jh_slow(const jh_tab *t, uint64_t acc, int *len) {
    unsigned code16 = (unsigned)(acc >> 48);
    for (int l = JFB + 1; l <= 16; l++) {
        int32_t c = (int32_t)(code16 >> (16 - l));
        if (c <= t->maxcode[l]) {
            int idx = c + t->valoff[l];
            if (idx < 0 || idx > 255) return -1;
            *len = l;
            return t->huffval[idx];
        }
    }
    return -1;
}

/* Decode one symbol into kind/len/r/s/v using the LUT (or slow path).  Returns 0 or -1. */
#define DECODE_SYM(T, RUN, SZ, VAL, HASVAL, FAIL)                              \
    do {                                                                       \
        uint32_t e_ = (T)->lut[PEEK(JFB)];                                     \
        unsigned k_ = EK(e_);                                                  \
        if (__builtin_expect(k_ == 1, 1)) {                                    \
            acc <<= (e_ & 63); cnt -= (int)(e_ & 63);                          \
            RUN = (int)ER(e_); SZ = (int)ES(e_); VAL = EV(e_); HASVAL = 1;     \
        } else if (k_ == 3) {                                                  \
            CONSUME(EL(e_)); RUN = (int)ER(e_); SZ = 0; VAL = 0; HASVAL = 0;   \
        } else {                                                               \
            int sym_, l_;                                                      \
            if (k_ == 2) { l_ = (int)EL(e_); sym_ = (int)((ER(e_) << 4) | ES(e_)); } \
            else { sym_ = jh_slow((T), acc, &l_); if (sym_ < 0) FAIL; }        \
            CONSUME(l_);                                                       \
            RUN = sym_ >> 4; SZ = sym_ & 15;                                   \
            if (SZ) { VAL = hext((int)PEEK(SZ), SZ); CONSUME(SZ); HASVAL = 1; } \
            else { VAL = 0; HASVAL = 0; }                                      \
        }                                                                      \
    } while (0)

/* DC tables store sym in the s field (r = 0), so the same macro works for them too
 * (sym_ for kind 0/2 must then be taken as s only): */
#define DECODE_DC(T, VAL, FAIL)                                                \
    do {                                                                       \
        uint32_t e_ = (T)->lut[PEEK(JFB)];                                     \
        unsigned k_ = EK(e_);                                                  \
        if (k_ == 1) { CONSUME(EL(e_)); VAL = EV(e_); }                        \
        else if (k_ == 3) { CONSUME(EL(e_)); VAL = 0; }                        \
        else {                                                                 \
            int s_, l_;                                                        \
            if (k_ == 2) { l_ = (int)EL(e_); s_ = (int)ES(e_); }               \
            else { s_ = jh_slow((T), acc, &l_); if (s_ < 0 || s_ > 15) FAIL; } \
            CONSUME(l_);                                                       \
            if (s_) { REFILL(); VAL = hext((int)PEEK(s_), s_); CONSUME(s_); } else VAL = 0; \
        }                                                                      \
    } while (0)

static int nlj_restart(nlj_ctx *X, j_decompress_ptr cinfo) {
    size_t pos = br_pos_bits(X, X->p, X->cnt);
    if (X->ivl >= X->nivl) return -1;
    size_t endb = X->ivl_end[X->ivl] * 8;
    if (pos > endb) return -1;                       /* ran into the marker */
    if (X->ivl_rst[X->ivl] != (int)X->rst_expect) return -1;
    X->rst_expect = (X->rst_expect + 1) & 7;
    X->ivl++;
    X->p = X->d + X->ivl_end[X->ivl - 1];
    X->acc = 0;
    X->cnt = 0;
    for (int i = 0; i < 4; i++) X->last_dc[i] = 0;
    X->eobrun = 0;
    X->rtg = cinfo->restart_interval;
    return 0;
}

static inline int nlj_mcu_end(nlj_ctx *X, j_decompress_ptr cinfo) {
    size_t pos = br_pos_bits(X, X->p, X->cnt);
    if (X->ivl >= X->nivl || pos > X->ivl_end[X->ivl] * 8) return -1;
    if (cinfo->restart_interval) X->rtg--;
    if (++X->mcu_done == X->mcu_total) {
        cinfo->src->next_input_byte = X->seg_end;
        cinfo->src->bytes_in_buffer = (size_t)(X->buf + X->len - X->seg_end);
    }
    return 0;
}

static boolean nlj_seq(j_decompress_ptr cinfo, JBLOCKROW *MCU_data) {
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort) return FALSE;
    if (cinfo->restart_interval && X->rtg == 0 && nlj_restart(X, cinfo)) goto fail;
    {
        BR_LOAD(X);
        int nb = cinfo->blocks_in_MCU;
        for (int b = 0; b < nb; b++) {
            JCOEF *blk = MCU_data[b][0];
            const jh_tab *dt = X->dct[b], *at = X->act[b];
            int ci = X->memb[b];
            REFILL();
            int diff;
            DECODE_DC(dt, diff, goto failb);
            int dc = X->last_dc[ci] + diff;
            X->last_dc[ci] = dc;
            blk[0] = (JCOEF)dc;
            const uint32_t *lut = at->lut;
            long k = 1;
            for (;;) {
                if (cnt < 32) REFILL();
                uint32_t e = lut[acc >> (64 - JFB)];
                if (__builtin_expect((e & 0x40) != 0, 1)) {
                    acc <<= (e & 63);
                    cnt -= (int)(e & 63);
                    k += (e >> 8) & 15;
                    blk[nl_natural[k]] = (JCOEF)((int32_t)e >> 16);
                    if (++k >= 64) break;
                    continue;
                }
                int r, s, v, hv;
                DECODE_SYM(at, r, s, v, hv, goto failb);
                if (hv) {
                    k += r;
                    blk[nl_natural[k]] = (JCOEF)v;
                } else {
                    if (r != 15) break;
                    k += 15;
                }
                if (++k >= 64) break;
            }
        }
        BR_SAVE(X);
    }
    if (nlj_mcu_end(X, cinfo)) goto fail;
    return TRUE;
failb:
fail:
    X->abort = 1;
    return FALSE;
}

static boolean nlj_dc_first(j_decompress_ptr cinfo, JBLOCKROW *MCU_data) {
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort) return FALSE;
    if (cinfo->restart_interval && X->rtg == 0 && nlj_restart(X, cinfo)) goto fail;
    {
        BR_LOAD(X);
        int Al = cinfo->Al;
        for (int b = 0; b < cinfo->blocks_in_MCU; b++) {
            JCOEF *blk = MCU_data[b][0];
            int ci = X->memb[b];
            REFILL();
            int diff;
            DECODE_DC(X->dct[b], diff, goto fail);
            int dc = X->last_dc[ci] + diff;
            X->last_dc[ci] = dc;
            blk[0] = (JCOEF)(int)((unsigned)dc << Al);
        }
        BR_SAVE(X);
    }
    if (nlj_mcu_end(X, cinfo)) goto fail;
    return TRUE;
fail:
    X->abort = 1;
    return FALSE;
}

static boolean nlj_dc_refine(j_decompress_ptr cinfo, JBLOCKROW *MCU_data) {
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort) return FALSE;
    if (cinfo->restart_interval && X->rtg == 0 && nlj_restart(X, cinfo)) goto fail;
    {
        BR_LOAD(X);
        int p1 = 1 << cinfo->Al;
        for (int b = 0; b < cinfo->blocks_in_MCU; b++) {
            JCOEF *blk = MCU_data[b][0];
            REFILL();
            if (PEEK(1)) blk[0] = (JCOEF)(blk[0] | p1);
            CONSUME(1);
        }
        BR_SAVE(X);
    }
    if (nlj_mcu_end(X, cinfo)) goto fail;
    return TRUE;
fail:
    X->abort = 1;
    return FALSE;
}

static boolean nlj_ac_first(j_decompress_ptr cinfo, JBLOCKROW *MCU_data) {
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort) return FALSE;
    if (cinfo->restart_interval && X->rtg == 0 && nlj_restart(X, cinfo)) goto fail;
    if (X->eobrun > 0) {
        X->eobrun--;
    } else {
        BR_LOAD(X);
        int Se = cinfo->Se, Al = cinfo->Al;
        JCOEF *blk = MCU_data[0][0];
        const jh_tab *at = X->act[0];
        int eobrun = 0;
        for (int k = cinfo->Ss; k <= Se; k++) {
            REFILL();
            int r, s, v, hv;
            DECODE_SYM(at, r, s, v, hv, goto fail);
            if (hv) {
                k += r;
                blk[nl_natural[k]] = (JCOEF)(int)((unsigned)v << Al);
            } else {
                if (r == 15) {
                    k += 15;
                } else {
                    eobrun = 1 << r;
                    if (r) {
                        REFILL();
                        eobrun += (int)PEEK(r);
                        CONSUME(r);
                    }
                    eobrun--;
                    break;
                }
            }
        }
        X->eobrun = eobrun;
        BR_SAVE(X);
    }
    if (nlj_mcu_end(X, cinfo)) goto fail;
    return TRUE;
fail:
    X->abort = 1;
    return FALSE;
}

/* zigzag-order nonzero mask of a block: bit k set iff blk[nl_natural[k]] != 0 */
static uint64_t nl_zz_tab[8][256];
__attribute__((constructor)) static void nl_zz_init(void) {
    for (int k = 0; k < 64; k++) {
        int n = nl_natural[k];
        for (int v = 0; v < 256; v++)
            if (v & (1 << (n & 7))) nl_zz_tab[n >> 3][v] |= 1ull << k;
    }
}
static inline uint64_t nz_zigzag(const JCOEF *blk) {
    const __m256i z = _mm256_setzero_si256();
    __m256i c0 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)blk), z);
    __m256i c1 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)(blk + 16)), z);
    __m256i c2 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)(blk + 32)), z);
    __m256i c3 = _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)(blk + 48)), z);
    __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi16(c0, c1), 0xD8);
    __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi16(c2, c3), 0xD8);
    uint64_t zero = (uint32_t)_mm256_movemask_epi8(p01) | ((uint64_t)(uint32_t)_mm256_movemask_epi8(p23) << 32);
    uint64_t nzn = ~zero; /* natural order */
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) r |= nl_zz_tab[i][(nzn >> (8 * i)) & 255];
    return r;
}
static inline uint64_t brange(int a, int b) { /* bits a..b inclusive, 0 <= a, b <= 63 */
    if (a > b) return 0;
    return (~0ull << a) & (~0ull >> (63 - b));
}

#define CORRECT1(tc)                                                           \
    do {                                                                       \
        REFILL();                                                              \
        if (PEEK(1)) {                                                         \
            if ((*(tc) & p1) == 0) {                                           \
                if (*(tc) >= 0) *(tc) = (JCOEF)(*(tc) + p1);                   \
                else *(tc) = (JCOEF)(*(tc) + m1);                              \
            }                                                                  \
        }                                                                      \
        CONSUME(1);                                                            \
    } while (0)

static boolean nlj_ac_refine(j_decompress_ptr cinfo, JBLOCKROW *MCU_data) {
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort) return FALSE;
    if (cinfo->restart_interval && X->rtg == 0 && nlj_restart(X, cinfo)) goto fail;
    {
        BR_LOAD(X);
        int Se = cinfo->Se;
        int p1 = 1 << cinfo->Al;
        int m1 = (int)(~0u << cinfo->Al);
        JCOEF *blk = MCU_data[0][0];
        const jh_tab *at = X->act[0];
        int eobrun = X->eobrun;
        int k = cinfo->Ss;
        uint64_t nz = nz_zigzag(blk);
        if (eobrun == 0) {
            for (; k <= Se; k++) {
                REFILL();
                int r, s, v, hv;
                DECODE_SYM(at, r, s, v, hv, goto fail);
                (void)hv;
                if (s) {
                    if (s != 1) goto fail; /* libjpeg warns (JWRN_HUFF_BAD_CODE) */
                    s = v > 0 ? p1 : m1;
                } else {
                    if (r != 15) {
                        eobrun = 1 << r;
                        if (r) {
                            REFILL();
                            eobrun += (int)PEEK(r);
                            CONSUME(r);
                        }
                        break;
                    }
                }
                /* advance over nonzero coefs (correction bits) and r zero coefs: stop at
                 * the (r+1)-th zero in [k, Se], or at Se+1 */
                uint64_t zm = ~nz & brange(k, Se);
                int tgt = Se + 1;
                if (r < __builtin_popcountll(zm)) tgt = __builtin_ctzll(_pdep_u64(1ull << r, zm));
                uint64_t cm = nz & brange(k, tgt - 1);
                while (cm) {
                    int pos = __builtin_ctzll(cm);
                    cm &= cm - 1;
                    CORRECT1(blk + nl_natural[pos]);
                }
                k = tgt;
                if (s) {
                    blk[nl_natural[k]] = (JCOEF)s;
                    if (k < 64) nz |= 1ull << k;
                }
            }
        }
        if (eobrun > 0) {
            if (k <= Se) {
                uint64_t cm = nz & brange(k, Se);
                while (cm) {
                    int pos = __builtin_ctzll(cm);
                    cm &= cm - 1;
                    CORRECT1(blk + nl_natural[pos]);
                }
            }
            eobrun--;
        }
        X->eobrun = eobrun;
        BR_SAVE(X);
    }
    if (nlj_mcu_end(X, cinfo)) goto fail;
    return TRUE;
fail:
    X->abort = 1;
    return FALSE;
}

static const jh_tab *nlj_table(nlj_ctx *X, j_decompress_ptr cinfo, int isDC, int no) {
    if (no < 0 || no > 3) return NULL;
    const JHUFF_TBL *h = isDC ? cinfo->dc_huff_tbl_ptrs[no] : cinfo->ac_huff_tbl_ptrs[no];
    if (!h) return NULL;
    int ti = (isDC ? 0 : 4) + no;
    jh_tab *t = &X->tabs[ti];
    if (X->tab_ok[ti] && t->src == h && !memcmp(t->bits, h->bits, 17) && !memcmp(t->vals, h->huffval, 256))
        return t;
    X->tab_ok[ti] = 0;
    if (jh_build(t, h, isDC)) return NULL;
    X->tab_ok[ti] = 1;
    return t;
}

/* Destuff the scan's entropy-coded data starting at q.  0 ok, -1 unusual. */
static int nlj_destuff(nlj_ctx *X, const uint8_t *q) {
    const uint8_t *end = X->buf + X->len;
    size_t need = (size_t)(end - q) + 8192;
    if (need > X->dcap) {
        free(X->d);
        X->d = (uint8_t *)malloc(need);
        if (!X->d) { X->dcap = 0; return -1; }
        X->dcap = need;
    }
    uint8_t *d = X->d;
    size_t dn = 0;
    X->nivl = 0;
    for (;;) {
        const uint8_t *f = (const uint8_t *)memchr(q, 0xFF, (size_t)(end - q));
        if (!f) return -1; /* no terminating marker */
        size_t run = (size_t)(f - q);
        memcpy(d + dn, q, run);
        dn += run;
        if (f + 1 >= end) return -1;
        uint8_t m = f[1];
        if (m == 0x00) {
            d[dn++] = 0xFF;
            q = f + 2;
            continue;
        }
        if (m == 0xFF) return -1; /* fill bytes: leave to libjpeg */
        if (X->nivl + 1 >= X->ivlcap) {
            size_t nc = X->ivlcap * 2 + 64;
            size_t *a = (size_t *)realloc(X->ivl_end, nc * sizeof(size_t));
            if (!a) return -1;
            X->ivl_end = a;
            int *b = (int *)realloc(X->ivl_rst, nc * sizeof(int));
            if (!b) return -1;
            X->ivl_rst = b;
            X->ivlcap = nc;
        }
        X->ivl_end[X->nivl] = dn;
        if (m >= 0xD0 && m <= 0xD7) {
            X->ivl_rst[X->nivl++] = m - 0xD0;
            q = f + 2;
            continue;
        }
        X->ivl_rst[X->nivl++] = -1;
        X->seg_end = f;
        break;
    }
    memset(d + dn, 0, 8192 > X->dcap - dn ? X->dcap - dn : 8192);
    X->dn = dn;
    return 0;
}

static void nlj_progress(j_common_ptr c) {
    j_decompress_ptr cinfo = (j_decompress_ptr)c;
    nlj_ctx *X = (nlj_ctx *)cinfo->client_data;
    if (X->abort || X->disabled) return;
    struct jpeg_entropy_decoder *ent = cinfo->entropy;
    if (!ent || !ent->decode_mcu) return;
    if (ent->decode_mcu == nlj_seq || ent->decode_mcu == nlj_dc_first || ent->decode_mcu == nlj_dc_refine ||
        ent->decode_mcu == nlj_ac_first || ent->decode_mcu == nlj_ac_refine)
        return; /* scan already hooked */
    if (cinfo->input_iMCU_row != 0 || cinfo->arith_code) { X->disabled = 1; return; }
    /* new scan, nothing decoded yet */
    boolean (*fn)(j_decompress_ptr, JBLOCKROW *);
    int nb = cinfo->blocks_in_MCU;
    if (nb <= 0 || nb > D_MAX_BLOCKS_IN_MCU) { X->disabled = 1; return; }
    if (!cinfo->progressive_mode) {
        if (cinfo->Ss != 0 || cinfo->Se != 63 || cinfo->Ah != 0 || cinfo->Al != 0) { X->disabled = 1; return; }
        fn = nlj_seq;
    } else if (cinfo->Ss == 0) {
        fn = cinfo->Ah == 0 ? nlj_dc_first : nlj_dc_refine;
    } else {
        if (cinfo->comps_in_scan != 1) { X->disabled = 1; return; }
        fn = cinfo->Ah == 0 ? nlj_ac_first : nlj_ac_refine;
    }
    for (int b = 0; b < nb; b++) {
        int ci = cinfo->MCU_membership[b];
        if (ci < 0 || ci >= cinfo->comps_in_scan || ci > 3) { X->disabled = 1; return; }
        jpeg_component_info *cp = cinfo->cur_comp_info[ci];
        X->memb[b] = ci;
        X->dct[b] = X->act[b] = NULL;
        if (fn == nlj_seq || fn == nlj_dc_first) {
            X->dct[b] = nlj_table(X, cinfo, 1, cp->dc_tbl_no);
            if (!X->dct[b]) { X->disabled = 1; return; }
        }
        if (fn == nlj_seq || fn == nlj_ac_first || fn == nlj_ac_refine) {
            X->act[b] = nlj_table(X, cinfo, 0, cp->ac_tbl_no);
            if (!X->act[b]) { X->disabled = 1; return; }
        }
    }
    if (nlj_destuff(X, cinfo->src->next_input_byte)) { X->disabled = 1; return; }
    X->ivl = 0;
    X->p = X->d;
    X->acc = 0;
    X->cnt = 0;
    for (int i = 0; i < 4; i++) X->last_dc[i] = 0;
    X->eobrun = 0;
    X->rtg = cinfo->restart_interval;
    X->rst_expect = 0;
    X->mcu_done = 0;
    X->mcu_total = (long)cinfo->MCUs_per_row * (long)cinfo->MCU_rows_in_scan;
    X->installed_scans++;
    ent->decode_mcu = fn;
}
