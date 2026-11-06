/* nailong-hpc module A: image decoding.
 *
 * nl_decode() reproduces, byte for byte, what the Python reference does in
 * nailong.features._load up to and including the convert() call:
 *
 *   im = Image.open(BytesIO(buf))
 *   if im.format == "JPEG": im.draft("RGB", (256, 256))
 *   has_alpha = im.mode in ("RGBA","LA","PA") or (im.mode == "P" and "transparency" in im.info)
 *   im = im.convert("RGBA" if has_alpha else "RGB")
 *
 * (Pillow 13.0.0.)  The rule throughout: whenever Pillow's behaviour for an input is
 * not reproduced exactly (odd chunk layouts, metadata that makes Image.open fail,
 * unusual modes, ...), return NL_UNSUPPORTED so the caller falls back to Python.
 * NL_ERROR is only returned where the reference is known to raise.
 *
 * Formats:
 *   JPEG  libjpeg-turbo, same settings as Pillow's JpegDecode.c (ISLOW, fancy
 *         upsampling, DCT scaling chosen exactly as JpegImageFile.draft), and the
 *         input is fed in the same 64 KiB suspension increments as ImageFile.load.
 *         MPO (format "MPO", first frame, no draft) is detected as jpeg_factory does.
 *   PNG   own chunk walker mirroring PngImagePlugin (CRC checks, tRNS/PLTE rules,
 *         post-IDAT chunk handling), libdeflate one-shot inflate, SIMD unfilter.
 *   GIF   own first-frame LZW decoder with the exact semantics of Pillow's
 *         GifDecode.c (including its sub-block / end-code quirks).
 *   WEBP  libwebp WebPAnimDecoder, as Pillow's _webp.c.
 */
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>
#include <jpeglib.h>
#include <libdeflate.h>
#include <webp/decode.h>
#include <webp/demux.h>

#include "../include/nl.h"

/* PIL.Image.MAX_IMAGE_PIXELS; beyond it Pillow warns (and at 2x raises). */
#define NL_MAX_PIXELS 89478485ULL

static inline uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static inline uint32_t be16(const uint8_t *p) { return ((uint32_t)p[0] << 8) | p[1]; }
static inline uint32_t le16(const uint8_t *p) { return ((uint32_t)p[1] << 8) | p[0]; }
static inline uint32_t le32(const uint8_t *p) {
    return ((uint32_t)p[3] << 24) | ((uint32_t)p[2] << 16) | ((uint32_t)p[1] << 8) | p[0];
}

/* statistics (testing aid, not part of nl.h) */
static long nl_stat_fast, nl_stat_fallback;
void nl_decode_stats(long *jpeg_fast, long *jpeg_fallback) {
    *jpeg_fast = nl_stat_fast;
    *jpeg_fallback = nl_stat_fallback;
}

/* NL_JPEG_NOFAST=1 disables the fast JPEG entropy path (testing aid). */
static int getenv_nofast(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("NL_JPEG_NOFAST");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static int too_big(uint64_t w, uint64_t h) { return w == 0 || h == 0 || w * h > NL_MAX_PIXELS; }

/* ======================================================================== */
/* JPEG                                                                     */
/* ======================================================================== */

typedef struct {
    int w, h, layers;
    int have_sof, have_sos;
    int mpo;          /* 1: Pillow returns an MpoImageFile (no draft) */
} jpeg_hdr;

/* Lightweight emulation of JpegImagePlugin._getmp + jpeg_factory's decision.
 * Returns 0 = plain JPEG, 1 = MPO, -1 = unsure (caller -> NL_UNSUPPORTED). */
static int jpeg_mp_decision(const uint8_t *mp, size_t n, int hdrgm) {
    if (n < 8) return hdrgm ? 0 : -1;
    int big;
    if (mp[0] == 'M' && mp[1] == 'M' && mp[2] == 0 && mp[3] == 42) big = 1;
    else if (mp[0] == 'I' && mp[1] == 'I' && mp[2] == 42 && mp[3] == 0) big = 0;
    else return hdrgm ? 0 : -1;
#define U16(p) (big ? be16(p) : le16(p))
#define U32(p) (big ? be32(p) : le32(p))
    uint32_t ifd = U32(mp + 4);
    if ((size_t)ifd + 2 > n) return hdrgm ? 0 : -1;
    uint32_t cnt = U16(mp + ifd);
    if ((size_t)ifd + 2 + 12ull * cnt + 4 > n) return hdrgm ? 0 : -1;
    int nb001 = 0, nb002 = 0;
    uint32_t quant = 0;
    uint32_t e_typ = 0, e_cnt = 0, e_off = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        const uint8_t *e = mp + ifd + 2 + 12 * i;
        uint32_t tag = U16(e), typ = U16(e + 2), c = U32(e + 4);
        if (tag == 0xB001) {
            nb001++;
            if (typ == 4 && c == 1) quant = U32(e + 8);
            else if (typ == 3 && c == 1) quant = U16(e + 8);
            else nb001 += 100; /* unusual: give up */
        } else if (tag == 0xB002) {
            nb002++;
            e_typ = typ;
            e_cnt = c;
            e_off = (c > 4) ? U32(e + 8) : (uint32_t)(e + 8 - mp);
        }
    }
    if (hdrgm) return 0; /* Ultra HDR: jpeg_factory keeps it a JPEG whatever _getmp does */
    if (nb001 != 1) return -1;
    if (quant <= 1) return 0;
    if (nb002 != 1 || e_typ != 7) return -1;
    if (e_cnt < 16ull * quant || (size_t)e_off + e_cnt > n) return -1;
    for (uint32_t k = 0; k < quant; k++) {
        uint32_t attr = U32(mp + e_off + 16 * k);
        if ((attr >> 24) & 7) return 0; /* SyntaxError in _getmp -> JPEG (with a warning) */
    }
    return 1;
#undef U16
#undef U32
}

static int memfind(const uint8_t *h, size_t hn, const char *needle) {
    size_t nn = strlen(needle);
    if (hn < nn) return 0;
    for (size_t i = 0; i + nn <= hn; i++)
        if (h[i] == (uint8_t)needle[0] && !memcmp(h + i, needle, nn)) return 1;
    return 0;
}

/* Emulates JpegImageFile._open's marker loop.  Returns NL_OK or NL_UNSUPPORTED. */
static int jpeg_parse_pil(const uint8_t *b, size_t n, jpeg_hdr *jh) {
    memset(jh, 0, sizeof *jh);
    size_t pos = 3; /* read(3) */
    int s0 = 0xFF;  /* s = b"\xff" */
    int have_s = 1;
    const uint8_t *mp = NULL;
    size_t mpn = 0;
    int hdrgm = 0;
    int icc_short = 0, icc_n = 0;
    while (have_s) {
        int marker;
        if (s0 == 0xFF) {
            if (pos >= n) break;
            marker = 0xFF00 | b[pos++];
        } else {
            if (pos >= n) { have_s = 0; break; }
            s0 = b[pos++];
            continue;
        }
        int lo = marker & 0xFF;
        if (lo >= 0xC0 && lo <= 0xFE) {
            int has_handler = !(lo == 0xC8 || (lo >= 0xD0 && lo <= 0xD9) || (lo >= 0xF0 && lo <= 0xFD));
            if (has_handler) {
                if (pos + 2 > n) return NL_UNSUPPORTED;
                long L = (long)be16(b + pos) - 2;
                pos += 2;
                if (L < 0) return NL_UNSUPPORTED;
                if (pos + (size_t)L > n) return NL_UNSUPPORTED; /* _safe_read OSError */
                const uint8_t *s = b + pos;
                pos += (size_t)L;
                int is_sof = (lo >= 0xC0 && lo <= 0xCF && lo != 0xC4 && lo != 0xC8 && lo != 0xCC) || lo == 0xDE;
                if (lo >= 0xE0 && lo <= 0xEF) {
                    if (lo == 0xEE && L >= 5 && !memcmp(s, "Adobe", 5) && L < 7) return NL_UNSUPPORTED;
                    if (lo == 0xE2 && L >= 4 && !memcmp(s, "MPF\0", 4)) { mp = s + 4; mpn = (size_t)L - 4; }
                    if (lo == 0xE2 && L >= 12 && !memcmp(s, "ICC_PROFILE\0", 12)) {
                        icc_n++;
                        if (L < 14) icc_short = 1;
                    }
                    if (lo == 0xE1 && memfind(s, (size_t)L, " hdrgm:Version=\"")) hdrgm = 1;
                } else if (is_sof) {
                    if (L < 6) return NL_UNSUPPORTED;
                    if (s[0] != 8) return NL_UNSUPPORTED;
                    jh->h = (int)be16(s + 1);
                    jh->w = (int)be16(s + 3);
                    jh->layers = s[5];
                    if (jh->layers != 1 && jh->layers != 3) return NL_UNSUPPORTED; /* CMYK etc. */
                    if (icc_n && icc_short) return NL_UNSUPPORTED;
                    icc_n = 0;
                    icc_short = 0;
                    if ((L - 6) % 3) return NL_UNSUPPORTED;
                    jh->have_sof = 1;
                } else if (lo == 0xDB) {
                    const uint8_t *q = s;
                    long r = L;
                    while (r > 0) {
                        long ql = 1 + ((q[0] >> 4) == 0 ? 1 : 2) * 64;
                        if (r < ql) return NL_UNSUPPORTED;
                        q += ql;
                        r -= ql;
                    }
                } else if (lo == 0xDA) {
                    jh->have_sos = 1;
                    break;
                }
            }
            if (pos >= n) { have_s = 0; break; }
            s0 = b[pos++];
        } else if (marker == 0xFFFF) {
            s0 = 0xFF;
        } else if (marker == 0xFF00) {
            if (pos >= n) { have_s = 0; break; }
            s0 = b[pos++];
        } else {
            return NL_UNSUPPORTED; /* "no marker found" */
        }
    }
    if (!jh->have_sof || !jh->have_sos) return NL_UNSUPPORTED;
    if (too_big((uint64_t)jh->w, (uint64_t)jh->h)) return NL_UNSUPPORTED;
    if (mp) {
        int d = jpeg_mp_decision(mp, mpn, hdrgm);
        if (d < 0) return NL_UNSUPPORTED;
        jh->mpo = d;
    }
    return NL_OK;
}

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
} nl_jerr;

static void nl_jerr_exit(j_common_ptr c) { longjmp(((nl_jerr *)c->err)->jb, 1); }
static void nl_jerr_output(j_common_ptr c) { (void)c; }

typedef struct {
    struct jpeg_source_mgr pub;
    long skip;
} nl_jsrc;

static void js_stub(j_decompress_ptr c) { (void)c; }
static boolean js_fill(j_decompress_ptr c) { (void)c; return FALSE; } /* suspend, like Pillow */
static void js_skip(j_decompress_ptr c, long num) {
    nl_jsrc *s = (nl_jsrc *)c->src;
    if (num > (long)s->pub.bytes_in_buffer) {
        s->skip = num - (long)s->pub.bytes_in_buffer;
        s->pub.next_input_byte += s->pub.bytes_in_buffer;
        s->pub.bytes_in_buffer = 0;
    } else {
        s->pub.bytes_in_buffer -= num;
        s->pub.next_input_byte += num;
        s->skip = 0;
    }
}

#define PIL_MAXBLOCK 65536

#include "jhuff.h"

/* ---- fast path: whole buffer as a non-suspending source + own Huffman decoder ---- */
typedef struct {
    struct jpeg_source_mgr pub;
    int eof;
} nl_fsrc;

static const JOCTET nl_fake_eoi[2] = {0xFF, JPEG_EOI};
static boolean fs_fill(j_decompress_ptr c) {
    nl_fsrc *s = (nl_fsrc *)c->src;
    s->eof = 1;
    s->pub.next_input_byte = nl_fake_eoi;
    s->pub.bytes_in_buffer = 2;
    return TRUE;
}
static void fs_skip(j_decompress_ptr c, long num) {
    nl_fsrc *s = (nl_fsrc *)c->src;
    if (num <= 0) return;
    if ((size_t)num > s->pub.bytes_in_buffer) {
        s->eof = 1;
        s->pub.next_input_byte += s->pub.bytes_in_buffer;
        s->pub.bytes_in_buffer = 0;
    } else {
        s->pub.next_input_byte += num;
        s->pub.bytes_in_buffer -= (size_t)num;
    }
}

/* NL_OK, or -1: redo with the exact (Pillow-chunked, plain libjpeg) path */
static int jpeg_fast(const uint8_t *buf, size_t len, const jpeg_hdr *jh, int s, nl_img *out) {
    int ow = (jh->w + s - 1) / s, oh = (jh->h + s - 1) / s;
    struct jpeg_decompress_struct cinfo;
    nl_jerr jerr;
    nl_fsrc src;
    struct jpeg_progress_mgr prog;
    nlj_ctx *X = (nlj_ctx *)calloc(1, sizeof(nlj_ctx));
    if (!X) return -1;
    uint8_t *volatile px = NULL;
    volatile int created = 0;
    X->buf = buf;
    X->len = len;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = nl_jerr_exit;
    jerr.pub.output_message = nl_jerr_output;
    if (setjmp(jerr.jb)) {
        if (created) jpeg_destroy_decompress(&cinfo);
        free(px);
        free(X->d); free(X->ivl_end); free(X->ivl_rst); free(X);
        return -1;
    }
    jpeg_create_decompress(&cinfo);
    created = 1;
    cinfo.client_data = X;
    memset(&src, 0, sizeof src);
    src.pub.init_source = js_stub;
    src.pub.fill_input_buffer = fs_fill;
    src.pub.skip_input_data = fs_skip;
    src.pub.resync_to_restart = jpeg_resync_to_restart;
    src.pub.term_source = js_stub;
    src.pub.next_input_byte = buf;
    src.pub.bytes_in_buffer = len;
    cinfo.src = &src.pub;
    memset(&prog, 0, sizeof prog);
    prog.progress_monitor = nlj_progress;
    cinfo.progress = &prog;

    int ok;
    do {
        ok = jpeg_read_header(&cinfo, FALSE);
    } while (ok == JPEG_HEADER_TABLES_ONLY && !src.eof);
    int fail = (ok != JPEG_HEADER_OK) || src.eof;
    if (!fail && ((int)cinfo.image_width != jh->w || (int)cinfo.image_height != jh->h ||
                  cinfo.num_components != jh->layers))
        fail = 1;
    if (!fail) {
        cinfo.out_color_space = JCS_EXT_RGB;
        if (s > 1) {
            cinfo.scale_num = 1;
            cinfo.scale_denom = (unsigned)s;
        }
        if (!jpeg_start_decompress(&cinfo) || (int)cinfo.output_width != ow || (int)cinfo.output_height != oh ||
            cinfo.output_components != 3)
            fail = 1;
    }
    if (!fail) {
        px = (uint8_t *)malloc((size_t)ow * oh * 3);
        if (!px) fail = 1;
    }
    if (!fail) {
        JSAMPROW rows[16];
        int y = 0;
        while (y < oh) {
            int k = 0;
            for (; k < 16 && y + k < oh; k++) rows[k] = px + (size_t)(y + k) * ow * 3;
            JDIMENSION got = jpeg_read_scanlines(&cinfo, rows, (JDIMENSION)k);
            if (got == 0) { fail = 1; break; }
            y += (int)got;
        }
    }
    if (!fail) {
        if (!jpeg_finish_decompress(&cinfo)) fail = 1;
    }
    if (X->abort || src.eof || jerr.pub.num_warnings) fail = 1;
    jpeg_destroy_decompress(&cinfo);
    free(X->d); free(X->ivl_end); free(X->ivl_rst); free(X);
    if (fail) { free(px); return -1; }
    out->w = ow;
    out->h = oh;
    out->ch = 3;
    out->px = px;
    return NL_OK;
}

static int jpeg_exact(const uint8_t *buf, size_t len, const jpeg_hdr *jhp, int s, nl_img *out);

static int decode_jpeg(const uint8_t *buf, size_t len, nl_img *out) {
    jpeg_hdr jh;
    int r = jpeg_parse_pil(buf, len, &jh);
    if (r != NL_OK) return r;
    int s = 1;
    if (!jh.mpo) { /* JpegImageFile.draft("RGB", (256, 256)) */
        int sc = jh.w / 256 < jh.h / 256 ? jh.w / 256 : jh.h / 256;
        s = sc >= 8 ? 8 : sc >= 4 ? 4 : sc >= 2 ? 2 : 1;
    }
    if (!getenv_nofast()) {
        if (jpeg_fast(buf, len, &jh, s, out) == NL_OK) { __atomic_add_fetch(&nl_stat_fast, 1, __ATOMIC_RELAXED); return NL_OK; }
        __atomic_add_fetch(&nl_stat_fallback, 1, __ATOMIC_RELAXED);
    }
    return jpeg_exact(buf, len, &jh, s, out);
}

/* Exact emulation of Pillow's JpegDecode.c + ImageFile.load feeding (64 KiB reads,
 * suspending source), with libjpeg-turbo's own entropy decoder. */
static int jpeg_exact(const uint8_t *buf, size_t len, const jpeg_hdr *jhp, int s, nl_img *out) {
    jpeg_hdr jh = *jhp;
    int ow = (jh.w + s - 1) / s, oh = (jh.h + s - 1) / s;

    struct jpeg_decompress_struct cinfo;
    nl_jerr jerr;
    nl_jsrc src;
    uint8_t *volatile px = NULL;
    volatile int created = 0;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = nl_jerr_exit;
    jerr.pub.output_message = nl_jerr_output;
    if (setjmp(jerr.jb)) {
        if (created) jpeg_destroy_decompress(&cinfo);
        free(px);
        return NL_ERROR; /* Pillow: IMAGING_CODEC_BROKEN -> OSError */
    }
    jpeg_create_decompress(&cinfo);
    created = 1;
    memset(&src, 0, sizeof src);
    src.pub.init_source = js_stub;
    src.pub.fill_input_buffer = js_fill;
    src.pub.skip_input_data = js_skip;
    src.pub.resync_to_restart = jpeg_resync_to_restart;
    src.pub.term_source = js_stub;
    src.pub.bytes_in_buffer = 0;
    src.skip = 0;
    cinfo.src = &src.pub;

    /* Emulate ImageFile.load: read 64 KiB at a time, decoder sees unconsumed+new. */
    size_t read_end = 0;
    int state = 1;
    int y = 0;
    for (;;) {
        /* read(MAXBLOCK) */
        if (read_end >= len) { /* read() returned b"": "image file is truncated" */
            jpeg_destroy_decompress(&cinfo);
            free(px);
            return NL_ERROR;
        }
        size_t consumed = (size_t)(src.pub.next_input_byte ? (src.pub.next_input_byte - buf) : 0);
        read_end = read_end + PIL_MAXBLOCK < len ? read_end + PIL_MAXBLOCK : len;
        src.pub.next_input_byte = buf + consumed;
        src.pub.bytes_in_buffer = read_end - consumed;
        if (src.skip > 0) {
            js_skip(&cinfo, src.skip);
            if (src.skip > 0) continue;
        }
        if (state == 1) {
            int ok;
            do {
                ok = jpeg_read_header(&cinfo, FALSE);
            } while (ok == JPEG_HEADER_TABLES_ONLY);
            if (ok == JPEG_SUSPENDED) continue;
            if ((int)cinfo.image_width != jh.w || (int)cinfo.image_height != jh.h ||
                cinfo.num_components != jh.layers) {
                jpeg_destroy_decompress(&cinfo);
                return NL_UNSUPPORTED;
            }
            /* jpegmode "" -> trust the decoder's colour space; rawmode RGB(X)/L */
            cinfo.out_color_space = JCS_EXT_RGB; /* L is expanded by libjpeg like l2rgb */
            if (s > 1) {
                cinfo.scale_num = 1;
                cinfo.scale_denom = (unsigned)s;
            }
            state = 2;
        }
        if (state == 2) {
            if (!jpeg_start_decompress(&cinfo)) continue;
            if ((int)cinfo.output_width != ow || (int)cinfo.output_height != oh ||
                cinfo.output_components != 3) {
                jpeg_destroy_decompress(&cinfo);
                return NL_UNSUPPORTED;
            }
            px = (uint8_t *)malloc((size_t)ow * oh * 3);
            if (!px) {
                jpeg_destroy_decompress(&cinfo);
                return NL_UNSUPPORTED;
            }
            state = 3;
        }
        if (state == 3) {
            JSAMPROW rows[16];
            while (y < oh) {
                int k = 0;
                for (; k < 16 && y + k < oh; k++) rows[k] = px + (size_t)(y + k) * ow * 3;
                JDIMENSION got = jpeg_read_scanlines(&cinfo, rows, (JDIMENSION)k);
                if (got == 0) break;
                y += (int)got;
            }
            if (y < oh) continue;
            state = 4;
        }
        if (state == 4) {
            (void)jpeg_finish_decompress(&cinfo); /* suspension here is fine (y == ysize) */
            jpeg_destroy_decompress(&cinfo);
            break;
        }
    }
    out->w = ow;
    out->h = oh;
    out->ch = 3;
    out->px = px;
    return NL_OK;
}

/* ======================================================================== */
/* PNG                                                                      */
/* ======================================================================== */

static __thread struct libdeflate_decompressor *tl_dec;

static struct libdeflate_decompressor *get_dec(void) {
    if (!tl_dec) tl_dec = libdeflate_alloc_decompressor();
    return tl_dec;
}

/* Optional: free this thread's cached libdeflate decompressor (call at worker exit). */
void nl_decode_thread_cleanup(void) {
    if (tl_dec) libdeflate_free_decompressor(tl_dec);
    tl_dec = NULL;
}

static int is_cid(const uint8_t *c) {
    for (int i = 0; i < 4; i++) {
        uint8_t x = c[i];
        if (!((x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z') || (x >= '0' && x <= '9') || x == '_'))
            return 0;
    }
    return 1;
}

/* _safe_zlib_decompress: raises ValueError if the plaintext exceeds MAX_TEXT_CHUNK (1 MiB).
 * Returns 0 if certainly fine, -1 if it might raise (caller -> unsupported). */
static int png_text_zlib_ok(const uint8_t *s, size_t n) {
    if (n * 1032 + 64 < (1u << 20)) return 0; /* deflate cannot expand > 1032:1 */
    size_t cap = (1u << 20);
    uint8_t *tmp = (uint8_t *)malloc(cap);
    if (!tmp) return -1;
    size_t ain = 0, aout = 0;
    enum libdeflate_result rr = libdeflate_zlib_decompress_ex(get_dec(), s, n, tmp, cap, &ain, &aout);
    free(tmp);
    if (rr == LIBDEFLATE_SUCCESS && aout < cap - 65536) return 0;
    return -1;
}

static const char *const KNOWN_CHUNKS[] = {"iCCP", "IHDR", "IDAT", "IEND", "PLTE", "tRNS", "gAMA", "cHRM",
                                           "sRGB", "pHYs", "tEXt", "zTXt", "iTXt", "eXIf", "acTL", "fcTL",
                                           "fdAT", NULL};

static int chunk_is(const uint8_t *cid, const char *name) { return !memcmp(cid, name, 4); }
static int chunk_known(const uint8_t *cid) {
    for (int i = 0; KNOWN_CHUNKS[i]; i++)
        if (chunk_is(cid, KNOWN_CHUNKS[i])) return 1;
    return 0;
}

/* ---- unfiltering (in place, cur row has stride bytes, prev row or NULL) ---- */

static void unf_sub(uint8_t *cur, size_t n, int bpp) {
    if (bpp == 4) {
        uint32_t a = 0;
        __m128i va = _mm_setzero_si128();
        for (size_t i = 0; i < n; i += 4) {
            uint32_t v;
            memcpy(&v, cur + i, 4);
            va = _mm_add_epi8(va, _mm_cvtsi32_si128((int)v));
            a = (uint32_t)_mm_cvtsi128_si32(va);
            memcpy(cur + i, &a, 4);
        }
    } else if (bpp == 3) {
        uint8_t a0 = 0, a1 = 0, a2 = 0;
        for (size_t i = 0; i < n; i += 3) {
            a0 = cur[i] += a0;
            a1 = cur[i + 1] += a1;
            a2 = cur[i + 2] += a2;
        }
    } else {
        for (size_t i = (size_t)bpp; i < n; i++) cur[i] += cur[i - bpp];
    }
}

static void unf_up(uint8_t *cur, const uint8_t *prev, size_t n) {
    if (!prev) return;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(cur + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(prev + i));
        _mm256_storeu_si256((__m256i *)(cur + i), _mm256_add_epi8(a, b));
    }
    for (; i < n; i++) cur[i] += prev[i];
}

static void unf_avg(uint8_t *cur, const uint8_t *prev, size_t n, int bpp) {
    if (!prev) { /* previous line is black */
        for (size_t i = (size_t)bpp; i < n; i++) cur[i] += cur[i - bpp] >> 1;
        return;
    }
    if (bpp == 4 || bpp == 3) {
        __m128i a = _mm_setzero_si128();
        const __m128i one = _mm_set1_epi8(1);
        size_t i = 0;
        for (; i + 4 <= n; i += (size_t)bpp) {
            uint32_t xv, bv;
            memcpy(&xv, cur + i, 4);
            memcpy(&bv, prev + i, 4);
            __m128i x = _mm_cvtsi32_si128((int)xv), b = _mm_cvtsi32_si128((int)bv);
            /* floor((a+b)/2) = avg_round_up - ((a^b)&1) */
            __m128i av = _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one));
            a = _mm_add_epi8(x, av);
            uint32_t o = (uint32_t)_mm_cvtsi128_si32(a);
            memcpy(cur + i, &o, (size_t)bpp);
            if (bpp == 3) a = _mm_and_si128(a, _mm_cvtsi32_si128(0x00FFFFFF));
        }
        for (; i < n; i++) cur[i] += (uint8_t)(((i >= (size_t)bpp ? cur[i - bpp] : 0) + prev[i]) >> 1);
        return;
    }
    size_t i = 0;
    for (; i < (size_t)bpp && i < n; i++) cur[i] += prev[i] >> 1;
    for (; i < n; i++) cur[i] += (uint8_t)((cur[i - bpp] + prev[i]) >> 1);
}

static inline uint8_t paeth1(int a, int b, int c) {
    int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)((pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c);
}

static void unf_paeth(uint8_t *cur, const uint8_t *prev, size_t n, int bpp) {
    if (!prev) { /* b = c = 0: predictor = a  -> same as sub */
        unf_sub(cur, n, bpp);
        return;
    }
    if (bpp == 4 || bpp == 3) {
        const __m128i z = _mm_setzero_si128();
        __m128i a = z, c = z; /* 16-bit lanes */
        size_t i = 0;
        for (; i + 4 <= n; i += (size_t)bpp) {
            uint32_t xv, bv;
            memcpy(&xv, cur + i, 4);
            memcpy(&bv, prev + i, 4);
            __m128i b = _mm_unpacklo_epi8(_mm_cvtsi32_si128((int)bv), z);
            __m128i x = _mm_unpacklo_epi8(_mm_cvtsi32_si128((int)xv), z);
            __m128i pa = _mm_sub_epi16(b, c); /* p - a = b - c */
            __m128i pb = _mm_sub_epi16(a, c); /* p - b = a - c */
            __m128i pc = _mm_add_epi16(pa, pb);
            pa = _mm_abs_epi16(pa);
            pb = _mm_abs_epi16(pb);
            pc = _mm_abs_epi16(pc);
            __m128i sm = _mm_min_epi16(pc, _mm_min_epi16(pa, pb));
            __m128i usea = _mm_cmpeq_epi16(pa, sm);
            __m128i useb = _mm_cmpeq_epi16(pb, sm);
            __m128i pred = _mm_blendv_epi8(c, b, useb);
            pred = _mm_blendv_epi8(pred, a, usea);
            /* byte-wise add: high bytes of x and pred are 0, so lanes stay < 256 */
            __m128i res = _mm_add_epi8(x, pred);
            uint32_t o = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(res, res));
            memcpy(cur + i, &o, (size_t)bpp);
            a = res;
            c = b;
            if (bpp == 3) {
                const __m128i m3 = _mm_set_epi16(0, 0, 0, 0, 0, -1, -1, -1);
                a = _mm_and_si128(a, m3);
                c = _mm_and_si128(c, m3);
            }
        }
        for (; i < n; i++) {
            int aa = i >= (size_t)bpp ? cur[i - bpp] : 0;
            int cc = i >= (size_t)bpp ? prev[i - bpp] : 0;
            cur[i] += paeth1(aa, prev[i], cc);
        }
        return;
    }
    size_t i = 0;
    for (; i < (size_t)bpp && i < n; i++) cur[i] += prev[i];
    for (; i < n; i++) cur[i] += paeth1(cur[i - bpp], prev[i], prev[i - bpp]);
}

/* returns 0 ok, -1 bad filter type */
static int unfilter_row(uint8_t f, uint8_t *cur, const uint8_t *prev, size_t n, int bpp) {
    switch (f) {
    case 0: return 0;
    case 1: unf_sub(cur, n, bpp); return 0;
    case 2: unf_up(cur, prev, n); return 0;
    case 3: unf_avg(cur, prev, n, bpp); return 0;
    case 4: unf_paeth(cur, prev, n, bpp); return 0;
    default: return -1;
    }
}

/* PNG pixel layouts we support (after Pillow's unpacker + convert). */
enum { PK_RGB8, PK_RGBA8, PK_RGB16, PK_RGBA16, PK_LA16, PK_LA8, PK_L8, PK_L4, PK_L2, PK_1, PK_P };

typedef struct {
    int pk, bitspp, ch;      /* packing, bits per pixel in file, output channels */
    int depth;               /* sample bit depth (for P / L sub-byte) */
    uint32_t pal[256];       /* P: RGBA table (little-endian bytes R,G,B,A) */
} png_conv;

/* convert one unfiltered row (raw) to output pixels */
static void png_conv_row(const png_conv *cv, const uint8_t *raw, uint8_t *o, int w) {
    switch (cv->pk) {
    case PK_RGB8: memcpy(o, raw, (size_t)w * 3); break;
    case PK_RGBA8: memcpy(o, raw, (size_t)w * 4); break;
    case PK_RGB16:
        for (int x = 0; x < w; x++) { o[3 * x] = raw[6 * x]; o[3 * x + 1] = raw[6 * x + 2]; o[3 * x + 2] = raw[6 * x + 4]; }
        break;
    case PK_RGBA16:
        for (int x = 0; x < w; x++) {
            o[4 * x] = raw[8 * x]; o[4 * x + 1] = raw[8 * x + 2]; o[4 * x + 2] = raw[8 * x + 4]; o[4 * x + 3] = raw[8 * x + 6];
        }
        break;
    case PK_LA16:
        for (int x = 0; x < w; x++) { uint8_t v = raw[4 * x]; o[4 * x] = o[4 * x + 1] = o[4 * x + 2] = v; o[4 * x + 3] = raw[4 * x + 2]; }
        break;
    case PK_LA8:
        for (int x = 0; x < w; x++) { uint8_t v = raw[2 * x]; o[4 * x] = o[4 * x + 1] = o[4 * x + 2] = v; o[4 * x + 3] = raw[2 * x + 1]; }
        break;
    case PK_L8:
        for (int x = 0; x < w; x++) o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = raw[x];
        break;
    case PK_L4:
        for (int x = 0; x < w; x++) { uint8_t v = (uint8_t)(((raw[x >> 1] >> (4 - 4 * (x & 1))) & 15) * 0x11); o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = v; }
        break;
    case PK_L2:
        for (int x = 0; x < w; x++) { uint8_t v = (uint8_t)(((raw[x >> 2] >> (6 - 2 * (x & 3))) & 3) * 0x55); o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = v; }
        break;
    case PK_1:
        for (int x = 0; x < w; x++) { uint8_t v = ((raw[x >> 3] >> (7 - (x & 7))) & 1) ? 255 : 0; o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = v; }
        break;
    case PK_P: {
        int d = cv->depth;
        if (cv->ch == 4) {
            for (int x = 0; x < w; x++) {
                int idx = d == 8 ? raw[x] : (raw[(x * d) >> 3] >> (8 - d - ((x * d) & 7))) & ((1 << d) - 1);
                memcpy(o + 4 * x, &cv->pal[idx], 4);
            }
        } else {
            if (d == 8) {
                int x = 0;
                for (; x + 1 < w; x++) memcpy(o + 3 * x, &cv->pal[raw[x]], 4); /* writes 1 byte ahead */
                for (; x < w; x++) memcpy(o + 3 * x, &cv->pal[raw[x]], 3);
            } else {
                for (int x = 0; x < w; x++) {
                    int idx = (raw[(x * d) >> 3] >> (8 - d - ((x * d) & 7))) & ((1 << d) - 1);
                    memcpy(o + 3 * x, &cv->pal[idx], 3);
                }
            }
        }
        break;
    }
    }
}

/* put one pixel (interlaced path): raw row index x -> out position */
static void png_conv_px(const png_conv *cv, const uint8_t *raw, int x, uint8_t *o) {
    /* reuse row converter on a single pixel by extracting it */
    uint8_t tmp[8];
    int bpp = cv->bitspp;
    if (bpp >= 8) {
        memcpy(tmp, raw + (size_t)x * (bpp / 8), (size_t)(bpp / 8));
    } else {
        int bit = x * bpp;
        uint8_t byte = (uint8_t)(raw[bit >> 3] << (bit & 7));
        tmp[0] = byte;
    }
    uint8_t ob[8];
    png_conv_row(cv, tmp, ob, 1);
    memcpy(o, ob, (size_t)cv->ch);
}

static int decode_png(const uint8_t *b, size_t n, nl_img *out) {
    if (n < 8) return NL_UNSUPPORTED;
    size_t pos = 8;
    int have_ihdr = 0, ihdr_count = 0;
    uint32_t W = 0, H = 0;
    int bd = 0, ct = 0, interlace = 0;
    int mode_ok = 0;           /* _MODES lookup ever succeeded */
    int mbd = 0, mct = 0;      /* the (bitdepth, colortype) that set the mode */
    const uint8_t *plte = NULL;
    size_t plte_n = 0;
    int trns_kind = 0;         /* P: 0 none, 1 int index, 2 bytes */
    int trns_index = -1;
    const uint8_t *trns = NULL;
    size_t trns_n = 0;
    long seq_num = -1;
    int n_frames_set = 0;      /* im_n_frames is not None */
    uint32_t n_frames_val = 0;
    int have_bbox = 0;
    uint32_t bbox[4] = {0, 0, 0, 0};
    uint64_t text_mem = 0;

    /* ---- pre-IDAT chunks (PngImageFile._open) ---- */
    size_t idat_pos = 0;
    for (;;) {
        if (pos + 8 > n) return NL_UNSUPPORTED;
        uint32_t L = be32(b + pos);
        const uint8_t *cid = b + pos + 4;
        if (!is_cid(cid)) return NL_UNSUPPORTED;
        size_t dpos = pos + 8;
        if (chunk_is(cid, "IDAT")) { idat_pos = pos; break; }
        if (chunk_is(cid, "IEND")) return NL_UNSUPPORTED; /* no image data */
        if ((uint64_t)dpos + L + 4 > n) return NL_UNSUPPORTED;
        const uint8_t *s = b + dpos;
        if (chunk_is(cid, "IHDR")) {
            if (L < 13) return NL_UNSUPPORTED;
            ihdr_count++;
            W = be32(s); H = be32(s + 4);
            bd = s[8]; ct = s[9];
            static const int okc[][2] = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {8, 2}, {16, 2}, {1, 3}, {2, 3},
                                         {4, 3}, {8, 3}, {8, 4}, {16, 4}, {8, 6}, {16, 6}};
            for (size_t k = 0; k < sizeof okc / sizeof okc[0]; k++)
                if (okc[k][0] == bd && okc[k][1] == ct) { mode_ok = 1; mbd = bd; mct = ct; }
            if (s[12]) interlace = 1;
            if (s[11]) return NL_UNSUPPORTED;
            have_ihdr = 1;
        } else if (chunk_is(cid, "PLTE")) {
            if (mode_ok && mct == 3) { plte = s; plte_n = L; }
        } else if (chunk_is(cid, "tRNS")) {
            if (mode_ok && mct == 3) {
                /* _simple_palette = re.compile(b"^\xff*\x00\xff*$") ($ may precede a final \n) */
                size_t m = L;
                int simple = 0;
                for (int pass = 0; pass < 2 && !simple; pass++) {
                    if (pass == 1) { if (m && s[m - 1] == '\n') m--; else break; }
                    size_t k = 0;
                    while (k < m && s[k] == 0xFF) k++;
                    if (k < m && s[k] == 0) {
                        k++;
                        while (k < m && s[k] == 0xFF) k++;
                        if (k == m) simple = 1;
                    }
                }
                if (simple) {
                    size_t k = 0;
                    while (s[k]) k++;
                    trns_kind = 1;
                    trns_index = (int)k;
                } else {
                    trns_kind = 2;
                    trns = s;
                    trns_n = L;
                }
            } else if (mode_ok && (mct == 0 || mct == 2)) {
                if (L < (mct == 2 ? 6u : 2u)) return NL_UNSUPPORTED;
            }
        } else if (chunk_is(cid, "gAMA")) {
            if (L < 4) return NL_UNSUPPORTED;
        } else if (chunk_is(cid, "cHRM")) {
            if (L < 32) return NL_UNSUPPORTED;
        } else if (chunk_is(cid, "sRGB")) {
            if (L < 1) return NL_UNSUPPORTED;
        } else if (chunk_is(cid, "pHYs")) {
            if (L < 9) return NL_UNSUPPORTED;
        } else if (chunk_is(cid, "tEXt")) {
            text_mem += L;
        } else if (chunk_is(cid, "zTXt")) {
            const uint8_t *z = memchr(s, 0, L);
            if (z) {
                size_t vn = L - (size_t)(z + 1 - s);
                if (vn) {
                    if (z[1] != 0) return NL_UNSUPPORTED;
                    if (png_text_zlib_ok(z + 2, vn - 1)) return NL_UNSUPPORTED;
                }
            }
            text_mem += (1u << 20);
        } else if (chunk_is(cid, "iTXt")) {
            const uint8_t *z = memchr(s, 0, L);
            if (z) {
                size_t rn = L - (size_t)(z + 1 - s);
                const uint8_t *r = z + 1;
                if (rn >= 2) {
                    int cf = r[0], cm = r[1];
                    const uint8_t *r2 = r + 2;
                    size_t r2n = rn - 2;
                    const uint8_t *z1 = memchr(r2, 0, r2n);
                    const uint8_t *z2 = z1 ? memchr(z1 + 1, 0, r2n - (size_t)(z1 + 1 - r2)) : NULL;
                    if (z2 && cf != 0 && cm == 0) {
                        if (png_text_zlib_ok(z2 + 1, r2n - (size_t)(z2 + 1 - r2))) return NL_UNSUPPORTED;
                    }
                }
            }
            text_mem += L + (1u << 20);
        } else if (chunk_is(cid, "iCCP")) {
            const uint8_t *z = memchr(s, 0, L);
            size_t i = z ? (size_t)(z - s) : (size_t)-1;
            size_t cmi = i + 1; /* s[i + 1], i == -1 -> s[0] */
            if (cmi >= L) return NL_UNSUPPORTED;
            if (s[cmi] != 0) return NL_UNSUPPORTED;
            size_t st = i + 2;
            if (st < L && png_text_zlib_ok(s + st, L - st)) return NL_UNSUPPORTED;
        } else if (chunk_is(cid, "acTL")) {
            if (L < 8) return NL_UNSUPPORTED;
            if (n_frames_set) n_frames_set = 0;
            else {
                uint32_t nf = be32(s);
                if (!(nf == 0 || nf > 0x80000000u)) { n_frames_set = 1; n_frames_val = nf; }
            }
        } else if (chunk_is(cid, "fcTL")) {
            if (L < 26) return NL_UNSUPPORTED;
            long seq = (long)be32(s);
            if ((seq_num < 0 && seq != 0) || (seq_num >= 0 && seq_num != seq - 1)) return NL_UNSUPPORTED;
            seq_num = seq;
            uint64_t fw = be32(s + 4), fh = be32(s + 8), fx = be32(s + 12), fy = be32(s + 16);
            if (fx + fw > W || fy + fh > H) return NL_UNSUPPORTED;
            have_bbox = 1;
            bbox[0] = (uint32_t)fx; bbox[1] = (uint32_t)fy; bbox[2] = (uint32_t)(fx + fw); bbox[3] = (uint32_t)(fy + fh);
        } else if (chunk_is(cid, "fdAT")) {
            return NL_UNSUPPORTED;
        }
        /* crc */
        uint32_t crc = libdeflate_crc32(libdeflate_crc32(0, cid, 4), s, L);
        if (crc != be32(b + dpos + L)) return NL_UNSUPPORTED;
        pos = dpos + L + 4;
    }
    if (text_mem > 60u * 1024 * 1024) return NL_UNSUPPORTED;
    if (!have_ihdr || ihdr_count != 1 || !mode_ok) return NL_UNSUPPORTED;
    if (mbd != bd || mct != ct) return NL_UNSUPPORTED;
    if (too_big(W, H)) return NL_UNSUPPORTED;
    if (have_bbox && !(bbox[0] == 0 && bbox[1] == 0 && bbox[2] == W && bbox[3] == H)) return NL_UNSUPPORTED;
    /* PngImageFile: n_frames = im_n_frames or 1, +1 if the IDAT is a default image
     * (IDAT seen while im_n_frames is set and no fcTL bbox); is_animated = n_frames > 1 */
    int default_image = (!have_bbox && n_frames_set);
    uint64_t n_frames = (n_frames_set ? n_frames_val : 1) + (default_image ? 1 : 0);
    int is_animated = n_frames > 1;

    png_conv cv;
    memset(&cv, 0, sizeof cv);
    cv.depth = bd;
    int has_alpha = 0;
    switch (ct) {
    case 0:
        if (bd == 16) return NL_UNSUPPORTED; /* I;16 */
        cv.pk = bd == 8 ? PK_L8 : bd == 4 ? PK_L4 : bd == 2 ? PK_L2 : PK_1;
        cv.bitspp = bd; cv.ch = 3;
        break;
    case 2:
        cv.pk = bd == 8 ? PK_RGB8 : PK_RGB16; cv.bitspp = 3 * bd; cv.ch = 3;
        break;
    case 3: {
        if (!plte) return NL_UNSUPPORTED;
        if (plte_n > 768) return NL_UNSUPPORTED;
        cv.pk = PK_P; cv.bitspp = bd;
        size_t ne = plte_n / 3;
        for (int i = 0; i < 256; i++) cv.pal[i] = 0xFF000000u;
        for (size_t i = 0; i < ne; i++)
            cv.pal[i] = (uint32_t)plte[3 * i] | ((uint32_t)plte[3 * i + 1] << 8) | ((uint32_t)plte[3 * i + 2] << 16) | 0xFF000000u;
        if (trns_kind == 1) {
            if (trns_index >= 256) return NL_UNSUPPORTED;
            has_alpha = 1;
            cv.pal[trns_index] &= 0x00FFFFFFu;
        } else if (trns_kind == 2) {
            if (trns_n > 256) return NL_UNSUPPORTED;
            has_alpha = 1;
            for (size_t i = 0; i < trns_n; i++) cv.pal[i] = (cv.pal[i] & 0x00FFFFFFu) | ((uint32_t)trns[i] << 24);
        }
        cv.ch = has_alpha ? 4 : 3;
        break;
    }
    case 4:
        cv.pk = bd == 8 ? PK_LA8 : PK_LA16; cv.bitspp = 2 * bd; cv.ch = 4;
        break;
    case 6:
        cv.pk = bd == 8 ? PK_RGBA8 : PK_RGBA16; cv.bitspp = 4 * bd; cv.ch = 4;
        break;
    default: return NL_UNSUPPORTED;
    }

    /* ---- IDAT run ---- */
    size_t nidat = 0, zlen = 0;
    size_t p = idat_pos;
    const uint8_t *first_data = NULL;
    for (;;) {
        uint32_t L = be32(b + p);
        if ((uint64_t)p + 12 + L > n) return NL_UNSUPPORTED; /* truncated IDAT */
        if (!nidat) first_data = b + p + 8;
        nidat++;
        zlen += L;
        p += 12 + (size_t)L;
        if (p + 8 > n) break;
        const uint8_t *cid = b + p + 4;
        if (chunk_is(cid, "IDAT")) continue;
        if (chunk_is(cid, "DDAT") || chunk_is(cid, "fdAT")) return NL_UNSUPPORTED;
        break;
    }
    size_t after_idat = p;

    /* ---- post-IDAT chunks (PngImageFile.load_end) ---- */
    for (size_t q = after_idat;;) {
        if (q + 4 > n) break;                /* struct.error */
        if (q + 8 > n) break;                /* short cid -> SyntaxError */
        const uint8_t *cid = b + q + 4;
        if (!is_cid(cid)) break;
        uint32_t L = be32(b + q);
        if (chunk_is(cid, "IEND")) break;
        if (chunk_is(cid, "fcTL") && is_animated) break;
        if ((uint64_t)q + 8 + L > n) return NL_UNSUPPORTED; /* _safe_read raises */
        const uint8_t *s = b + q + 8;
        if (chunk_is(cid, "tEXt")) {
            text_mem += L;
        } else if (chunk_is(cid, "zTXt")) {
            const uint8_t *z = memchr(s, 0, L);
            if (z) {
                size_t vn = L - (size_t)(z + 1 - s);
                if (vn) {
                    if (z[1] != 0) return NL_UNSUPPORTED;
                    if (png_text_zlib_ok(z + 2, vn - 1)) return NL_UNSUPPORTED;
                }
            }
            text_mem += (1u << 20);
        } else if (chunk_is(cid, "iTXt")) {
            const uint8_t *z = memchr(s, 0, L);
            if (z) {
                size_t rn = L - (size_t)(z + 1 - s);
                const uint8_t *r = z + 1;
                if (rn >= 2) {
                    int cf = r[0], cm = r[1];
                    const uint8_t *r2 = r + 2;
                    size_t r2n = rn - 2;
                    const uint8_t *z1 = memchr(r2, 0, r2n);
                    const uint8_t *z2 = z1 ? memchr(z1 + 1, 0, r2n - (size_t)(z1 + 1 - r2)) : NULL;
                    if (z2 && cf != 0 && cm == 0) {
                        if (png_text_zlib_ok(z2 + 1, r2n - (size_t)(z2 + 1 - r2))) return NL_UNSUPPORTED;
                    }
                }
            }
            text_mem += L + (1u << 20);
        } else if (chunk_is(cid, "eXIf") || chunk_is(cid, "IDAT")) {
            /* fine */
        } else if (chunk_known(cid)) {
            return NL_UNSUPPORTED; /* handlers with side effects / possible raises */
        }
        q += 8 + (size_t)L + 4;
    }
    if (text_mem > 60u * 1024 * 1024) return NL_UNSUPPORTED;

    /* ---- inflate ---- */
    size_t rowb = ((size_t)W * cv.bitspp + 7) / 8;
    int bpp = (cv.bitspp + 7) / 8;
    size_t need;
    static const int OFF[] = {7, 3, 3, 1, 1, 0, 0}, SC[] = {0, 4, 0, 2, 0, 1, 0}, SR[] = {0, 0, 4, 0, 2, 0, 1},
                     CI[] = {8, 8, 4, 4, 2, 2, 1}, RI[] = {8, 8, 8, 4, 4, 2, 2};
    if (!interlace) {
        need = (rowb + 1) * H;
    } else {
        need = 0;
        for (int ps = 0; ps < 7; ps++) {
            size_t pw = ((size_t)W + OFF[ps]) / CI[ps];
            if (!pw || (uint32_t)SR[ps] >= H) continue;
            size_t ph = (H - SR[ps] + RI[ps] - 1) / RI[ps];
            need += ph * (((pw * cv.bitspp) + 7) / 8 + 1);
        }
    }
    const uint8_t *zin = first_data;
    uint8_t *zbuf = NULL;
    if (nidat > 1) {
        zbuf = (uint8_t *)malloc(zlen ? zlen : 1);
        if (!zbuf) return NL_UNSUPPORTED;
        size_t o = 0;
        size_t pp = idat_pos;
        for (size_t k = 0; k < nidat; k++) {
            uint32_t L = be32(b + pp);
            memcpy(zbuf + o, b + pp + 8, L);
            o += L;
            pp += 12 + (size_t)L;
        }
        zin = zbuf;
    }
    int direct = !interlace && (cv.pk == PK_RGB8 || cv.pk == PK_RGBA8);
    uint8_t *raw = (uint8_t *)malloc(need + 64);
    if (!raw) { free(zbuf); return NL_UNSUPPORTED; }
    size_t ain = 0, aout = 0;
    enum libdeflate_result rr = libdeflate_zlib_decompress_ex(get_dec(), zin, zlen, raw, need, &ain, &aout);
    free(zbuf);
    if (rr != LIBDEFLATE_SUCCESS || aout != need) { free(raw); return NL_UNSUPPORTED; }

    int ch = cv.ch;
    if (!interlace) {
        if (direct) {
            /* unfilter in place, then compact rows to the start of the buffer */
            uint8_t *prev = NULL;
            for (uint32_t y = 0; y < H; y++) {
                uint8_t *row = raw + (size_t)y * (rowb + 1);
                uint8_t f = row[0];
                if (unfilter_row(f, row + 1, prev, rowb, bpp)) { free(raw); return NL_ERROR; }
                uint8_t *dst = raw + (size_t)y * rowb;
                memmove(dst, row + 1, rowb);
                prev = dst;
            }
            out->w = (int)W; out->h = (int)H; out->ch = ch; out->px = raw;
            return NL_OK;
        }
        uint8_t *px = (uint8_t *)malloc((size_t)W * H * ch + 4);
        if (!px) { free(raw); return NL_UNSUPPORTED; }
        uint8_t *prev = NULL;
        for (uint32_t y = 0; y < H; y++) {
            uint8_t *row = raw + (size_t)y * (rowb + 1);
            if (unfilter_row(row[0], row + 1, prev, rowb, bpp)) { free(raw); free(px); return NL_ERROR; }
            png_conv_row(&cv, row + 1, px + (size_t)y * W * ch, (int)W);
            prev = row + 1;
        }
        free(raw);
        out->w = (int)W; out->h = (int)H; out->ch = ch; out->px = px;
        return NL_OK;
    }
    /* Adam7 */
    uint8_t *px = (uint8_t *)calloc((size_t)W * H * ch + 4, 1);
    if (!px) { free(raw); return NL_UNSUPPORTED; }
    size_t off = 0;
    for (int ps = 0; ps < 7; ps++) {
        size_t pw = ((size_t)W + OFF[ps]) / CI[ps];
        if (!pw || (uint32_t)SR[ps] >= H) continue;
        size_t prb = (pw * cv.bitspp + 7) / 8;
        uint8_t *prev = NULL;
        for (uint32_t y = (uint32_t)SR[ps]; y < H; y += (uint32_t)RI[ps]) {
            uint8_t *row = raw + off;
            if (unfilter_row(row[0], row + 1, prev, prb, bpp)) { free(raw); free(px); return NL_ERROR; }
            size_t col = (size_t)SC[ps];
            for (size_t i = 0; i < pw; i++, col += (size_t)CI[ps])
                png_conv_px(&cv, row + 1, (int)i, px + ((size_t)y * W + col) * ch);
            prev = row + 1;
            off += prb + 1;
        }
    }
    free(raw);
    out->w = (int)W; out->h = (int)H; out->ch = ch; out->px = px;
    return NL_OK;
}

/* ======================================================================== */
/* GIF                                                                      */
/* ======================================================================== */

typedef struct {
    const uint8_t *b;
    size_t n;
    size_t pos;
} rd_t;

/* GifImageFile.data(): s = read(1); if s and s[0]: return read(s[0]); else None.
 * returns length (>0) or 0 for None; *p set to the data (possibly short at EOF). */
static size_t gif_data(rd_t *r, const uint8_t **p, size_t *got) {
    if (r->pos >= r->n) { *got = 0; return 0; }
    size_t L = r->b[r->pos++];
    if (!L) { *got = 0; return 0; }
    *p = r->b + r->pos;
    size_t avail = r->n - r->pos;
    *got = L < avail ? L : avail;
    r->pos += *got;
    return L;
}

static int gif_pal_needed(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i += 3)
        if (!(i / 3 == p[i] && p[i] == p[i + 1] && p[i + 1] == p[i + 2])) return 1;
    return 0;
}

/* sub-block stream (GifDecode.c reads length-prefixed blocks forever, across
 * terminators), plus the ImageFile.load 64 KiB read emulation. */
typedef struct {
    const uint8_t *b;
    size_t n, tile;       /* file, tile offset (first block length byte) */
    size_t fpos;          /* next block length byte position in file */
    uint8_t *ds;          /* gathered data stream */
    size_t dn, dcap;
    size_t *bend_cum;     /* per block: cumulative stream bytes after it */
    size_t *bend_rel;     /* per block: end offset relative to tile */
    size_t nb, bcap;
    size_t need_blk;      /* blocks [0, need_blk) have been "opened" by the decoder */
    size_t k;             /* number of read() calls so far */
    int eof;              /* no more complete blocks in file */
} gstream;

static int gs_grow(gstream *g) {
    /* append blocks until a 0-length block (inclusive) or EOF / partial block */
    int added = 0;
    while (!g->eof) {
        if (g->fpos >= g->n) { g->eof = 1; break; }
        size_t L = g->b[g->fpos];
        if (g->fpos + 1 + L > g->n) { g->eof = 1; break; }
        if (g->dn + L + 16 > g->dcap) {
            size_t nc = g->dcap * 2 + L + 4096;
            uint8_t *nd = (uint8_t *)realloc(g->ds, nc);
            if (!nd) return -1;
            g->ds = nd;
            g->dcap = nc;
        }
        if (g->nb + 1 > g->bcap) {
            size_t nc = g->bcap * 2 + 256;
            size_t *a = (size_t *)realloc(g->bend_cum, nc * sizeof(size_t));
            if (!a) return -1;
            g->bend_cum = a;
            a = (size_t *)realloc(g->bend_rel, nc * sizeof(size_t));
            if (!a) return -1;
            g->bend_rel = a;
            g->bcap = nc;
        }
        memcpy(g->ds + g->dn, g->b + g->fpos + 1, L);
        g->dn += L;
        g->fpos += 1 + L;
        g->bend_cum[g->nb] = g->dn;
        g->bend_rel[g->nb] = g->fpos - g->tile;
        g->nb++;
        added = 1;
        if (L == 0) break;
    }
    memset(g->ds + g->dn, 0, 16);
    return added;
}

/* Make stream bytes [0, nbytes) available, doing the read() accounting.
 * returns 0 ok, 1 truncated (reference raises), -1 alloc failure */
static int gs_need(gstream *g, size_t nbytes) {
    for (;;) {
        while (g->need_blk < g->nb && (g->need_blk == 0 ? 0 : g->bend_cum[g->need_blk - 1]) < nbytes) {
            /* opening block need_blk: it must lie fully within the data read so far */
            size_t R = g->n - g->tile;
            size_t end = g->bend_rel[g->need_blk];
            while (end > PIL_MAXBLOCK * g->k) {
                if (PIL_MAXBLOCK * g->k >= R) return 1;
                g->k++;
            }
            g->need_blk++;
        }
        if (g->need_blk && g->bend_cum[g->need_blk - 1] >= nbytes) return 0;
        int a = gs_grow(g);
        if (a < 0) return -1;
        if (!a) return 1; /* the decoder wants more, file has no more complete blocks */
    }
}

/* Decode the LZW stream into lin[0..npix).  0 ok, NL_ERROR, NL_UNSUPPORTED. */
static int gif_lzw(gstream *g, int bits, uint8_t *lin, size_t npix) {
    if (bits < 1 || bits > 11) return NL_UNSUPPORTED;
    const int clear = 1 << bits, endc = clear + 1;
    /* table: for codes >= clear+2: position and length in lin[] */
    static __thread uint32_t tpos[4096];
    static __thread uint16_t tlen[4096];
    int next = 0, codesize = 0, codemask = 0;
    int state = 1;
    size_t lastpos = 0, lastlen = 0;
    size_t op = 0;
    uint64_t bitpos = 0;
    size_t avail_bits = 0; /* bits made available via gs_need */
    for (;;) {
        if (state == 1) {
            next = clear + 2;
            codesize = bits + 1;
            codemask = (1 << codesize) - 1;
            state = 2;
        }
        if (bitpos + (uint64_t)codesize > avail_bits) {
            int r = gs_need(g, (size_t)((bitpos + codesize + 7) >> 3));
            if (r > 0) return NL_ERROR;
            if (r < 0) return NL_UNSUPPORTED;
            avail_bits = g->bend_cum[g->need_blk - 1] * 8;
        }
        uint64_t v;
        memcpy(&v, g->ds + (bitpos >> 3), 8);
        int c = (int)((v >> (bitpos & 7)) & (uint64_t)codemask);
        bitpos += (uint64_t)codesize;

        if (c == clear) {
            if (state != 2) state = 1;
            continue;
        }
        if (c == endc) {
            /* decoder returns; ImageFile.load reads again */
            size_t R = g->n - g->tile;
            if (PIL_MAXBLOCK * g->k >= R) return NL_ERROR;
            g->k++;
            continue;
        }
        if (state == 2) {
            if (c > clear) return NL_ERROR;
            lin[op] = (uint8_t)c;
            lastpos = op;
            lastlen = 1;
            op += 1;
            state = 3;
        } else {
            int thiscode = c;
            if (c > next) return NL_ERROR;
            size_t sp, sl;
            if (c == next) {
                /* KwKwK: string(lastcode) + first(string(lastcode)) */
                memmove(lin + op, lin + lastpos, lastlen);
                lin[op + lastlen] = lin[lastpos];
                sp = op;
                sl = lastlen + 1;
            } else if (c < clear) {
                lin[op] = (uint8_t)c;
                sp = op;
                sl = 1;
            } else {
                size_t p0 = tpos[c], l0 = tlen[c];
                if (l0 <= 16) {
                    __m128i t = _mm_loadu_si128((const __m128i *)(lin + p0));
                    _mm_storeu_si128((__m128i *)(lin + op), t);
                } else {
                    memcpy(lin + op, lin + p0, l0);
                }
                sp = op;
                sl = l0;
            }
            if (next < 4096) {
                /* entry = string(lastcode) + first char of this string; it sits at lastpos */
                tpos[next] = (uint32_t)lastpos;
                tlen[next] = (uint16_t)(lastlen + 1);
                if (next == codemask && codesize < 12) {
                    codesize++;
                    codemask = (1 << codesize) - 1;
                }
                next++;
            }
            (void)thiscode;
            lastpos = sp;
            lastlen = sl;
            op += sl;
        }
        if (op >= npix) return 0;
    }
}

static int decode_gif(const uint8_t *b, size_t n, nl_img *out) {
    if (n < 13) return NL_UNSUPPORTED;
    uint32_t W = le16(b + 6), H = le16(b + 8);
    int flags = b[10];
    rd_t r = {b, n, 13};
    const uint8_t *gpal = NULL;
    size_t gpal_n = 0;
    if (flags & 128) {
        size_t ps = (size_t)3 << ((flags & 7) + 1);
        if (r.pos + ps > n) return NL_UNSUPPORTED;
        if (gif_pal_needed(b + r.pos, ps)) { gpal = b + r.pos; gpal_n = ps; }
        r.pos += ps;
    }
    /* _seek(0) */
    if (r.pos >= n || b[r.pos] == ';') return NL_UNSUPPORTED;
    int s = b[r.pos++];
    int have_s = 1;
    int ftrans = -1;
    int interlace = -1;
    int lpal_state = 0; /* 0: None, 1: palette, 2: False */
    const uint8_t *lpal = NULL;
    size_t lpal_n = 0;
    uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int lzwbits = 0;
    size_t tile = 0;
    for (;;) {
        if (!have_s) {
            if (r.pos >= n) break;
            s = b[r.pos++];
            have_s = 1;
        }
        if (s == ';') break;
        if (s == '!') {
            if (r.pos >= n) return NL_UNSUPPORTED;
            int label = b[r.pos++];
            const uint8_t *blk = NULL;
            size_t got = 0;
            size_t bl = gif_data(&r, &blk, &got);
            if (label == 249 && bl) {
                if (got < 3) return NL_UNSUPPORTED;
                int fl = blk[0];
                if (fl & 1) {
                    if (got < 4) return NL_UNSUPPORTED;
                    ftrans = blk[3];
                }
            } else if (label == 254) {
                while (bl) bl = gif_data(&r, &blk, &got);
                have_s = 0;
                continue;
            } else if (label == 255 && bl) {
                if (got >= 11 && !memcmp(blk, "NETSCAPE2.0", 11)) {
                    bl = gif_data(&r, &blk, &got);
                    (void)bl;
                }
            }
            const uint8_t *d2;
            size_t g2;
            while (gif_data(&r, &d2, &g2)) {
            }
        } else if (s == ',') {
            if (r.pos + 9 > n) return NL_UNSUPPORTED;
            const uint8_t *d = b + r.pos;
            r.pos += 9;
            x0 = le16(d); y0 = le16(d + 2);
            x1 = x0 + le16(d + 4); y1 = y0 + le16(d + 6);
            int fl = d[8];
            interlace = (fl & 64) != 0;
            if (fl & 128) {
                size_t ps = (size_t)3 << ((fl & 7) + 1);
                if (r.pos + ps > n) return NL_UNSUPPORTED;
                if (gif_pal_needed(b + r.pos, ps)) { lpal_state = 1; lpal = b + r.pos; lpal_n = ps; }
                else lpal_state = 2;
                r.pos += ps;
            }
            if (r.pos >= n) return NL_UNSUPPORTED;
            lzwbits = b[r.pos++];
            tile = r.pos;
            break;
        }
        have_s = 0;
    }
    if (interlace < 0) return NL_UNSUPPORTED; /* EOFError: image not found */
    uint32_t CW = x1 > W ? x1 : W, CH = y1 > H ? y1 : H;
    if (too_big(CW, CH)) return NL_UNSUPPORTED;
    uint32_t fw = x1 - x0, fh = y1 - y0;
    if (!fw || !fh) return NL_UNSUPPORTED;

    const uint8_t *pal = NULL;
    size_t pal_n = 0;
    int mode_p;
    if (lpal_state == 1) { pal = lpal; pal_n = lpal_n; mode_p = 1; }
    else if (lpal_state == 2) { mode_p = 0; }
    else if (gpal) { pal = gpal; pal_n = gpal_n; mode_p = 1; }
    else mode_p = 0;
    int has_alpha = mode_p && ftrans >= 0;
    int ch = has_alpha ? 4 : 3;
    uint8_t fill = ftrans >= 0 ? (uint8_t)ftrans : 0;

    /* LZW */
    size_t npix = (size_t)fw * fh;
    uint8_t *lin = (uint8_t *)malloc(npix + 4096 + 64);
    if (!lin) return NL_UNSUPPORTED;
    gstream g;
    memset(&g, 0, sizeof g);
    g.b = b; g.n = n; g.tile = tile; g.fpos = tile; g.k = 1;
    int rc = gif_lzw(&g, lzwbits, lin, npix);
    free(g.ds); free(g.bend_cum); free(g.bend_rel);
    if (rc != 0) { free(lin); return rc; }

    /* colour table */
    uint32_t tab[256];
    if (mode_p) {
        for (int i = 0; i < 256; i++) tab[i] = 0xFF000000u;
        size_t ne = pal_n / 3;
        for (size_t i = 0; i < ne && i < 256; i++)
            tab[i] = (uint32_t)pal[3 * i] | ((uint32_t)pal[3 * i + 1] << 8) | ((uint32_t)pal[3 * i + 2] << 16) | 0xFF000000u;
        if (has_alpha) tab[ftrans] &= 0x00FFFFFFu;
    } else {
        for (int i = 0; i < 256; i++) tab[i] = (uint32_t)i * 0x010101u | 0xFF000000u;
    }

    uint8_t *px = (uint8_t *)malloc((size_t)CW * CH * ch + 4);
    if (!px) { free(lin); return NL_UNSUPPORTED; }
    size_t rowbytes = (size_t)CW * ch;
    int full = (x0 == 0 && y0 == 0 && fw == CW && fh == CH);
    if (!full) {
        uint32_t fv = tab[fill];
        for (size_t i = 0; i < (size_t)CW * CH; i++) memcpy(px + i * ch, &fv, 4 > ch ? (size_t)ch : 4);
    }
    /* row order of the decoded stream (GifDecode.c NEWLINE) */
    uint32_t yy = 0;
    int pass = interlace ? 1 : 0, step = interlace ? 8 : 1;
    for (uint32_t r_i = 0; r_i < fh; r_i++) {
        if (yy >= fh) { free(lin); free(px); return NL_UNSUPPORTED; } /* cannot happen */
        const uint8_t *src = lin + (size_t)r_i * fw;
        uint8_t *dst = px + (size_t)(y0 + yy) * rowbytes + (size_t)x0 * ch;
        if (ch == 4) {
            for (uint32_t x = 0; x < fw; x++) memcpy(dst + 4 * x, &tab[src[x]], 4);
        } else {
            uint32_t x = 0;
            for (; x + 1 < fw; x++) memcpy(dst + 3 * x, &tab[src[x]], 4); /* 1 byte ahead, same row */
            memcpy(dst + 3 * x, &tab[src[x]], 3);
        }
        yy += (uint32_t)step;
        while (yy >= fh && pass) {
            if (pass == 1) { yy = 4; pass = 2; }
            else if (pass == 2) { step = 4; yy = 2; pass = 3; }
            else { step = 2; yy = 1; pass = 0; }
        }
    }
    free(lin);
    out->w = (int)CW; out->h = (int)CH; out->ch = ch; out->px = px;
    return NL_OK;
}

/* ======================================================================== */
/* WEBP                                                                     */
/* ======================================================================== */

static int decode_webp(const uint8_t *b, size_t n, nl_img *out) {
    WebPBitstreamFeatures feat;
    int has_alpha = 1; /* Pillow: RGBA unless WebPGetFeatures says no alpha */
    if (WebPGetFeatures(b, n, &feat) == VP8_STATUS_OK && !feat.has_alpha) has_alpha = 0;
    WebPData data = {b, n};
    WebPAnimDecoder *dec = WebPAnimDecoderNew(&data, NULL);
    if (!dec) return NL_UNSUPPORTED;
    WebPAnimInfo info;
    if (!WebPAnimDecoderGetInfo(dec, &info)) { WebPAnimDecoderDelete(dec); return NL_UNSUPPORTED; }
    uint32_t W = info.canvas_width, H = info.canvas_height;
    if (too_big(W, H)) { WebPAnimDecoderDelete(dec); return NL_UNSUPPORTED; }
    uint8_t *fb;
    int ts;
    if (!WebPAnimDecoderGetNext(dec, &fb, &ts)) { WebPAnimDecoderDelete(dec); return NL_UNSUPPORTED; }
    int ch = has_alpha ? 4 : 3;
    uint8_t *px = (uint8_t *)malloc((size_t)W * H * ch + 4);
    if (!px) { WebPAnimDecoderDelete(dec); return NL_UNSUPPORTED; }
    size_t np = (size_t)W * H;
    if (ch == 4) memcpy(px, fb, np * 4);
    else
        for (size_t i = 0; i < np; i++) memcpy(px + 3 * i, fb + 4 * i, 3);
    WebPAnimDecoderDelete(dec);
    out->w = (int)W; out->h = (int)H; out->ch = ch; out->px = px;
    return NL_OK;
}

/* ======================================================================== */

int nl_decode(const uint8_t *buf, size_t len, nl_img *out) {
    out->w = out->h = out->ch = 0;
    out->px = NULL;
    if (!buf) return NL_UNSUPPORTED;
    if (len >= 3 && buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF) return decode_jpeg(buf, len, out);
    if (len >= 8 && !memcmp(buf, "\x89PNG\r\n\x1a\n", 8)) return decode_png(buf, len, out);
    if (len >= 6 && (!memcmp(buf, "GIF87a", 6) || !memcmp(buf, "GIF89a", 6))) return decode_gif(buf, len, out);
    if (len >= 12 && !memcmp(buf, "RIFF", 4) && !memcmp(buf + 8, "WEBP", 4)) return decode_webp(buf, len, out);
    return NL_UNSUPPORTED;
}
