/* module B private helpers (exported from the test library only for validation). */
#ifndef NL_PREP_INTERNAL_H
#define NL_PREP_INTERNAL_H
#include <stdint.h>

/* Pillow Convert.c rgb2hsv_row, verbatim (scalar reference). */
void nlp_rgb2hsv_scalar(uint8_t *out, const uint8_t *in);
/* Exhaustive check of the AVX2 HSV against the scalar reference; returns #mismatches. */
long nlp_hsv_selftest(void);
/* Free the calling thread's scratch arena (optional; e.g. before thread exit). */
void nlp_release_scratch(void);
/* HSV of n packed RGB pixels (test export). */
void nlp_hsv(const uint8_t *rgb, size_t n, uint8_t *H, uint8_t *S, uint8_t *V);

#endif
