/*
 * engine_escha.c - the Escha dense op of our llama.cpp fork, on the CPU, outside ggml.
 *
 * ggml_escha_hadamard_128, ggml_escha_codebook, ggml_escha_decode_tile and the body of
 * ggml_compute_forward_escha_mul_mat are copied from the engine source (release tarball c07efbec5,
 * ggml/src/ggml-cpu/ops.cpp, MIT; the op is by Ajay9o9 / AjHomelab, llama.cpp-escha). The only changes:
 * ggml types and macros replaced by plain C, one thread, file I/O. The fp16 <-> fp32 conversions use the
 * F16C instructions, which is what GGML_CPU_FP16_TO_FP32 / GGML_CPU_FP32_TO_FP16 use on x86 with F16C.
 *
 * usage: engine_escha code.bin dep.bin IC OC K w_out.bin [x.bin M rin.bin rout.bin y_out.bin]
 *   code.bin : int16 [IC/16][OC/16][16K]  (the GGUF tensor bytes)
 *   dep.bin  : int16 [256][16]            (escha_dep_k2 or escha_dep_k3 of the GGUF)
 *   w_out    : f32 [IC][OC]  decoded bare weight (tile (i, j), entry r*16+c -> W[i*16+r][j*16+c])
 *   x.bin    : f32 [M][IC];  rin.bin f16 [IC];  rout.bin f16 [OC];  y_out f32 [M][OC]
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t ggml_fp16_t;

/* the MSVC forms that ggml uses for F16C (ggml-impl.h / simd-mappings.h) */
static inline float fp16_to_fp32(ggml_fp16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
static inline ggml_fp16_t fp32_to_fp16(float f) { return (ggml_fp16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0); }

#define GGML_CPU_FP16_TO_FP32(x) fp16_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) fp32_to_fp16(x)

/* ---- copied from ops.cpp (c07efbec5) ---------------------------------------------------------- */

static void ggml_escha_hadamard_128(float * v, int64_t n) {
    const float scale = 1.0f/sqrtf(128.0f);

    for (int64_t off = 0; off < n; off += 128) {
        float * b = v + off;

        for (int len = 1; len < 128; len <<= 1) {
            for (int i = 0; i < 128; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    const float a0 = b[i + j];
                    const float a1 = b[i + j + len];

                    b[i + j]       = a0 + a1;
                    b[i + j + len] = a0 - a1;
                }
            }
        }

        for (int i = 0; i < 128; i++) {
            b[i] *= scale;
        }
    }
}

static float ggml_escha_codebook(uint32_t idx) {
    const uint32_t x = ((idx*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
    ggml_fp16_t lo, hi;
    memcpy(&lo, (const char *) &x,     sizeof(lo));
    memcpy(&hi, (const char *) &x + 2, sizeof(hi));
    const float v = GGML_CPU_FP16_TO_FP32(lo) + GGML_CPU_FP16_TO_FP32(hi);
    return GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(v));
}

static void ggml_escha_decode_tile(
        const uint8_t * payload,
        const int16_t * dep,
        float * tile) {
    for (int p = 0; p < 256; ++p) {
        const int16_t * d = dep + p*16;

        uint32_t idx = 0;
        for (int b = 0; b < 16; ++b) {
            idx |= (uint32_t) ((payload[d[b] >> 3] >> (d[b] & 7)) & 1) << b;
        }

        tile[p] = ggml_escha_codebook(idx);
    }
}

/* ---- harness ---------------------------------------------------------------------------------- */

static void * slurp(const char * path, size_t want) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    void * p = malloc(want);
    if (fread(p, 1, want, f) != want) { fprintf(stderr, "%s: short read\n", path); exit(2); }
    fclose(f);
    return p;
}

static void spit(const char * path, const void * p, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 7 && argc != 12) {
        fprintf(stderr, "usage: engine_escha code.bin dep.bin IC OC K w_out.bin [x.bin M rin.bin rout.bin y_out.bin]\n");
        return 1;
    }
    const int64_t IC = atoll(argv[3]), OC = atoll(argv[4]), K = atoll(argv[5]);
    const int64_t nit = IC/16, nct = OC/16, tile_stride = 16*K;

    const int16_t * code_d = (const int16_t *) slurp(argv[1], (size_t) (nit*nct*tile_stride*2));
    const int16_t * dep_d  = (const int16_t *) slurp(argv[2], 256*16*2);

    /* decoded bare weight, through the same tile decoder */
    float * W = (float *) malloc((size_t) (IC*OC) * sizeof(float));
    float tile[256];
    for (int64_t i = 0; i < nit; ++i) {
        for (int64_t j = 0; j < nct; ++j) {
            ggml_escha_decode_tile((const uint8_t *)(code_d + (i*nct + j)*tile_stride), dep_d, tile);
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    W[(i*16 + r)*OC + j*16 + c] = tile[r*16 + c];
                }
            }
        }
    }
    spit(argv[6], W, (size_t) (IC*OC) * sizeof(float));
    if (argc == 7) {
        return 0;
    }

    /* the op itself: body of ggml_compute_forward_escha_mul_mat, one thread, rows of x */
    const int64_t M = atoll(argv[8]);
    const float       * x      = (const float *)       slurp(argv[7],  (size_t) (M*IC) * sizeof(float));
    const ggml_fp16_t * rin_d  = (const ggml_fp16_t *) slurp(argv[9],  (size_t) IC * 2);
    const ggml_fp16_t * rout_d = (const ggml_fp16_t *) slurp(argv[10], (size_t) OC * 2);
    float * y   = (float *) malloc((size_t) (M*OC) * sizeof(float));
    float * u   = (float *) malloc((size_t) IC * sizeof(float));
    float * acc = (float *) malloc((size_t) OC * sizeof(float));

    for (int64_t ir = 0; ir < M; ++ir) {
        const float * xr = x + ir*IC;

        for (int64_t i = 0; i < IC; ++i) {
            u[i] = xr[i]*GGML_CPU_FP16_TO_FP32(rin_d[i]);
        }
        ggml_escha_hadamard_128(u, IC);

        memset(acc, 0, OC*sizeof(float));

        for (int64_t i = 0; i < nit; ++i) {
            const float * ub = u + i*16;

            for (int64_t j = 0; j < nct; ++j) {
                ggml_escha_decode_tile((const uint8_t *)(code_d + (i*nct + j)*tile_stride), dep_d, tile);

                float * ab = acc + j*16;
                for (int r = 0; r < 16; ++r) {
                    const float ur = ub[r];
                    if (ur == 0.0f) {
                        continue;
                    }

                    const float * tr = tile + r*16;
                    for (int c = 0; c < 16; ++c) {
                        ab[c] += ur*tr[c];
                    }
                }
            }
        }

        ggml_escha_hadamard_128(acc, OC);

        float * dr_ = y + ir*OC;
        for (int64_t c = 0; c < OC; ++c) {
            dr_[c] = acc[c]*GGML_CPU_FP16_TO_FP32(rout_d[c]);
        }
    }
    spit(argv[11], y, (size_t) (M*OC) * sizeof(float));
    return 0;
}
