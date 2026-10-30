/*
 * minimaxh3.c - MiniMax-H3 text-to-video-with-audio generation on the GPU with
 * hand-written PTX kernels (minimaxh3.ptx) driven through the CUDA Driver API.
 * No other libraries. A prompt becomes a short 24 fps video with a stereo
 * 32 kHz soundtrack, written as an AVI (plus optional PNG frames and a WAV).
 *
 * Four networks run one after the other, each with the whole GPU to itself
 * (its weights are loaded, used and freed before the next one starts):
 *
 *   1. text encoder: the Qwen3-VL-32B language model cut to 50 layers
 *      (qwen3vl_32b_minimax_h3-Q4_K_M.gguf, 15.5 GB of Q4_K / Q6_K matrices).
 *      The prompt (raw text, no chat template) is tokenized with Qwen2's
 *      byte-level BPE (tokenizer.c) and the hidden state after the last layer
 *      (no final norm) is the condition. The layers don't fit in GPU memory:
 *      every layer is copied in from the memory-mapped file, two slots ahead.
 *   2. diffusion transformer (minimax_h3_ref2va_pruned-Q4_K.gguf): 2 token-
 *      refiner blocks over the text, then 50 AdaLN blocks over one packed
 *      sequence [text | audio left | audio right | video patches] with 3-D
 *      RoPE, run for every step of a flow-matching Euler sampler. Blocks that
 *      don't fit stay in pinned host memory and stream through two GPU slots.
 *   3. video VAE decoder (minimax_h3_video_vae_fp16.safetensors): a 36-layer
 *      ViT that turns every latent token into a 4 x 16 x 16 block of pixels.
 *   4. audio VAE decoder (minimax_h3_audio_vae_fp32.safetensors): BigVGAN,
 *      40 latents per second -> 32 kHz, once per stereo channel.
 *
 * Big matrices stay in their file formats on the GPU (Q4_K, Q6_K, f16) and
 * are expanded to f16 one slice of rows at a time right before their GEMM
 * (f16 tensor cores, f32 accumulation). Residual streams, norms and the
 * small heads are f32.
 *
 * Usage (see usage() for every option):
 *   ./minimaxh3_demo [options] PROMPT       generate out.avi
 *   ./minimaxh3_demo test                   kernel self-tests against the CPU
 *   ./minimaxh3_demo check DIR              compare every stage with the
 *                                           tensors stable-diffusion.cpp
 *                                           dumped into DIR (see README.md)
 *   ./minimaxh3_demo tokenize TEXT          show the tokens of TEXT
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "gguf.h"
#include "media.h"
#include "quant.h"
#include "safetensors.h"
#include "tokenizer.h"

/* Abort with the driver's error name/message if a CUDA Driver API call fails. */
#define CHECK(call)                                                          \
    do {                                                                     \
        CUresult err_ = (call);                                              \
        if (err_ != CUDA_SUCCESS) {                                          \
            const char *name_ = NULL, *msg_ = NULL;                          \
            cuGetErrorName(err_, &name_);                                    \
            cuGetErrorString(err_, &msg_);                                   \
            fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__,         \
                    __LINE__, #call, name_ ? name_ : "?", msg_ ? msg_ : ""); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define DEFAULT_CUBIN "minimaxh3.cubin"
#define DEFAULT_DIT "minimax_h3_ref2va_pruned-Q4_K.gguf"
#define DEFAULT_TE "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
#define DEFAULT_VAE "minimax_h3_video_vae_fp16.safetensors"
#define DEFAULT_AVAE "minimax_h3_audio_vae_fp32.safetensors"
#define DEFAULT_MERGES "qwen_merges.txt"
#define SPARE_BYTES (192u << 20)   /* GPU memory left free */
#define SCRATCH_BYTES (64u << 20)  /* f16 rows of a quantized matrix, before its GEMM */

/* Host wall-clock time in milliseconds. */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of host memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

/* xorshift64* - small deterministic RNG for the self-tests */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rng_u32(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}
static float rng_uniform(float lo, float hi) /* [lo, hi) */
{
    return lo + (hi - lo) * (rng_u32() >> 8) * (1.0f / 16777216.0f);
}

/* ------------------------------------------------------------------------ */
/* Kernels and launch helpers                                               */

static CUfunction k_dq4k, k_dq6k, k_rms_mod, k_layernorm, k_qk_rope, k_swiglu, k_adaln,
    k_linear_f32, k_cast_f16, k_axpby, k_im2col, k_col2im, k_snake, k_gemm, k_attn128, k_attn64;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    if (cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the minimaxh3 directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_dq4k, "dequant_q4k"}, {&k_dq6k, "dequant_q6k"}, {&k_rms_mod, "rms_mod"},
        {&k_layernorm, "layernorm"}, {&k_qk_rope, "qk_norm_rope"}, {&k_swiglu, "swiglu"},
        {&k_adaln, "adaln"}, {&k_linear_f32, "linear_f32"}, {&k_cast_f16, "cast_f16"},
        {&k_axpby, "axpby"}, {&k_im2col, "im2col1d"}, {&k_col2im, "col2im_t"},
        {&k_snake, "aa_snake"}, {&k_gemm, "gemm_f16"}, {&k_attn128, "attn_d128"},
        {&k_attn64, "attn_d64"},
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); i++)
        CHECK(cuModuleGetFunction(k[i].fn, mod, k[i].name));
    CUdevice dev;
    int sms;
    CHECK(cuCtxGetDevice(&dev));
    CHECK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    max_blocks = 8 * sms;
}

/* 1-D launch of a grid-stride kernel over `total` items. */
static void launch_1d(CUfunction fn, size_t total, void **args)
{
    const unsigned threads = 256;
    size_t blocks = (total + threads - 1) / threads;
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks == 0)
        blocks = 1;
    CHECK(cuLaunchKernel(fn, (unsigned)blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL));
}

static void dequant(uint32_t type, CUdeviceptr src, CUdeviceptr dst, size_t blocks)
{
    if (type == GGML_Q4_K) {
        unsigned n = (unsigned)(blocks * 8);
        void *args[] = {&src, &dst, &n};
        launch_1d(k_dq4k, n, args);
    } else {
        unsigned n = (unsigned)(blocks * 16);
        void *args[] = {&src, &dst, &n};
        launch_1d(k_dq6k, n, args);
    }
}

/* per row of x f32 [rows][n]: RMS norm (times w), the AdaLN modulation of the
 * row's segment, out as f16 (times oscale) and/or f32; see minimaxh3.ptx */
static void rms_mod(CUdeviceptr x, CUdeviceptr w, CUdeviceptr shift, CUdeviceptr scale,
                    unsigned mstride, CUdeviceptr seg, CUdeviceptr y16, CUdeviceptr y32,
                    unsigned rows, unsigned n, float eps, float oscale)
{
    void *args[] = {&x, &w, &shift, &scale, &mstride, &seg, &y16, &y32, &rows, &n, &eps, &oscale};
    CHECK(cuLaunchKernel(k_rms_mod, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void layernorm(CUdeviceptr x, CUdeviceptr w, CUdeviceptr b, CUdeviceptr y16,
                      unsigned rows, unsigned n, float eps)
{
    void *args[] = {&x, &w, &b, &y16, &rows, &n, &eps};
    CHECK(cuLaunchKernel(k_layernorm, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* RMS-norm every q and k head of a fused projection in place, then rotate
 * the first `rot` dims by the row's angles (cs f32 [rows][rot/2][2], or 0) */
static void qk_norm_rope(CUdeviceptr x, unsigned ld, unsigned rows, unsigned nq, unsigned nk,
                         unsigned qoff, unsigned koff, unsigned hstride, unsigned d,
                         CUdeviceptr qw, CUdeviceptr kw, float eps, CUdeviceptr cs, unsigned rot)
{
    void *args[] = {&x, &ld, &rows, &nq, &nk, &qoff, &koff, &hstride, &d, &qw, &kw, &eps, &cs, &rot};
    size_t threads = (size_t)rows * (nq + nk) * 32;
    CHECK(cuLaunchKernel(k_qk_rope, (unsigned)((threads + 255) / 256), 1, 1, 256, 1, 1, 0, NULL,
                         args, NULL));
}

static void swiglu(CUdeviceptr in, unsigned ldi, CUdeviceptr out, unsigned ldo, unsigned rows,
                   unsigned f, float si, float so)
{
    void *args[] = {&in, &ldi, &out, &ldo, &rows, &f, &si, &so};
    launch_1d(k_swiglu, (size_t)rows * f / 2, args);
}

/* out f32 [L][R][J] = b + W . t for L stacked [J][8] modulation matrices */
static void adaln(CUdeviceptr w, CUdeviceptr b, CUdeviceptr t, CUdeviceptr out, unsigned j,
                  unsigned l, unsigned r)
{
    void *args[] = {&w, &b, &t, &out, &j, &l, &r};
    launch_1d(k_adaln, (size_t)l * j, args);
}

/* y [m][ldy] (=, or += when acc) b + x . w^T, all f32 */
static void linear_f32(CUdeviceptr x, unsigned ldx, CUdeviceptr w, CUdeviceptr b, CUdeviceptr y,
                       unsigned ldy, unsigned m, unsigned n, unsigned k, unsigned acc)
{
    void *args[] = {&x, &ldx, &w, &b, &y, &ldy, &m, &n, &k, &acc};
    size_t outs = (size_t)m * n;
    CHECK(cuLaunchKernel(k_linear_f32, (unsigned)((outs + 7) / 8), 1, 1, 256, 1, 1, 0, NULL, args,
                         NULL));
}

static void cast_f16(CUdeviceptr x, unsigned ldx, CUdeviceptr y, unsigned ldy, unsigned rows,
                     unsigned cols, float s)
{
    void *args[] = {&x, &ldx, &y, &ldy, &rows, &cols, &s};
    launch_1d(k_cast_f16, (size_t)rows * cols / 2, args);
}

static void axpby(CUdeviceptr y, CUdeviceptr x, float a, float b, unsigned n)
{
    void *args[] = {&y, &x, &a, &b, &n};
    launch_1d(k_axpby, n, args);
}

static void im2col1d(CUdeviceptr src, unsigned lds, CUdeviceptr dst, unsigned l, unsigned cin,
                     unsigned k, unsigned dil, unsigned pad, unsigned kpad)
{
    void *args[] = {&src, &lds, &dst, &l, &cin, &k, &dil, &pad, &kpad};
    launch_1d(k_im2col, (size_t)l * kpad, args);
}

static void col2im_t(CUdeviceptr z, unsigned ldz, CUdeviceptr b, CUdeviceptr y, unsigned ldy,
                     unsigned lin, unsigned lout, unsigned cout, unsigned k, unsigned r,
                     unsigned pad)
{
    void *args[] = {&z, &ldz, &b, &y, &ldy, &lin, &lout, &cout, &k, &r, &pad};
    launch_1d(k_col2im, (size_t)lout * cout, args);
}

static void aa_snake(CUdeviceptr x, CUdeviceptr y, unsigned ld, unsigned l, unsigned c,
                     CUdeviceptr alpha, CUdeviceptr beta, CUdeviceptr up, CUdeviceptr dn)
{
    void *args[] = {&x, &y, &ld, &l, &c, &alpha, &beta, &up, &dn};
    launch_1d(k_snake, (size_t)l * c, args);
}

enum { G_F32 = 0, G_F16 = 1, G_ACC = 4, G_GATED = 5 };

/* C[M][ldc] (mode G_*) = alpha * A[M][K] . B[N][K]^T + bias; N % 128 == 0,
 * K % 32 == 0, A readable up to M rounded up to 128 rows */
static void gemm_f16(CUdeviceptr a, CUdeviceptr b, CUdeviceptr bias, CUdeviceptr c, unsigned m,
                     unsigned n, unsigned k, unsigned ldc, unsigned mode, CUdeviceptr gate,
                     CUdeviceptr seg, unsigned gld, float alpha)
{
    if (n % 128 || k % 32) {
        fprintf(stderr, "internal error: gemm_f16 %u x %u x %u\n", m, n, k);
        exit(1);
    }
    void *args[] = {&a, &b, &bias, &c, &m, &n, &k, &ldc, &mode, &gate, &seg, &gld, &alpha};
    CHECK(cuLaunchKernel(k_gemm, n / 128, (m + 127) / 128, 1, 256, 1, 1, 0, NULL, args, NULL));
}

/* multi-head attention straight from a fused projection (f16 [n][ld]):
 * head h's q at qoff + h*hstride, its k / v at koff / voff + (h/group)*hstride */
static void attention(unsigned d, CUdeviceptr qkv, CUdeviceptr o, unsigned n, unsigned heads,
                      unsigned ld, unsigned ldo, unsigned qoff, unsigned koff, unsigned voff,
                      unsigned hstride, unsigned group, unsigned causal)
{
    float scale = 1.0f / sqrtf((float)d);
    void *args[] = {&qkv, &o, &n, &ld, &ldo, &qoff, &koff, &voff, &hstride, &group, &scale, &causal};
    CHECK(cuLaunchKernel(d == 128 ? k_attn128 : k_attn64, (n + 63) / 64, heads, 1, 128, 1, 1, 0,
                         NULL, args, NULL));
}

/* ------------------------------------------------------------------------ */
/* GPU memory                                                                */

/* Every network's GPU memory is carved out of one cuMemAlloc'ed arena (made
 * as large as the free memory allows), every piece rounded up to 256 bytes;
 * arena_reset frees it all at once between networks. */
static CUdeviceptr arena;
static size_t arena_size, arena_used;

static size_t round256(size_t n)
{
    return (n + 255) & ~(size_t)255;
}

static void arena_init(void)
{
    size_t free_b, total_b;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    arena_size = free_b > SPARE_BYTES ? free_b - SPARE_BYTES : free_b / 2;
    while (cuMemAlloc(&arena, arena_size) != CUDA_SUCCESS)
        arena_size -= 64u << 20;
    arena_used = 0;
}

static size_t arena_free(void)
{
    return arena_size - arena_used;
}

static CUdeviceptr dalloc(size_t n)
{
    if (arena_used + round256(n) > arena_size) {
        fprintf(stderr, "out of GPU memory: %zu bytes wanted, %zu free\n", n, arena_free());
        exit(1);
    }
    CUdeviceptr p = arena + arena_used;
    arena_used += round256(n);
    return p;
}

static CUdeviceptr dzeros(size_t n)
{
    CUdeviceptr p = dalloc(n);
    CHECK(cuMemsetD8(p, 0, n));
    return p;
}

static CUdeviceptr upload(const void *p, size_t n)
{
    CUdeviceptr d = dalloc(n);
    CHECK(cuMemcpyHtoD(d, p, n));
    return d;
}

static float *download(CUdeviceptr d, size_t n_floats)
{
    float *h = xmalloc(n_floats * 4);
    CHECK(cuMemcpyDtoH(h, d, n_floats * 4));
    return h;
}

static float *download_f16(CUdeviceptr d, size_t n)
{
    uint16_t *h = xmalloc(n * 2);
    float *f = xmalloc(n * 4);
    CHECK(cuMemcpyDtoH(h, d, n * 2));
    for (size_t i = 0; i < n; i++)
        f[i] = half_to_float(h[i]);
    free(h);
    return f;
}

static CUdeviceptr upload_f16(const float *x, size_t n)
{
    uint16_t *h = xmalloc(n * 2);
    for (size_t i = 0; i < n; i++)
        h[i] = float_to_half(x[i]);
    CUdeviceptr d = upload(h, n * 2);
    free(h);
    return d;
}

/* a float tensor of any of the file types as f32 */
static float *to_f32(uint32_t type, const uint8_t *src, size_t n)
{
    float *y = xmalloc(n * 4);
    if (!dequantize_row(type, src, y, n)) {
        fprintf(stderr, "internal error: tensor type %u not convertible\n", type);
        exit(1);
    }
    return y;
}

/* ------------------------------------------------------------------------ */
/* Matrices                                                                  */

/* A weight matrix [n][k] as stored: Q4_K / Q6_K blocks, or f16 values. In a
 * streamed block, w is an offset into the block's slot (see Stream). */
typedef struct {
    uint32_t type; /* GGML_Q4_K, GGML_Q6_K or GGML_F16 */
    CUdeviceptr w;
    unsigned n, k;
} Mat;

static CUdeviceptr scratch; /* SCRATCH_BYTES of f16 rows of a quantized matrix */
static int emulate_q8;      /* MINIMAXH3_EMU_Q8: experiment, see README */

static size_t block_bytes(uint32_t type)
{
    return type == GGML_Q4_K ? 144 : 210;
}

/* C = A . W^T (+ bias) for M rows of A f16 [M][K], through the epilogue mode
 * (gate / seg / gld for G_GATED); a quantized W is expanded to f16 in
 * slices of rows, each followed by its part of the product. base: the slot
 * of a streamed block, or 0. */
static void matmul(const Mat *w, CUdeviceptr base, CUdeviceptr a, unsigned m, CUdeviceptr c,
                   unsigned ldc, unsigned mode, CUdeviceptr bias, CUdeviceptr gate,
                   CUdeviceptr seg, unsigned gld, float alpha)
{
    CUdeviceptr wp = base + w->w;
    size_t es = mode == G_F16 ? 2 : 4;
    if (w->type == GGML_F16) {
        gemm_f16(a, wp, bias, c, m, w->n, w->k, ldc, mode, gate, seg, gld, alpha);
        return;
    }
    if (emulate_q8) { /* debugging: round A to q8_1 blocks as ggml's MMQ path does */
        size_t n = (size_t)m * w->k;
        uint16_t *h = xmalloc(n * 2);
        CHECK(cuMemcpyDtoH(h, a, n * 2));
        for (size_t b = 0; b < n; b += 32) {
            float v[32], amax = 0;
            for (int i = 0; i < 32; i++) {
                v[i] = half_to_float(h[b + i]);
                if (fabsf(v[i]) > amax)
                    amax = fabsf(v[i]);
            }
            float d = amax / 127, id = d ? 1 / d : 0;
            for (int i = 0; i < 32; i++)
                h[b + i] = float_to_half(roundf(v[i] * id) * d);
        }
        CHECK(cuMemcpyHtoD(a, h, n * 2));
        free(h);
    }
    unsigned rows = (unsigned)(SCRATCH_BYTES / ((size_t)w->k * 2)) / 128 * 128;
    size_t row_bytes = (size_t)(w->k / 256) * block_bytes(w->type);
    for (unsigned n0 = 0; n0 < w->n; n0 += rows) {
        unsigned ns = w->n - n0 < rows ? w->n - n0 : rows;
        dequant(w->type, wp + n0 * row_bytes, scratch, (size_t)ns * (w->k / 256));
        gemm_f16(a, scratch, bias ? bias + n0 * 4 : 0, c + n0 * es, m, ns, w->k, ldc, mode,
                 gate ? gate + n0 * 4 : 0, seg, gld, alpha);
    }
}

/* ------------------------------------------------------------------------ */
/* Kernel self-tests against straightforward CPU versions                   */

static int report(const char *name, double err, double tol)
{
    int ok = err <= tol;
    printf("  %-34s max rel err %.2e  %s\n", name, err, ok ? "ok" : "FAIL");
    return ok;
}

/* max |got - ref| / max(|ref|) */
static double rel_err(const float *got, const double *ref, size_t n)
{
    double e = 0, s = 1e-30;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(got[i] - ref[i]);
        if (!(d <= e))
            e = isnan(d) ? INFINITY : d;
        if (fabs(ref[i]) > s)
            s = fabs(ref[i]);
    }
    return e / s;
}

static float *rand_vec(size_t n, float lo, float hi)
{
    float *v = xmalloc(n * 4);
    for (size_t i = 0; i < n; i++)
        v[i] = rng_uniform(lo, hi);
    return v;
}

/* values rounded through f16, so the CPU sees what the GPU sees */
static float *rand_half_vec(size_t n, float lo, float hi)
{
    float *v = rand_vec(n, lo, hi);
    for (size_t i = 0; i < n; i++)
        v[i] = half_to_float(float_to_half(v[i]));
    return v;
}

static void put_half(uint8_t *p, float v)
{
    uint16_t h = float_to_half(v);
    memcpy(p, &h, 2);
}

static int test_dequant(uint32_t type)
{
    const size_t nb = 300, bb = block_bytes(type);
    uint8_t *blk = xmalloc(nb * bb);
    for (size_t i = 0; i < nb * bb; i++)
        blk[i] = (uint8_t)rng_u32();
    for (size_t b = 0; b < nb; b++) { /* sane scales (no inf / nan) */
        uint8_t *p = blk + b * bb;
        if (type == GGML_Q4_K) {
            put_half(p, rng_uniform(-0.01f, 0.01f));
            put_half(p + 2, rng_uniform(-0.01f, 0.01f));
        } else {
            put_half(p + 208, rng_uniform(-0.01f, 0.01f));
        }
    }
    float *ref32 = xmalloc(nb * 256 * 4);
    dequantize_row(type, blk, ref32, nb * 256);
    double *ref = xmalloc(nb * 256 * 8);
    for (size_t i = 0; i < nb * 256; i++)
        ref[i] = half_to_float(float_to_half(ref32[i]));
    size_t used = arena_used;
    CUdeviceptr src = upload(blk, nb * bb), dst = dalloc(nb * 256 * 2);
    dequant(type, src, dst, nb);
    float *got = download_f16(dst, nb * 256);
    arena_used = used;
    int ok = report(type == GGML_Q4_K ? "dequant_q4k" : "dequant_q6k", rel_err(got, ref, nb * 256),
                    0);
    free(blk), free(ref32), free(ref), free(got);
    return ok;
}

static int test_gemm(unsigned mode, unsigned m, unsigned n, unsigned k, float alpha)
{
    float *a = rand_half_vec((size_t)m * k, -1, 1), *b = rand_half_vec((size_t)n * k, -1, 1);
    float *bias = rand_vec(n, -1, 1), *c0 = rand_vec((size_t)m * n, -1, 1);
    const unsigned S = 3, gld = n + 32;
    float *gate = rand_vec((size_t)S * gld, -2, 2);
    uint8_t *seg = xmalloc(m);
    for (unsigned i = 0; i < m; i++)
        seg[i] = (uint8_t)(rng_u32() % S);
    double *ref = xmalloc((size_t)m * n * 8);
    for (unsigned i = 0; i < m; i++)
        for (unsigned j = 0; j < n; j++) {
            double s = 0;
            for (unsigned l = 0; l < k; l++)
                s += (double)a[(size_t)i * k + l] * b[(size_t)j * k + l];
            double v = s * alpha + bias[j];
            double c = c0[(size_t)i * n + j];
            ref[(size_t)i * n + j] = mode == G_ACC     ? c + v
                                     : mode == G_GATED ? c + v * gate[seg[i] * gld + j]
                                                       : v;
        }
    size_t used = arena_used;
    unsigned mp = (m + 127) / 128 * 128;
    CUdeviceptr da = dzeros((size_t)mp * k * 2);
    float *apad = xmalloc((size_t)m * k * 4);
    memcpy(apad, a, (size_t)m * k * 4);
    CUdeviceptr ta = upload_f16(apad, (size_t)m * k);
    CHECK(cuMemcpyDtoD(da, ta, (size_t)m * k * 2));
    CUdeviceptr db = upload_f16(b, (size_t)n * k), dbias = upload(bias, n * 4);
    CUdeviceptr dc = upload(c0, (size_t)m * n * 4), dg = upload(gate, (size_t)S * gld * 4);
    CUdeviceptr ds = upload(seg, m);
    gemm_f16(da, db, dbias, dc, m, n, k, n, mode, dg, ds, gld, alpha);
    float *got = mode == G_F16 ? download_f16(dc, (size_t)m * n) : download(dc, (size_t)m * n);
    arena_used = used;
    char name[64];
    snprintf(name, sizeof(name), "gemm_f16 mode %u (%ux%ux%u)", mode, m, n, k);
    int ok = report(name, rel_err(got, ref, (size_t)m * n), mode == G_F16 ? 2e-3 : 1e-5);
    free(a), free(b), free(bias), free(c0), free(gate), free(seg), free(ref), free(got), free(apad);
    return ok;
}

/* attention over a fused [n][ld] projection laid out like the callers' */
static int test_attention(unsigned d, unsigned n, unsigned heads, unsigned group,
                          unsigned causal, int interleaved)
{
    unsigned kvh = heads / group, ld, qoff, koff, voff, hstride;
    if (interleaved) { /* [h][q k v] per head (the video VAE) */
        hstride = 3 * d, qoff = 0, koff = d, voff = 2 * d, ld = heads * 3 * d;
    } else {           /* [q heads | k heads | v heads] */
        hstride = d, qoff = 0, koff = heads * d, voff = koff + kvh * d, ld = voff + kvh * d;
    }
    float *x = rand_half_vec((size_t)n * ld, -2, 2);
    const unsigned ldo = heads * d;
    double *ref = xmalloc((size_t)n * ldo * 8), *p = xmalloc(n * 8);
    double scale = 1.0 / sqrt((double)d);
    for (unsigned h = 0; h < heads; h++) {
        unsigned kh = h / group;
        for (unsigned i = 0; i < n; i++) {
            const float *q = x + (size_t)i * ld + qoff + h * hstride;
            double mx = -1e300, sum = 0;
            unsigned nk = causal ? i + 1 : n;
            for (unsigned j = 0; j < nk; j++) {
                const float *kk = x + (size_t)j * ld + koff + kh * hstride;
                double s = 0;
                for (unsigned l = 0; l < d; l++)
                    s += (double)q[l] * kk[l];
                p[j] = s * scale;
                if (p[j] > mx)
                    mx = p[j];
            }
            for (unsigned j = 0; j < nk; j++)
                sum += p[j] = exp(p[j] - mx);
            for (unsigned l = 0; l < d; l++) {
                double o = 0;
                for (unsigned j = 0; j < nk; j++)
                    o += p[j] * x[(size_t)j * ld + voff + kh * hstride + l];
                ref[(size_t)i * ldo + h * d + l] = o / sum;
            }
        }
    }
    size_t used = arena_used;
    CUdeviceptr dx = upload_f16(x, (size_t)n * ld), dout = dalloc((size_t)n * ldo * 2);
    attention(d, dx, dout, n, heads, ld, ldo, qoff, koff, voff, hstride, group, causal);
    float *got = download_f16(dout, (size_t)n * ldo);
    arena_used = used;
    char name[64];
    snprintf(name, sizeof(name), "attn_d%u n=%u g=%u%s%s", d, n, group, causal ? " causal" : "",
             interleaved ? " interleaved" : "");
    int ok = report(name, rel_err(got, ref, (size_t)n * ldo), 4e-3);
    free(x), free(ref), free(p), free(got);
    return ok;
}

static int test_rms_mod(void)
{
    const unsigned rows = 37, n = 256, S = 4, ms = n + 64;
    float *x = rand_vec((size_t)rows * n, -3, 3), *w = rand_vec(n, 0.5f, 1.5f);
    float *sh = rand_vec((size_t)S * ms, -1, 1), *sc = rand_vec((size_t)S * ms, -1, 1);
    uint8_t seg[37];
    for (unsigned i = 0; i < rows; i++)
        seg[i] = (uint8_t)(rng_u32() % S);
    const float eps = 1e-5f, os = 0.25f;
    double *ref = xmalloc((size_t)rows * n * 8), *ref16 = xmalloc((size_t)rows * n * 8);
    for (unsigned r = 0; r < rows; r++) {
        double ss = 0;
        for (unsigned j = 0; j < n; j++)
            ss += (double)x[r * n + j] * x[r * n + j];
        double inv = 1 / sqrt(ss / n + eps);
        for (unsigned j = 0; j < n; j++) {
            double v = x[r * n + j] * inv * w[j];
            v = v + v * sc[seg[r] * ms + j] + sh[seg[r] * ms + j];
            ref[r * n + j] = v;
            ref16[r * n + j] = v * os;
        }
    }
    size_t used = arena_used;
    CUdeviceptr dx = upload(x, (size_t)rows * n * 4), dw = upload(w, n * 4);
    CUdeviceptr dsh = upload(sh, (size_t)S * ms * 4), dsc = upload(sc, (size_t)S * ms * 4);
    CUdeviceptr dseg = upload(seg, rows), y16 = dalloc((size_t)rows * n * 2);
    CUdeviceptr y32 = dalloc((size_t)rows * n * 4);
    rms_mod(dx, dw, dsh, dsc, ms, dseg, y16, y32, rows, n, eps, os);
    float *g16 = download_f16(y16, (size_t)rows * n), *g32 = download(y32, (size_t)rows * n);
    int ok = report("rms_mod (f32 out)", rel_err(g32, ref, (size_t)rows * n), 1e-6);
    ok &= report("rms_mod (f16 out)", rel_err(g16, ref16, (size_t)rows * n), 1e-3);
    /* plain norm: no weight, no modulation */
    for (unsigned r = 0; r < rows; r++) {
        double ss = 0;
        for (unsigned j = 0; j < n; j++)
            ss += (double)x[r * n + j] * x[r * n + j];
        for (unsigned j = 0; j < n; j++)
            ref[r * n + j] = x[r * n + j] / sqrt(ss / n + eps);
    }
    free(g32);
    rms_mod(dx, 0, 0, 0, 0, 0, 0, y32, rows, n, eps, 1);
    g32 = download(y32, (size_t)rows * n);
    ok &= report("rms_mod (plain)", rel_err(g32, ref, (size_t)rows * n), 1e-6);
    /* layernorm */
    float *b = rand_vec(n, -1, 1);
    for (unsigned r = 0; r < rows; r++) {
        double mu = 0, var = 0;
        for (unsigned j = 0; j < n; j++)
            mu += x[r * n + j];
        mu /= n;
        for (unsigned j = 0; j < n; j++)
            var += (x[r * n + j] - mu) * (x[r * n + j] - mu);
        var /= n;
        for (unsigned j = 0; j < n; j++)
            ref[r * n + j] = (x[r * n + j] - mu) / sqrt(var + eps) * w[j] + b[j];
    }
    CUdeviceptr db = upload(b, n * 4);
    layernorm(dx, dw, db, y16, rows, n, eps);
    free(g16);
    g16 = download_f16(y16, (size_t)rows * n);
    ok &= report("layernorm", rel_err(g16, ref, (size_t)rows * n), 1e-3);
    arena_used = used;
    free(x), free(w), free(sh), free(sc), free(ref), free(ref16), free(g16), free(g32), free(b);
    return ok;
}

static int test_qk_rope(unsigned d, unsigned rot, int with_w)
{
    const unsigned rows = 21, nq = 3, nk = 2, hs = d + 16, qoff = 8, koff = qoff + nq * hs;
    const unsigned ld = koff + nk * hs + 8;
    const float eps = 1e-5f;
    float *x = rand_half_vec((size_t)rows * ld, -2, 2);
    float *qw = rand_vec(d, 0.5f, 1.5f), *kw = rand_vec(d, 0.5f, 1.5f);
    float *cs = xmalloc((size_t)rows * (rot / 2) * 2 * 4);
    for (size_t i = 0; i < (size_t)rows * (rot / 2); i++) {
        float a = rng_uniform(-10, 10);
        cs[2 * i] = cosf(a), cs[2 * i + 1] = sinf(a);
    }
    double *ref = xmalloc((size_t)rows * ld * 8), v[128];
    for (size_t i = 0; i < (size_t)rows * ld; i++)
        ref[i] = x[i];
    for (unsigned r = 0; r < rows; r++)
        for (unsigned h = 0; h < nq + nk; h++) {
            unsigned col = h < nq ? qoff + h * hs : koff + (h - nq) * hs;
            const float *wt = h < nq ? qw : kw;
            double ss = 0;
            for (unsigned i = 0; i < d; i++)
                ss += (double)x[r * ld + col + i] * x[r * ld + col + i];
            double inv = 1 / sqrt(ss / d + eps);
            for (unsigned i = 0; i < d; i++)
                v[i] = x[r * ld + col + i] * inv * (with_w ? wt[i] : 1);
            for (unsigned i = 0; i < d; i++)
                ref[r * ld + col + i] = v[i];
            for (unsigned i = 0; i < rot / 2; i++) {
                double c = cs[(r * (rot / 2) + i) * 2], s = cs[(r * (rot / 2) + i) * 2 + 1];
                ref[r * ld + col + i] = v[i] * c - v[i + rot / 2] * s;
                ref[r * ld + col + i + rot / 2] = v[i] * s + v[i + rot / 2] * c;
            }
        }
    size_t used = arena_used;
    CUdeviceptr dx = upload_f16(x, (size_t)rows * ld);
    CUdeviceptr dqw = with_w ? upload(qw, d * 4) : 0, dkw = with_w ? upload(kw, d * 4) : 0;
    CUdeviceptr dcs = upload(cs, (size_t)rows * (rot / 2) * 2 * 4);
    qk_norm_rope(dx, ld, rows, nq, nk, qoff, koff, hs, d, dqw, dkw, eps, dcs, rot);
    float *got = download_f16(dx, (size_t)rows * ld);
    arena_used = used;
    char name[64];
    snprintf(name, sizeof(name), "qk_norm_rope d=%u rot=%u%s", d, rot, with_w ? "" : " (no w)");
    int ok = report(name, rel_err(got, ref, (size_t)rows * ld), 2e-3);
    free(x), free(qw), free(kw), free(cs), free(ref), free(got);
    return ok;
}

static int test_small_kernels(void)
{
    int ok = 1;
    size_t used = arena_used;
    { /* swiglu */
        const unsigned rows = 9, f = 70, ldi = 2 * f + 6, ldo = f + 2;
        const float si = 128, so = 1.0f / 128;
        float *in = rand_half_vec((size_t)rows * ldi, -0.05f, 0.05f);
        double *ref = xmalloc((size_t)rows * ldo * 8);
        for (size_t i = 0; i < (size_t)rows * ldo; i++)
            ref[i] = 0;
        for (unsigned r = 0; r < rows; r++)
            for (unsigned j = 0; j < f; j++) {
                double a = in[r * ldi + j] * si, b = in[r * ldi + f + j] * si;
                ref[r * ldo + j] = a / (1 + exp(-a)) * b * so;
            }
        CUdeviceptr din = upload_f16(in, (size_t)rows * ldi), dout = dzeros((size_t)rows * ldo * 2);
        swiglu(din, ldi, dout, ldo, rows, f, si, so);
        float *got = download_f16(dout, (size_t)rows * ldo);
        ok &= report("swiglu", rel_err(got, ref, (size_t)rows * ldo), 2e-3);
        free(in), free(ref), free(got);
    }
    { /* adaln */
        const unsigned L = 3, J = 200, R = 4;
        float *w = rand_half_vec((size_t)L * J * 8, -1, 1), *b = rand_half_vec((size_t)L * J, -1, 1);
        float *t = rand_vec(R * 8, -1, 1);
        double *ref = xmalloc((size_t)L * R * J * 8);
        for (unsigned l = 0; l < L; l++)
            for (unsigned r = 0; r < R; r++)
                for (unsigned j = 0; j < J; j++) {
                    double s = b[l * J + j];
                    for (unsigned i = 0; i < 8; i++)
                        s += (double)w[((size_t)l * J + j) * 8 + i] * t[r * 8 + i];
                    ref[((size_t)l * R + r) * J + j] = s;
                }
        CUdeviceptr dw = upload_f16(w, (size_t)L * J * 8), db = upload_f16(b, (size_t)L * J);
        CUdeviceptr dt = upload(t, R * 32), dout = dalloc((size_t)L * R * J * 4);
        adaln(dw, db, dt, dout, J, L, R);
        float *got = download(dout, (size_t)L * R * J);
        ok &= report("adaln", rel_err(got, ref, (size_t)L * R * J), 1e-6);
        free(w), free(b), free(t), free(ref), free(got);
    }
    { /* linear_f32 (=, +=) */
        const unsigned m = 13, n = 50, k = 96, ldx = 100, ldy = 60;
        float *x = rand_vec((size_t)m * ldx, -1, 1), *w = rand_vec((size_t)n * k, -1, 1);
        float *b = rand_vec(n, -1, 1), *y0 = rand_vec((size_t)m * ldy, -1, 1);
        double *ref = xmalloc((size_t)m * ldy * 8), *ref2 = xmalloc((size_t)m * ldy * 8);
        for (size_t i = 0; i < (size_t)m * ldy; i++)
            ref[i] = ref2[i] = y0[i];
        for (unsigned i = 0; i < m; i++)
            for (unsigned j = 0; j < n; j++) {
                double s = b[j];
                for (unsigned l = 0; l < k; l++)
                    s += (double)x[i * ldx + l] * w[j * k + l];
                ref[i * ldy + j] = s;
                ref2[i * ldy + j] = s + s - b[j] + (y0[i * ldy + j] - y0[i * ldy + j]);
            }
        CUdeviceptr dx = upload(x, (size_t)m * ldx * 4), dw = upload(w, (size_t)n * k * 4);
        CUdeviceptr db = upload(b, n * 4), dy = upload(y0, (size_t)m * ldy * 4);
        linear_f32(dx, ldx, dw, db, dy, ldy, m, n, k, 0);
        float *got = download(dy, (size_t)m * ldy);
        ok &= report("linear_f32", rel_err(got, ref, (size_t)m * ldy), 1e-6);
        linear_f32(dx, ldx, dw, 0, dy, ldy, m, n, k, 1); /* += without bias */
        free(got);
        got = download(dy, (size_t)m * ldy);
        ok &= report("linear_f32 (acc)", rel_err(got, ref2, (size_t)m * ldy), 1e-6);
        free(x), free(w), free(b), free(y0), free(ref), free(ref2), free(got);
    }
    { /* cast_f16, axpby */
        const unsigned rows = 5, cols = 40, ldx = 48, ldy = 44;
        float *x = rand_vec((size_t)rows * ldx, -3, 3);
        double *ref = xmalloc((size_t)rows * ldy * 8);
        for (size_t i = 0; i < (size_t)rows * ldy; i++)
            ref[i] = 0;
        for (unsigned r = 0; r < rows; r++)
            for (unsigned c = 0; c < cols; c++)
                ref[r * ldy + c] = x[r * ldx + c] * 0.5;
        CUdeviceptr dx = upload(x, (size_t)rows * ldx * 4), dy = dzeros((size_t)rows * ldy * 2);
        cast_f16(dx, ldx, dy, ldy, rows, cols, 0.5f);
        float *got = download_f16(dy, (size_t)rows * ldy);
        ok &= report("cast_f16", rel_err(got, ref, (size_t)rows * ldy), 1e-3);
        free(got);
        float *y = rand_vec(1000, -1, 1), *xx = rand_vec(1000, -1, 1);
        double *r2 = xmalloc(1000 * 8);
        for (int i = 0; i < 1000; i++)
            r2[i] = 0.7 * y[i] - 1.3 * xx[i];
        CUdeviceptr d1 = upload(y, 4000), d2 = upload(xx, 4000);
        axpby(d1, d2, 0.7f, -1.3f, 1000);
        got = download(d1, 1000);
        ok &= report("axpby", rel_err(got, r2, 1000), 1e-6);
        free(x), free(ref), free(got), free(y), free(xx), free(r2);
    }
    { /* im2col1d */
        const unsigned L = 50, cin = 6, K = 7, dil = 3, pad = 9, kpad = 64, lds = 8;
        float *src = rand_vec((size_t)L * lds, -1, 1);
        double *ref = xmalloc((size_t)L * kpad * 8);
        for (unsigned t = 0; t < L; t++)
            for (unsigned c = 0; c < kpad; c++) {
                double v = 0;
                if (c < cin * K) {
                    int s = (int)t - (int)pad + (int)(c % K) * (int)dil;
                    if (s >= 0 && s < (int)L)
                        v = half_to_float(float_to_half(src[s * lds + c / K]));
                }
                ref[t * kpad + c] = v;
            }
        CUdeviceptr ds = upload(src, (size_t)L * lds * 4), dd = dalloc((size_t)L * kpad * 2);
        im2col1d(ds, lds, dd, L, cin, K, dil, pad, kpad);
        float *got = download_f16(dd, (size_t)L * kpad);
        ok &= report("im2col1d", rel_err(got, ref, (size_t)L * kpad), 0);
        free(src), free(ref), free(got);
    }
    { /* col2im_t: the scatter of a transposed conv (stride 5, K 9, pad 2) */
        const unsigned lin = 11, cout = 6, K = 9, r = 5, pad = 2, ldz = cout * K + 2, ldy = 8;
        const unsigned lout = (lin - 1) * r + K - 2 * pad;
        float *z = rand_vec((size_t)lin * ldz, -1, 1), *b = rand_vec(cout, -1, 1);
        double *ref = xmalloc((size_t)lout * ldy * 8);
        for (size_t i = 0; i < (size_t)lout * ldy; i++)
            ref[i] = 0;
        for (unsigned t = 0; t < lout; t++)
            for (unsigned co = 0; co < cout; co++) {
                double s = b[co];
                for (unsigned si = 0; si < lin; si++)
                    for (unsigned k = 0; k < K; k++)
                        if (si * r + k == t + pad)
                            s += z[si * ldz + co * K + k];
                ref[t * ldy + co] = s;
            }
        CUdeviceptr dz = upload(z, (size_t)lin * ldz * 4), db = upload(b, cout * 4);
        CUdeviceptr dy = dzeros((size_t)lout * ldy * 4);
        col2im_t(dz, ldz, db, dy, ldy, lin, lout, cout, K, r, pad);
        float *got = download(dy, (size_t)lout * ldy);
        ok &= report("col2im_t", rel_err(got, ref, (size_t)lout * ldy), 1e-6);
        free(z), free(b), free(ref), free(got);
    }
    { /* aa_snake: BigVGAN's Activation1D as stable-diffusion.cpp computes it */
        const unsigned L = 40, C = 5, ld = 7;
        float *x = rand_vec((size_t)L * ld, -2, 2);
        float *al = rand_vec(C, -0.5f, 0.5f), *be = rand_vec(C, -0.5f, 0.5f);
        float up[12], dn[12], up2[12];
        for (int i = 0; i < 12; i++) {
            up[i] = rng_uniform(-0.2f, 0.5f), dn[i] = rng_uniform(-0.2f, 0.5f);
            up2[i] = 2 * up[i]; /* the host folds the x2 upsampling gain into the filter */
        }
        double *ref = xmalloc((size_t)L * ld * 8), *u = xmalloc((2 * L + 30) * 8);
        for (size_t i = 0; i < (size_t)L * ld; i++)
            ref[i] = 0;
        for (unsigned c = 0; c < C; c++) {
            /* replicate-pad by 5, transposed conv stride 2, crop 15 each side, x2 */
            unsigned lp = L + 10, lt = (lp - 1) * 2 + 12;
            double *full = xmalloc(lt * 8);
            for (unsigned i = 0; i < lt; i++)
                full[i] = 0;
            for (unsigned i = 0; i < lp; i++) {
                int s = (int)i - 5;
                s = s < 0 ? 0 : s >= (int)L ? (int)L - 1 : s;
                for (unsigned k = 0; k < 12; k++)
                    full[2 * i + k] += x[s * ld + c] * (double)up[k];
            }
            double ea = exp(al[c]), eb = exp(be[c]);
            for (unsigned i = 0; i < 2 * L; i++) {
                double v = full[i + 15] * 2;
                double sn = sin(v * ea);
                u[i] = v + sn * sn / (eb + 1e-9);
            }
            for (unsigned t = 0; t < L; t++) {
                double s = 0;
                for (int j = 0; j < 12; j++) {
                    int idx = 2 * (int)t + j - 5;
                    idx = idx < 0 ? 0 : idx >= (int)(2 * L) ? (int)(2 * L) - 1 : idx;
                    s += dn[j] * u[idx];
                }
                ref[t * ld + c] = s;
            }
            free(full);
        }
        CUdeviceptr dx = upload(x, (size_t)L * ld * 4), dy = dzeros((size_t)L * ld * 4);
        CUdeviceptr da = upload(al, C * 4), dbe = upload(be, C * 4);
        CUdeviceptr du = upload(up2, 48), ddn = upload(dn, 48);
        aa_snake(dx, dy, ld, L, C, da, dbe, du, ddn);
        float *got = download(dy, (size_t)L * ld);
        ok &= report("aa_snake", rel_err(got, ref, (size_t)L * ld), 1e-5);
        free(x), free(al), free(be), free(ref), free(u), free(got);
    }
    arena_used = used;
    return ok;
}

static int self_test(void)
{
    int ok = 1;
    printf("kernel self-tests:\n");
    ok &= test_dequant(GGML_Q4_K);
    ok &= test_dequant(GGML_Q6_K);
    ok &= test_gemm(G_F32, 200, 256, 96, 1);
    ok &= test_gemm(G_F16, 130, 384, 64, 0.125f);
    ok &= test_gemm(G_ACC, 77, 128, 160, 2);
    ok &= test_gemm(G_GATED, 300, 256, 512, 1);
    ok &= test_attention(128, 150, 4, 1, 0, 0);
    ok &= test_attention(128, 100, 8, 4, 1, 0);
    ok &= test_attention(64, 210, 3, 1, 0, 1);
    ok &= test_attention(64, 70, 2, 1, 1, 0);
    ok &= test_rms_mod();
    ok &= test_qk_rope(128, 96, 1);
    ok &= test_qk_rope(128, 128, 1);
    ok &= test_qk_rope(64, 48, 0);
    ok &= test_small_kernels();
    printf(ok ? "all kernel tests passed\n" : "SOME KERNEL TESTS FAILED\n");
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------------ */
/* Weight blocks: resident in the arena, or streamed through two GPU slots   */

/* The matrices of one transformer block (or text-encoder layer) packed back
 * to back: on the GPU for good, or in host memory (pinned, or the mapped
 * file) and copied into a slot whenever the block runs. */
typedef struct {
    size_t bytes;      /* all matrices, each rounded up to 256 bytes */
    CUdeviceptr dev;   /* resident: where they are; 0 when streamed */
    const uint8_t *host;
} WBlock;

/* Streamed blocks run in a fixed cyclic order (the order of `list`): the
 * copy of the next two is in flight (on a second stream, into the other
 * slots) while one computes. */
typedef struct {
    CUdeviceptr slot[2];
    CUstream copy;
    CUevent ready[2], done[2];
    WBlock **list;
    unsigned n;
    unsigned long long visit;
} Stream;

static void stream_copy(Stream *s, unsigned long long v)
{
    const WBlock *b = s->list[v % s->n];
    unsigned i = v % 2;
    CHECK(cuStreamWaitEvent(s->copy, s->done[i], 0));
    CHECK(cuMemcpyHtoDAsync(s->slot[i], b->host, b->bytes, s->copy));
    CHECK(cuEventRecord(s->ready[i], s->copy));
}

static void stream_init(Stream *s, size_t slot_bytes)
{
    if (!s->n)
        return;
    s->slot[0] = dalloc(slot_bytes);
    s->slot[1] = dalloc(slot_bytes);
    CHECK(cuStreamCreate(&s->copy, CU_STREAM_NON_BLOCKING));
    for (int i = 0; i < 2; i++) {
        CHECK(cuEventCreate(&s->ready[i], CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventCreate(&s->done[i], CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventRecord(s->done[i], NULL));
    }
    s->visit = 0;
    stream_copy(s, 0);
    if (s->n > 1)
        stream_copy(s, 1);
}

/* Where block b's matrices are for this run (the compute stream waits for
 * its copy when it is streamed). */
static CUdeviceptr stream_begin(Stream *s, const WBlock *b)
{
    if (b->dev)
        return b->dev;
    if (s->list[s->visit % s->n] != b) {
        fprintf(stderr, "internal error: streamed blocks out of order\n");
        exit(1);
    }
    CHECK(cuStreamWaitEvent(NULL, s->ready[s->visit % 2], 0));
    return s->slot[s->visit % 2];
}

static void stream_end(Stream *s, const WBlock *b)
{
    if (b->dev)
        return;
    CHECK(cuEventRecord(s->done[s->visit % 2], NULL));
    if (s->n > 1)
        stream_copy(s, s->visit + 2);
    else
        stream_copy(s, s->visit + 1);
    s->visit++;
}

static void stream_free(Stream *s)
{
    if (!s->n)
        return;
    CHECK(cuStreamSynchronize(s->copy));
    CHECK(cuStreamDestroy(s->copy));
    for (int i = 0; i < 2; i++) {
        CHECK(cuEventDestroy(s->ready[i]));
        CHECK(cuEventDestroy(s->done[i]));
    }
    s->n = 0;
}

/* The tensor `name` of any type (n0: expected element count, 0 = any), or exit. */
static const GgufTensor *find_tensor(const Gguf *g, const char *name, size_t n0)
{
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (strcmp(t->name, name))
            continue;
        size_t n = 1;
        for (uint32_t j = 0; j < t->n_dims; j++)
            n *= t->ne[j];
        if (n0 && n != n0) {
            fprintf(stderr, "%s: %zu elements, expected %zu\n", name, n, n0);
            exit(1);
        }
        return t;
    }
    fprintf(stderr, "%s: tensor not found\n", name);
    exit(1);
}

/* Lays out the GGUF matrices `names` (each [n][k] of a supported type) as one
 * block: m[i].w becomes the offset inside it. */
static size_t block_layout(const Gguf *g, const char *const *names, Mat *const *m, int count,
                           const GgufTensor **t)
{
    size_t off = 0;
    for (int i = 0; i < count; i++) {
        t[i] = find_tensor(g, names[i], 0);
        if (t[i]->type != GGML_Q4_K && t[i]->type != GGML_Q6_K) {
            fprintf(stderr, "%s: unsupported matrix type %u\n", names[i], t[i]->type);
            exit(1);
        }
        m[i]->type = t[i]->type;
        m[i]->k = (unsigned)t[i]->ne[0];
        m[i]->n = (unsigned)t[i]->ne[1];
        m[i]->w = off;
        off += round256(t[i]->nbytes);
    }
    return off;
}

/* Fills block b (layout from block_layout): copies the matrices to the GPU
 * (resident) or to pinned host memory (streamed: pinned != 0). */
static void block_fill(WBlock *b, const GgufTensor **t, int count, int resident)
{
    uint8_t *h = NULL;
    if (resident)
        b->dev = dalloc(b->bytes);
    else
        CHECK(cuMemAllocHost((void **)&h, b->bytes));
    size_t off = 0;
    for (int i = 0; i < count; i++) {
        if (resident)
            CHECK(cuMemcpyHtoD(b->dev + off, t[i]->data, t[i]->nbytes));
        else
            memcpy(h + off, t[i]->data, t[i]->nbytes);
        off += round256(t[i]->nbytes);
    }
    b->host = h;
}

/* A float vector of any file type as f32 on the GPU (optionally scaled). */
static CUdeviceptr upload_vec(const Gguf *g, const char *name, size_t n, float scale)
{
    const GgufTensor *t = find_tensor(g, name, n);
    float *v = to_f32(t->type, t->data, n);
    for (size_t i = 0; i < n; i++)
        v[i] *= scale;
    CUdeviceptr d = upload(v, n * 4);
    free(v);
    return d;
}

/* ------------------------------------------------------------------------ */
/* Tensors dumped by stable-diffusion.cpp (SD_DUMP, see README.md):          */
/* i32 n_dims, i64 ne[n_dims] (ne[0] contiguous), f32 data                   */

static float *read_dump(const char *dir, const char *name, int64_t *ne, int *nd_out)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    int32_t nd;
    int64_t shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    size_t n = 1;
    if (fread(&nd, 4, 1, f) != 1 || nd < 1 || nd > 8 || fread(shape, 8, nd, f) != (size_t)nd) {
        fprintf(stderr, "%s: bad dump\n", path);
        exit(1);
    }
    for (int i = 0; i < nd; i++)
        n *= shape[i];
    float *v = xmalloc(n * 4);
    if (fread(v, 4, n, f) != n) {
        fprintf(stderr, "%s: truncated\n", path);
        exit(1);
    }
    fclose(f);
    if (ne)
        for (int i = 0; i < 8; i++)
            ne[i] = shape[i];
    if (nd_out)
        *nd_out = nd;
    return v;
}

/* relative errors of got vs ref: max |d| / max |ref| and rms(d) / rms(ref) */
static void compare(const char *what, const float *got, const float *ref, size_t n, double *maxe,
                    double *rmse)
{
    double md = 0, mr = 1e-30, sd = 0, sr = 1e-30;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - ref[i]);
        if (!(d <= md))
            md = isnan(d) ? INFINITY : d;
        if (fabs(ref[i]) > mr)
            mr = fabs(ref[i]);
        sd += d * d, sr += (double)ref[i] * ref[i];
    }
    *maxe = md / mr;
    *rmse = sqrt(sd / sr);
    printf("  %-28s max rel err %.2e  rms rel err %.2e\n", what, *maxe, *rmse);
}

/* ------------------------------------------------------------------------ */
/* Diffusion transformer                                                     */

#define DIT_HIDDEN 5376
#define DIT_HEADS 56
#define DIT_FFN 14336
#define DIT_LAYERS 50
#define DIT_REFINER 2
#define DIT_TEXT 5120
#define DIT_ROT 96         /* RoPE on 96 of the 128 dims: 16 frequencies x (t, h, w) */
#define DIT_MODS (6 * 3 * DIT_HIDDEN) /* per timestep: 3 modalities x 6 vectors */
#define VIDEO_C 24
#define AUDIO_C 32
#define VIDEO_SHIFT 12.0f   /* sigma shift of the video schedule (the sampler's) */
#define AUDIO_SHIFT 3.0f    /* ... and of the audio one */
#define FRAME_RESCALE (5.0f / 3.0f)

typedef struct {
    Mat qkv, out, fc1, fc2;
    CUdeviceptr qn, kn, n1, n2; /* f32 norm weights */
    WBlock wb;
} DitBlock;

/* The packed sequence of one sampling run: [text | audio L | audio R | video] */
typedef struct {
    unsigned text, audio_t, lt, lh, lw; /* text tokens, audio latents, latent t/h/w */
    unsigned n, n_audio, n_video, a0, v0; /* tokens; audio / video token counts and starts */
    float *pos;                           /* [n][3] (t, h, w) RoPE positions */
} Layout;

typedef struct {
    Gguf g;
    DitBlock ref[DIT_REFINER], blk[DIT_LAYERS];
    Stream st;
    CUdeviceptr cond_w_bias, ref_final, fin_norm;
    Mat cond;
    WBlock cond_wb;
    CUdeviceptr ada_w, ada_b, fada_w, fada_b; /* f16 AdaLN matrices (all blocks), final layer's */
    CUdeviceptr vp_w, vp_b, ap_w, ap_b, vo_w, vo_b, ao_w, ao_b; /* f32 patch embeds and heads */
    float inv_freq[16];
    float *t_table; /* adaln_t_table [1025][8] */
    unsigned t_grid;
    /* activations for up to cap tokens (A-matrix buffers have mp >= cap rows) */
    unsigned cap, mp;
    CUdeviceptr x, y16, qkv, o, h1, g16, seg, cs, temb, mods, fmods, y32, vout, aout, rows_v,
        rows_a, ctx32, text;
} Dit;

static void dit_block_layout(const Gguf *g, const char *prefix, DitBlock *b, const GgufTensor **t)
{
    char nm[4][160];
    const char *names[4] = {nm[0], nm[1], nm[2], nm[3]};
    snprintf(nm[0], 160, "%s.attn.qkv_proj.weight", prefix);
    snprintf(nm[1], 160, "%s.attn.out_proj.weight", prefix);
    snprintf(nm[2], 160, "%s.mlp.fc1.weight", prefix);
    snprintf(nm[3], 160, "%s.mlp.fc2.weight", prefix);
    Mat *m[4] = {&b->qkv, &b->out, &b->fc1, &b->fc2};
    b->wb.bytes = block_layout(g, names, m, 4, t);
}

static void dit_block_vectors(const Gguf *g, const char *prefix, DitBlock *b)
{
    char nm[160];
    snprintf(nm, 160, "%s.attn.q_norm.weight", prefix);
    b->qn = upload_vec(g, nm, 128, 1);
    snprintf(nm, 160, "%s.attn.k_norm.weight", prefix);
    b->kn = upload_vec(g, nm, 128, 1);
    snprintf(nm, 160, "%s.norm1.weight", prefix);
    b->n1 = upload_vec(g, nm, DIT_HIDDEN, 1);
    snprintf(nm, 160, "%s.norm2.weight", prefix);
    b->n2 = upload_vec(g, nm, DIT_HIDDEN, 1);
}

/* f16 copy of a stacked set of F16 GGUF tensors */
static CUdeviceptr upload_f16_tensors(const Gguf *g, const char *fmt, int count, size_t n)
{
    CUdeviceptr d = dalloc((size_t)count * n * 2);
    for (int i = 0; i < count; i++) {
        char nm[160];
        snprintf(nm, 160, fmt, i);
        const GgufTensor *t = gguf_tensor(g, nm, GGML_F16, 0, 0, 0);
        if (t->nbytes != n * 2) {
            fprintf(stderr, "%s: unexpected size\n", nm);
            exit(1);
        }
        CHECK(cuMemcpyHtoD(d + (size_t)i * n * 2, t->data, n * 2));
    }
    return d;
}

static void dit_load(Dit *d, const char *path, unsigned cap)
{
    double t0 = now_ms();
    gguf_open(&d->g, path);
    const Gguf *g = &d->g;
    const GgufTensor *t[4];
    char pre[64];

    /* small tensors first: they are always resident */
    d->ada_w = upload_f16_tensors(g, "blocks.%d.adaln_proj.linear.weight", DIT_LAYERS,
                                  (size_t)DIT_MODS * 8);
    d->ada_b = upload_f16_tensors(g, "blocks.%d.adaln_proj.linear.bias", DIT_LAYERS, DIT_MODS);
    d->fada_w = upload_f16_tensors(g, "final_layer.adaln_proj.linear.weight", 1,
                                   (size_t)2 * DIT_HIDDEN * 8);
    d->fada_b = upload_f16_tensors(g, "final_layer.adaln_proj.linear.bias", 1, 2 * DIT_HIDDEN);
    d->vp_w = upload_vec(g, "video_patch_proj.weight", (size_t)DIT_HIDDEN * VIDEO_C * 4, 1);
    d->vp_b = upload_vec(g, "video_patch_proj.bias", DIT_HIDDEN, 1);
    d->ap_w = upload_vec(g, "audio_patch_proj.weight", (size_t)DIT_HIDDEN * AUDIO_C, 1);
    d->ap_b = upload_vec(g, "audio_patch_proj.bias", DIT_HIDDEN, 1);
    d->vo_w = upload_vec(g, "final_layer.video_out.weight", (size_t)VIDEO_C * 4 * DIT_HIDDEN, 1);
    d->vo_b = upload_vec(g, "final_layer.video_out.bias", VIDEO_C * 4, 1);
    d->ao_w = upload_vec(g, "final_layer.audio_out.weight", (size_t)AUDIO_C * DIT_HIDDEN, 1);
    d->ao_b = upload_vec(g, "final_layer.audio_out.bias", AUDIO_C, 1);
    d->fin_norm = upload_vec(g, "final_layer.norm.weight", DIT_HIDDEN, 1);
    d->ref_final = upload_vec(g, "token_refiner.final_norm.weight", DIT_HIDDEN, 1);
    d->cond_w_bias = upload_vec(g, "condition_proj.bias", DIT_HIDDEN, 1);
    const GgufTensor *ft = gguf_tensor(g, "rope.inv_freq", GGML_F32, 16, 0, 0);
    memcpy(d->inv_freq, ft->data, 64);
    const GgufTensor *tt = gguf_tensor(g, "adaln_t_table", GGML_F32, 8, 0, 0);
    d->t_grid = (unsigned)tt->ne[1];
    d->t_table = xmalloc(tt->nbytes);
    memcpy(d->t_table, tt->data, tt->nbytes);
    for (int i = 0; i < DIT_REFINER; i++) {
        snprintf(pre, sizeof(pre), "token_refiner.blocks.%d", i);
        dit_block_vectors(g, pre, &d->ref[i]);
    }
    for (int i = 0; i < DIT_LAYERS; i++) {
        snprintf(pre, sizeof(pre), "blocks.%d", i);
        dit_block_vectors(g, pre, &d->blk[i]);
    }

    /* activations */
    d->cap = cap;
    d->mp = (cap + 127) / 128 * 128;
    size_t mp = d->mp;
    d->x = dzeros(mp * DIT_HIDDEN * 4);
    d->y16 = dzeros(mp * DIT_HIDDEN * 2);
    d->qkv = dzeros(mp * 3 * DIT_HEADS * 128 * 2);
    d->o = dzeros(mp * DIT_HEADS * 128 * 2);
    d->h1 = dzeros(mp * 2 * DIT_FFN * 2);
    d->g16 = dzeros(mp * DIT_FFN * 2);
    d->seg = dzeros(mp);
    d->cs = dzeros(mp * DIT_ROT * 4);
    d->temb = dzeros(4 * 8 * 4);
    d->mods = dalloc((size_t)DIT_LAYERS * 4 * DIT_MODS * 4);
    d->fmods = dalloc((size_t)4 * 2 * DIT_HIDDEN * 4);
    d->y32 = dalloc(mp * DIT_HIDDEN * 4);
    d->vout = dalloc(mp * VIDEO_C * 4 * 4);
    d->aout = dalloc(mp * AUDIO_C * 4);
    d->rows_v = dalloc(mp * VIDEO_C * 4 * 4);
    d->rows_a = dalloc(mp * AUDIO_C * 4);
    d->ctx32 = dzeros(mp * DIT_TEXT * 4);
    d->text = dalloc(mp * DIT_HIDDEN * 4);
    scratch = dalloc(SCRATCH_BYTES);

    /* the condition projection and the refiner: resident (they run once) */
    {
        Mat *m[1] = {&d->cond};
        const char *names[1] = {"condition_proj.weight"};
        d->cond_wb.bytes = block_layout(g, names, m, 1, t);
        block_fill(&d->cond_wb, t, 1, 1);
    }
    for (int i = 0; i < DIT_REFINER; i++) {
        snprintf(pre, sizeof(pre), "token_refiner.blocks.%d", i);
        dit_block_layout(g, pre, &d->ref[i], t);
        block_fill(&d->ref[i].wb, t, 4, 1);
    }

    /* the 50 blocks: as many resident as fit, the rest streamed */
    size_t bb = 0;
    for (int i = 0; i < DIT_LAYERS; i++) {
        snprintf(pre, sizeof(pre), "blocks.%d", i);
        dit_block_layout(g, pre, &d->blk[i], t);
        if (d->blk[i].wb.bytes > bb)
            bb = d->blk[i].wb.bytes;
    }
    size_t avail = arena_free();
    unsigned n_res = DIT_LAYERS;
    if ((size_t)DIT_LAYERS * bb > avail)
        n_res = avail > 2 * bb ? (unsigned)((avail - 2 * bb) / bb) : 0;
    const char *force = getenv("MINIMAXH3_STREAM"); /* testing: stream at least N blocks */
    if (force && DIT_LAYERS - n_res < (unsigned)atoi(force))
        n_res = DIT_LAYERS - (unsigned)atoi(force);
    d->st.n = DIT_LAYERS - n_res;
    d->st.list = xmalloc(sizeof(WBlock *) * DIT_LAYERS);
    for (int i = 0; i < DIT_LAYERS; i++) {
        snprintf(pre, sizeof(pre), "blocks.%d", i);
        dit_block_layout(g, pre, &d->blk[i], t);
        int resident = i < (int)n_res;
        block_fill(&d->blk[i].wb, t, 4, resident);
        if (!resident)
            d->st.list[i - n_res] = &d->blk[i].wb;
    }
    stream_init(&d->st, bb);
    fprintf(stderr, "diffusion model: %u of %d blocks resident, %u streamed (%.1f s)\n", n_res,
            DIT_LAYERS, d->st.n, (now_ms() - t0) / 1e3);
}

static void dit_free(Dit *d)
{
    stream_free(&d->st);
    for (int i = 0; i < DIT_LAYERS; i++)
        if (d->blk[i].wb.host)
            CHECK(cuMemFreeHost((void *)d->blk[i].wb.host));
    free(d->st.list);
    free(d->t_table);
    gguf_close(&d->g);
    arena_used = 0;
}

static float video_span(unsigned frame)
{
    static const int spans[5] = {1, 4, 4, 4, 4};
    return FRAME_RESCALE * spans[frame % 5];
}

/* stable-diffusion.cpp's build_layout for a prompt without reference media */
static void make_layout(Layout *L, unsigned text, unsigned audio_t, unsigned lt, unsigned lh,
                        unsigned lw)
{
    L->text = text, L->audio_t = audio_t, L->lt = lt, L->lh = lh, L->lw = lw;
    L->n_audio = 2 * audio_t;
    L->n_video = lt * (lh / 2) * (lw / 2);
    L->a0 = text;
    L->v0 = text + L->n_audio;
    L->n = L->v0 + L->n_video;
    L->pos = xmalloc((size_t)L->n * 3 * 4);
    float sqrt_area = sqrtf((float)(lh * lw));
    unsigned ch = lh / 2, cw = lw / 2;
    float ha[64], wa[64];
    float rh = (float)lh / sqrt_area, rw = (float)lw / sqrt_area;
    for (unsigned i = 0; i < ch; i++)
        ha[i] = ((float)i * (rh / ch) + (1.f - rh) * 0.5f) * 32.f;
    for (unsigned i = 0; i < cw; i++)
        wa[i] = ((float)i * (rw / cw) + (1.f - rw) * 0.5f) * 32.f;
    float *p = L->pos;
    for (unsigned i = 0; i < text; i++)
        *p++ = (float)i, *p++ = 0, *p++ = 0;
    float cursor = (float)text;
    for (int c = 0; c < 2; c++)
        for (unsigned t = 0; t < audio_t; t++)
            *p++ = cursor + (float)t, *p++ = 0, *p++ = c ? wa[cw - 1] : wa[0];
    for (unsigned t = 0; t < lt; t++) {
        for (unsigned h = 0; h < ch; h++)
            for (unsigned w = 0; w < cw; w++)
                *p++ = cursor, *p++ = ha[h], *p++ = wa[w];
        cursor += video_span(t);
    }
}

static float time_shift_sigma(float sigma, float from_shift, float to_shift)
{
    float base = sigma / (from_shift + sigma * (1.f - from_shift));
    return to_shift * base / (1.f + (to_shift - 1.f) * base);
}

static int find_or_add(float *v, int *n, float x)
{
    for (int i = 0; i < *n; i++)
        if (v[i] == x)
            return i;
    v[*n] = x;
    return (*n)++;
}

/* Refines the text condition (f32 [text][5120] on the host) once per run:
 * condition_proj, the 2 refiner blocks, the final norm -> d->text. */
static void dit_refine(Dit *d, const float *ctx, unsigned text)
{
    const unsigned H = DIT_HIDDEN, QKV = 3 * DIT_HEADS * 128, O = DIT_HEADS * 128;
    CHECK(cuMemcpyHtoD(d->ctx32, ctx, (size_t)text * DIT_TEXT * 4));
    cast_f16(d->ctx32, DIT_TEXT, d->h1, DIT_TEXT, text, DIT_TEXT, 1);
    matmul(&d->cond, d->cond_wb.dev, d->h1, text, d->text, H, G_F32, d->cond_w_bias, 0, 0, 0, 1);
    for (int i = 0; i < DIT_REFINER; i++) {
        const DitBlock *b = &d->ref[i];
        CUdeviceptr base = b->wb.dev;
        rms_mod(d->text, b->n1, 0, 0, 0, 0, d->y16, 0, text, H, 1e-5f, 1);
        matmul(&b->qkv, base, d->y16, text, d->qkv, QKV, G_F16, 0, 0, 0, 0, 1);
        qk_norm_rope(d->qkv, QKV, text, DIT_HEADS, DIT_HEADS, 0, O, 128, 128, b->qn, b->kn, 1e-5f,
                     0, 0);
        attention(128, d->qkv, d->o, text, DIT_HEADS, QKV, O, 0, O, 2 * O, 128, 1, 0);
        matmul(&b->out, base, d->o, text, d->text, H, G_ACC, 0, 0, 0, 0, 1);
        rms_mod(d->text, b->n2, 0, 0, 0, 0, d->y16, 0, text, H, 1e-5f, 1);
        matmul(&b->fc1, base, d->y16, text, d->h1, 2 * DIT_FFN, G_F16, 0, 0, 0, 0, 1.f / 128);
        swiglu(d->h1, 2 * DIT_FFN, d->g16, DIT_FFN, text, DIT_FFN, 128, 1.f / 128);
        matmul(&b->fc2, base, d->g16, text, d->text, H, G_ACC, 0, 0, 0, 0, 128);
    }
    rms_mod(d->text, d->ref_final, 0, 0, 0, 0, 0, d->text, text, H, 1e-5f, 1);
}

/* One evaluation of the network at sigma (packed latent x as the sampler
 * keeps it: [24 video channels | audio carrier] of [lt][lh][lw] values each,
 * see make_layout) -> out (same layout), the velocity in the sampler's
 * terms. The text must have been refined (dit_refine). */
static void dit_forward(Dit *d, const Layout *L, float sigma_in, const float *x, float *out)
{
    const unsigned H = DIT_HIDDEN, QKV = 3 * DIT_HEADS * 128, O = DIT_HEADS * 128;
    const unsigned n = L->n, sp = L->lt * L->lh * L->lw, lh = L->lh, lw = L->lw;
    const unsigned at = L->audio_t;
    float sigma_v = sigma_in < 1e-6f ? 1e-6f : sigma_in > 1 ? 1 : sigma_in;
    float sigma_a = time_shift_sigma(sigma_v, VIDEO_SHIFT, AUDIO_SHIFT);
    float audio_scale = VIDEO_SHIFT / AUDIO_SHIFT;
    float t_v = 1.f - sigma_v, t_a = 1.f - sigma_a;

    /* timestep rows: video, audio, then the condition rows (unused here
     * but present in the reference, which keeps the row numbering) */
    float ts[4];
    int nts = 0;
    int vrow = find_or_add(ts, &nts, t_v), arow = find_or_add(ts, &nts, t_a);
    find_or_add(ts, &nts, t_v > 0.999f ? t_v : 0.999f);
    find_or_add(ts, &nts, t_a > 1.f ? t_a : 1.f);
    float temb[4 * 8];
    for (int r = 0; r < nts; r++) { /* adaln_t_table, linearly interpolated */
        float tc = ts[r] < 0 ? 0 : ts[r] > 1 ? 1 : ts[r];
        float pos = tc * (d->t_grid - 1);
        int i = (int)floorf(pos);
        if (i > (int)d->t_grid - 2)
            i = d->t_grid - 2;
        float f = pos - i;
        for (int j = 0; j < 8; j++) {
            float lo = d->t_table[i * 8 + j], hi = d->t_table[(i + 1) * 8 + j];
            temb[r * 8 + j] = lo + (hi - lo) * f;
        }
    }
    CHECK(cuMemcpyHtoD(d->temb, temb, nts * 32));
    adaln(d->ada_w, d->ada_b, d->temb, d->mods, DIT_MODS, DIT_LAYERS, nts);
    adaln(d->fada_w, d->fada_b, d->temb, d->fmods, 2 * H, 1, nts);

    /* modulation row per token: timestep row * 3 + modality (0 video, 1 text,
     * 2 audio) */
    uint8_t *seg = xmalloc(n);
    for (unsigned i = 0; i < n; i++)
        seg[i] = (uint8_t)(i < L->a0 ? vrow * 3 + 1 : i < L->v0 ? arow * 3 + 2 : vrow * 3);
    CHECK(cuMemcpyHtoD(d->seg, seg, n));
    free(seg);

    /* RoPE angles of every token */
    float *cs = xmalloc((size_t)n * DIT_ROT * 4);
    for (unsigned i = 0; i < n; i++)
        for (int a = 0; a < 3; a++)
            for (int j = 0; j < 16; j++) {
                float ang = L->pos[i * 3 + a] * d->inv_freq[j];
                cs[((size_t)i * 48 + a * 16 + j) * 2] = (float)cos((double)ang);
                cs[((size_t)i * 48 + a * 16 + j) * 2 + 1] = (float)sin((double)ang);
            }
    CHECK(cuMemcpyHtoD(d->cs, cs, (size_t)n * DIT_ROT * 4));
    free(cs);

    /* patch rows: video (c, 2x2) per patch, audio [channel][t] x 32 */
    float *rv = xmalloc((size_t)L->n_video * VIDEO_C * 4 * 4);
    for (unsigned t = 0, p = 0; t < L->lt; t++)
        for (unsigned hh = 0; hh < lh / 2; hh++)
            for (unsigned ww = 0; ww < lw / 2; ww++, p++)
                for (unsigned c = 0; c < VIDEO_C; c++)
                    for (unsigned q = 0; q < 4; q++)
                        rv[(size_t)p * 96 + c * 4 + q] =
                            x[(size_t)c * sp + ((size_t)t * lh + 2 * hh + q / 2) * lw + 2 * ww + q % 2];
    const float *carrier = x + (size_t)VIDEO_C * sp;
    float *audio = xmalloc((size_t)at * 2 * AUDIO_C * 4); /* x_a = carrier * sigma_a / sigma_v */
    for (size_t i = 0; i < (size_t)at * 2 * AUDIO_C; i++)
        audio[i] = carrier[i] * (sigma_a / sigma_v);
    float *ra = xmalloc((size_t)L->n_audio * AUDIO_C * 4);
    for (unsigned lr = 0; lr < 2; lr++)
        for (unsigned t = 0; t < at; t++)
            for (unsigned c = 0; c < AUDIO_C; c++)
                ra[((size_t)lr * at + t) * AUDIO_C + c] = audio[t + at * lr + 2 * at * c];
    CHECK(cuMemcpyHtoD(d->rows_v, rv, (size_t)L->n_video * 96 * 4));
    CHECK(cuMemcpyHtoD(d->rows_a, ra, (size_t)L->n_audio * AUDIO_C * 4));
    free(rv), free(ra);

    /* the sequence: refined text | audio embeds | video embeds */
    CHECK(cuMemcpyDtoD(d->x, d->text, (size_t)L->text * H * 4));
    linear_f32(d->rows_a, AUDIO_C, d->ap_w, d->ap_b, d->x + (size_t)L->a0 * H * 4, H, L->n_audio,
               H, AUDIO_C, 0);
    linear_f32(d->rows_v, 96, d->vp_w, d->vp_b, d->x + (size_t)L->v0 * H * 4, H, L->n_video, H,
               96, 0);

    for (int l = 0; l < DIT_LAYERS; l++) {
        const DitBlock *b = &d->blk[l];
        CUdeviceptr base = stream_begin(&d->st, &b->wb);
        CUdeviceptr m = d->mods + (size_t)l * nts * DIT_MODS * 4; /* [row][3][6][H] */
        const unsigned ms = 6 * H;
        rms_mod(d->x, b->n1, m, m + H * 4, ms, d->seg, d->y16, 0, n, H, 1e-5f, 1);
        matmul(&b->qkv, base, d->y16, n, d->qkv, QKV, G_F16, 0, 0, 0, 0, 1);
        qk_norm_rope(d->qkv, QKV, n, DIT_HEADS, DIT_HEADS, 0, O, 128, 128, b->qn, b->kn, 1e-5f,
                     d->cs, DIT_ROT);
        attention(128, d->qkv, d->o, n, DIT_HEADS, QKV, O, 0, O, 2 * O, 128, 1, 0);
        matmul(&b->out, base, d->o, n, d->x, H, G_GATED, 0, m + 2 * H * 4, d->seg, ms, 1);
        rms_mod(d->x, b->n2, m + 3 * H * 4, m + 4 * H * 4, ms, d->seg, d->y16, 0, n, H, 1e-5f, 1);
        matmul(&b->fc1, base, d->y16, n, d->h1, 2 * DIT_FFN, G_F16, 0, 0, 0, 0, 1.f / 128);
        swiglu(d->h1, 2 * DIT_FFN, d->g16, DIT_FFN, n, DIT_FFN, 128, 1.f / 128);
        matmul(&b->fc2, base, d->g16, n, d->x, H, G_GATED, 0, m + 5 * H * 4, d->seg, ms, 128);
        stream_end(&d->st, &b->wb);
    }

    /* final layer: norm, the (shift, scale) of the segment's timestep, heads */
    CUdeviceptr fv = d->fmods + (size_t)vrow * 2 * H * 4, fa = d->fmods + (size_t)arow * 2 * H * 4;
    rms_mod(d->x + (size_t)L->v0 * H * 4, d->fin_norm, fv, fv + H * 4, 0, 0, 0, d->y32,
            L->n_video, H, 1e-5f, 1);
    linear_f32(d->y32, H, d->vo_w, d->vo_b, d->vout, 96, L->n_video, 96, H, 0);
    rms_mod(d->x + (size_t)L->a0 * H * 4, d->fin_norm, fa, fa + H * 4, 0, 0, 0, d->y32,
            L->n_audio, H, 1e-5f, 1);
    linear_f32(d->y32, H, d->ao_w, d->ao_b, d->aout, AUDIO_C, L->n_audio, AUDIO_C, H, 0);
    float *vo = download(d->vout, (size_t)L->n_video * 96);
    float *ao = download(d->aout, (size_t)L->n_audio * AUDIO_C);

    /* back to the packed latent layout; the network predicts -velocity */
    for (unsigned t = 0, p = 0; t < L->lt; t++)
        for (unsigned hh = 0; hh < lh / 2; hh++)
            for (unsigned ww = 0; ww < lw / 2; ww++, p++)
                for (unsigned c = 0; c < VIDEO_C; c++)
                    for (unsigned q = 0; q < 4; q++)
                        out[(size_t)c * sp + ((size_t)t * lh + 2 * hh + q / 2) * lw + 2 * ww + q % 2] =
                            -vo[(size_t)p * 96 + c * 4 + q];
    /* audio velocity -> d(carrier)/d(sigma_v) */
    float *oa = out + (size_t)VIDEO_C * sp;
    for (unsigned lr = 0; lr < 2; lr++)
        for (unsigned t = 0; t < at; t++)
            for (unsigned c = 0; c < AUDIO_C; c++) {
                size_t i = t + (size_t)at * lr + 2 * (size_t)at * c;
                float v = -ao[((size_t)lr * at + t) * AUDIO_C + c];
                oa[i] = audio[i] * (1.f - audio_scale) + v * (1.f + (audio_scale - 1.f) * sigma_a);
            }
    free(vo), free(ao), free(audio);
}

/* The packed latent has room for 24 video channels and then the audio
 * carrier (audio_t * 2 * 32 values) padded to whole channels. */
static size_t packed_size(unsigned lt, unsigned lh, unsigned lw, unsigned audio_t)
{
    size_t sp = (size_t)lt * lh * lw, a = (size_t)audio_t * 2 * AUDIO_C;
    return (VIDEO_C + (a + sp - 1) / sp) * sp;
}

/* ------------------------------------------------------------------------ */
/* Text encoder: Qwen3-VL-32B's language model, 50 layers, no final norm     */

#define TE_DIM 5120
#define TE_LAYERS 50
#define TE_HEADS 64
#define TE_KV 8
#define TE_FFN 25600
#define TE_QKV ((TE_HEADS + 2 * TE_KV) * 128)
#define TE_THETA 5000000.0

/* Hidden states after the last layer for T tokens -> ctx f32 [T][5120]. The
 * layers stream through two GPU slots straight from the mapped file: layer
 * l + 1 is copied (on the copy stream) while layer l computes. */
static void te_encode(const char *path, const int32_t *tok, unsigned T, float *ctx)
{
    double t0 = now_ms();
    Gguf g;
    gguf_open(&g, path);
    const GgufTensor *emb = gguf_tensor(&g, "model.embed_tokens.weight", GGML_BF16, TE_DIM, 0, 0);
    const unsigned mp = (T + 127) / 128 * 128;
    CUdeviceptr h = dzeros((size_t)mp * TE_DIM * 4), y16 = dzeros((size_t)mp * TE_DIM * 2);
    CUdeviceptr qkv = dzeros((size_t)mp * TE_QKV * 2), o = dzeros((size_t)mp * TE_HEADS * 128 * 2);
    CUdeviceptr gu = dzeros((size_t)mp * 2 * TE_FFN * 2), g16 = dzeros((size_t)mp * TE_FFN * 2);
    CUdeviceptr cs = dalloc((size_t)mp * 128 * 4);
    scratch = dalloc(SCRATCH_BYTES);

    float *x = xmalloc((size_t)T * TE_DIM * 4);
    for (unsigned i = 0; i < T; i++) {
        if (tok[i] < 0 || (uint64_t)tok[i] >= emb->ne[1]) {
            fprintf(stderr, "token id %d out of range\n", tok[i]);
            exit(1);
        }
        const uint16_t *row = (const uint16_t *)emb->data + (size_t)tok[i] * TE_DIM;
        for (unsigned j = 0; j < TE_DIM; j++)
            x[(size_t)i * TE_DIM + j] = bf16_to_float(row[j]);
    }
    CHECK(cuMemcpyHtoD(h, x, (size_t)T * TE_DIM * 4));
    /* rotate-half RoPE over all 128 dims (text only: the multimodal
     * sections all see the same position) */
    float *c = xmalloc((size_t)T * 128 * 4);
    for (unsigned p = 0; p < T; p++)
        for (int i = 0; i < 64; i++) {
            double a = p * pow(TE_THETA, -2.0 * i / 128);
            c[(p * 64 + i) * 2] = (float)cos(a), c[(p * 64 + i) * 2 + 1] = (float)sin(a);
        }
    CHECK(cuMemcpyHtoD(cs, c, (size_t)T * 128 * 4));
    free(c);

    /* the layers: per-layer weights packed into a slot */
    static const char *const mats[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                        "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
                                        "mlp.down_proj"};
    char nm[7][96];
    const char *names[7];
    for (int i = 0; i < 7; i++)
        names[i] = nm[i];
    Mat m[TE_LAYERS][7];
    WBlock wb[TE_LAYERS];
    const GgufTensor *t[7];
    size_t bb = 0;
    for (int l = 0; l < TE_LAYERS; l++) {
        Mat *mp7[7];
        for (int i = 0; i < 7; i++) {
            snprintf(nm[i], 96, "model.layers.%d.%s.weight", l, mats[i]);
            mp7[i] = &m[l][i];
        }
        memset(&wb[l], 0, sizeof(wb[l]));
        wb[l].bytes = block_layout(&g, names, mp7, 7, t);
        if (wb[l].bytes > bb)
            bb = wb[l].bytes;
    }
    CUdeviceptr vec = dalloc((size_t)TE_LAYERS * (2 * TE_DIM + 256) * 4);
    for (int l = 0; l < TE_LAYERS; l++) {
        char vn[96];
        float *v;
        CUdeviceptr base = vec + (size_t)l * (2 * TE_DIM + 256) * 4;
        snprintf(vn, 96, "model.layers.%d.input_layernorm.weight", l);
        v = to_f32(GGML_BF16, find_tensor(&g, vn, TE_DIM)->data, TE_DIM);
        CHECK(cuMemcpyHtoD(base, v, TE_DIM * 4));
        free(v);
        snprintf(vn, 96, "model.layers.%d.post_attention_layernorm.weight", l);
        v = to_f32(GGML_BF16, find_tensor(&g, vn, TE_DIM)->data, TE_DIM);
        CHECK(cuMemcpyHtoD(base + TE_DIM * 4, v, TE_DIM * 4));
        free(v);
        snprintf(vn, 96, "model.layers.%d.self_attn.q_norm.weight", l);
        v = to_f32(GGML_BF16, find_tensor(&g, vn, 128)->data, 128);
        CHECK(cuMemcpyHtoD(base + 2 * TE_DIM * 4, v, 512));
        free(v);
        snprintf(vn, 96, "model.layers.%d.self_attn.k_norm.weight", l);
        v = to_f32(GGML_BF16, find_tensor(&g, vn, 128)->data, 128);
        CHECK(cuMemcpyHtoD(base + (2 * TE_DIM + 128) * 4, v, 512));
        free(v);
    }
    /* as many layers resident as fit (they are still read from the file
     * once); the rest go through the slots */
    size_t avail = arena_free();
    unsigned n_res = (size_t)TE_LAYERS * bb <= avail ? TE_LAYERS
                     : avail > 2 * bb               ? (unsigned)((avail - 2 * bb) / bb)
                                                    : 0;
    CUdeviceptr slot[2] = {0, 0};
    for (int l = 0; l < (int)n_res; l++)
        wb[l].dev = dalloc(wb[l].bytes);
    if (n_res < TE_LAYERS)
        slot[0] = dalloc(bb), slot[1] = dalloc(bb);
    CUstream cp;
    CUevent ready[2], done[2];
    CHECK(cuStreamCreate(&cp, CU_STREAM_NON_BLOCKING));
    for (int i = 0; i < 2; i++) {
        CHECK(cuEventCreate(&ready[i], CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventCreate(&done[i], CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventRecord(done[i], NULL));
    }
    /* layer l's copy goes to its own memory or slot l % 2; pageable source,
     * so the async copies run in order on the copy stream */
    for (int l = 0; l <= TE_LAYERS; l++) {
        if (l < TE_LAYERS) {
            for (int i = 0; i < 7; i++) {
                snprintf(nm[i], 96, "model.layers.%d.%s.weight", l, mats[i]);
                t[i] = find_tensor(&g, nm[i], 0);
            }
            CUdeviceptr dst = wb[l].dev ? wb[l].dev : slot[l % 2];
            if (!wb[l].dev)
                CHECK(cuStreamWaitEvent(cp, done[l % 2], 0));
            for (int i = 0; i < 7; i++)
                CHECK(cuMemcpyHtoDAsync(dst + m[l][i].w, t[i]->data, t[i]->nbytes, cp));
            CHECK(cuEventRecord(ready[l % 2], cp));
        }
        int k = l - 1; /* compute the previous layer while this one copies */
        if (k < 0)
            continue;
        CUdeviceptr base = wb[k].dev ? wb[k].dev : slot[k % 2];
        CHECK(cuStreamWaitEvent(NULL, ready[k % 2], 0));
        CUdeviceptr v = vec + (size_t)k * (2 * TE_DIM + 256) * 4;
        const Mat *w = m[k];
        rms_mod(h, v, 0, 0, 0, 0, y16, 0, T, TE_DIM, 1e-6f, 1);
        matmul(&w[0], base, y16, T, qkv, TE_QKV, G_F16, 0, 0, 0, 0, 1);
        matmul(&w[1], base, y16, T, qkv + TE_HEADS * 128 * 2, TE_QKV, G_F16, 0, 0, 0, 0, 1);
        matmul(&w[2], base, y16, T, qkv + (TE_HEADS + TE_KV) * 128 * 2, TE_QKV, G_F16, 0, 0, 0, 0,
               1);
        qk_norm_rope(qkv, TE_QKV, T, TE_HEADS, TE_KV, 0, TE_HEADS * 128, 128, 128,
                     v + 2 * TE_DIM * 4, v + (2 * TE_DIM + 128) * 4, 1e-6f, cs, 128);
        attention(128, qkv, o, T, TE_HEADS, TE_QKV, TE_HEADS * 128, 0, TE_HEADS * 128,
                  (TE_HEADS + TE_KV) * 128, 128, TE_HEADS / TE_KV, 1);
        matmul(&w[3], base, o, T, h, TE_DIM, G_ACC, 0, 0, 0, 0, 1);
        rms_mod(h, v + TE_DIM * 4, 0, 0, 0, 0, y16, 0, T, TE_DIM, 1e-6f, 1);
        matmul(&w[4], base, y16, T, gu, 2 * TE_FFN, G_F16, 0, 0, 0, 0, 1.f / 128);
        matmul(&w[5], base, y16, T, gu + TE_FFN * 2, 2 * TE_FFN, G_F16, 0, 0, 0, 0, 1.f / 128);
        swiglu(gu, 2 * TE_FFN, g16, TE_FFN, T, TE_FFN, 128, 1.f / 128);
        matmul(&w[6], base, g16, T, h, TE_DIM, G_ACC, 0, 0, 0, 0, 128);
        if (!wb[k].dev)
            CHECK(cuEventRecord(done[k % 2], NULL));
    }
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(ctx, h, (size_t)T * TE_DIM * 4));
    CHECK(cuStreamDestroy(cp));
    for (int i = 0; i < 2; i++) {
        CHECK(cuEventDestroy(ready[i]));
        CHECK(cuEventDestroy(done[i]));
    }
    free(x);
    gguf_close(&g);
    arena_used = 0;
    fprintf(stderr, "text encoder: %u tokens, %u of %d layers resident (%.1f s)\n", T, n_res,
            TE_LAYERS, (now_ms() - t0) / 1e3);
}

/* stable-diffusion.cpp's default noise ("cuda" RNG): Philox 4x32-10 with
 * counter (offset, 0, i, 0) for element i and key = seed, then Box-Muller
 * on the first two words; offset counts the calls. */
static void philox_randn(uint64_t seed, uint32_t offset, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t c0 = offset, c1 = 0, c2 = (uint32_t)i, c3 = 0;
        uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
        for (int r = 0; r < 10; r++) {
            uint64_t v1 = (uint64_t)c0 * 0xD2511F53u, v2 = (uint64_t)c2 * 0xCD9E8D57u;
            uint32_t n0 = (uint32_t)(v2 >> 32) ^ c1 ^ k0, n2 = (uint32_t)(v1 >> 32) ^ c3 ^ k1;
            c0 = n0, c1 = (uint32_t)v2, c2 = n2, c3 = (uint32_t)v1;
            k0 += 0x9E3779B9u, k1 += 0xBB67AE85u;
        }
        const float inv = 2.3283064e-10f, inv2pi = 2.3283064e-10f * 6.2831855f;
        float u = (float)c0 * inv + inv / 2, v = (float)c1 * inv2pi + inv2pi / 2;
        /* bit-exact with the reference: its sqrt / log / sin are the double
         * C functions, and s is stored as a float in between */
        float s = (float)sqrt(-2.0f * log(u));
        out[i] = (float)(s * sin(v));
    }
}

/* the discrete flow schedule: n timesteps from 999 down to 0, shifted */
static void flow_sigmas(unsigned n, float shift, float *sig)
{
    const int t_max = 999;
    for (unsigned i = 0; i < n; i++) {
        float t = n == 1 ? (float)t_max : t_max - ((float)t_max / (float)(n - 1)) * i;
        float u = (t + 1) / 1000.f;
        sig[i] = shift * u / (1 + (shift - 1) * u);
    }
    sig[n] = 0;
}

/* Euler steps from pure noise x (sigma = 1); the audio part leaves scaled by
 * audio_shift / shift, as the reference's process_latent_out does. */
static void sample(Dit *d, const Layout *L, unsigned steps, float *x, size_t npk, int verbose)
{
    float *sig = xmalloc((steps + 1) * 4), *v = xmalloc(npk * 4);
    flow_sigmas(steps, VIDEO_SHIFT, sig);
    for (unsigned i = 0; i < steps; i++) {
        double t0 = now_ms();
        memset(v, 0, npk * 4);
        /* the network sees t = 1000 sigma, as the reference passes it */
        dit_forward(d, L, sig[i] * 1000.f / 1000.f, x, v);
        float dt = sig[i + 1] - sig[i];
        if (getenv("MINIMAXH3_SAVE")) { /* debugging: the trajectory */
            char fn[1100];
            snprintf(fn, sizeof(fn), "%s/traj_x_%u.f32", getenv("MINIMAXH3_SAVE"), i);
            FILE *f = fopen(fn, "wb");
            if (f)
                fwrite(x, 4, npk, f), fclose(f);
            snprintf(fn, sizeof(fn), "%s/traj_v_%u.f32", getenv("MINIMAXH3_SAVE"), i);
            f = fopen(fn, "wb");
            if (f)
                fwrite(v, 4, npk, f), fclose(f);
        }
        for (size_t j = 0; j < npk; j++)
            x[j] += dt * v[j];
        if (verbose)
            fprintf(stderr, "  step %u/%u  sigma %.4f  (%.0f ms)\n", i + 1, steps, sig[i],
                    now_ms() - t0);
    }
    size_t sp = (size_t)L->lt * L->lh * L->lw;
    for (size_t j = (size_t)VIDEO_C * sp; j < npk; j++)
        x[j] *= AUDIO_SHIFT / VIDEO_SHIFT;
    free(sig), free(v);
}

/* ------------------------------------------------------------------------ */
/* Video VAE decoder: a ViT over [latent tokens | 4 registers | 1 zero]      */

#define VAE_DIM 2048
#define VAE_LAYERS 36
#define VAE_HEADS 32
#define VAE_FF 8192
#define VAE_ROT 48 /* 8 frequencies x (t, h, w) on 48 of the 64 dims */

static const float latents_mean[VIDEO_C] = {
    0.858090341091156f,  -0.960659146308899f, 1.066164016723633f,  -0.509032547473907f,
    -0.272758185863495f, -1.367541432380676f, -0.255325496196747f, -0.269075542688370f,
    -0.537684082984924f, -0.046409729868174f, 0.665737032890320f,  0.196901276707649f,
    -0.546060800552368f, -0.403534203767776f, -0.236830249428749f, 0.259284526109695f,
    -0.301339447498322f, 0.211341992020607f,  -1.120684862136841f, 0.358193337917328f,
    -0.042251437902451f, 0.260482996702194f,  0.228640928864479f,  0.705603182315826f};
static const float latents_std[VIDEO_C] = {
    1.222377419471741f, 1.276726365089417f, 1.683177471160889f, 1.754945516586304f,
    1.563621640205383f, 2.194143533706665f, 0.965313792228699f, 1.056988596916199f,
    0.841948926448822f, 0.772995293140411f, 1.895593762397766f, 0.946841835975647f,
    0.799680948257446f, 0.449889004230499f, 0.719739973545075f, 0.693629324436188f,
    2.961095094680786f, 2.769419908523560f, 3.049618482589722f, 2.108805418014527f,
    3.276226282119751f, 3.162735700607300f, 2.281681299209595f, 2.612784385681153f};
static const float pixel_mean[3] = {0.485f, 0.456f, 0.406f}, pixel_std[3] = {0.229f, 0.224f, 0.225f};

typedef struct {
    Mat qkv, out, w1, w2;
    CUdeviceptr qkv_b, out_b, w1_b, w2_b, n1, n2, s1, s2;
} VaeBlock;

typedef struct {
    SafeTensors st;
    VaeBlock blk[VAE_LAYERS];
    Mat proj;
    CUdeviceptr xe_w, xe_b, norm_w, norm_b, proj_b;
    float reg[4 * VAE_DIM], pq_w[VIDEO_C * VIDEO_C], pq_b[VIDEO_C];
    unsigned cap, mp;
    CUdeviceptr x, y16, qkv, o, h1, g16, cs, tok, out;
} Vae;

static float *st_f32(const SafeTensors *s, const char *name, size_t n)
{
    const StTensor *t = st_get(s, name, ST_F16, n);
    float *v = xmalloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, t->data + 2 * i, 2);
        v[i] = half_to_float(h);
    }
    return v;
}

static CUdeviceptr st_vec(const SafeTensors *s, const char *name, size_t n, float scale)
{
    float *v = st_f32(s, name, n);
    for (size_t i = 0; i < n; i++)
        v[i] *= scale;
    CUdeviceptr d = upload(v, n * 4);
    free(v);
    return d;
}

static Mat st_mat(const SafeTensors *s, const char *name, unsigned n, unsigned k)
{
    const StTensor *t = st_get(s, name, ST_F16, (size_t)n * k);
    Mat m = {GGML_F16, upload(t->data, t->nbytes), n, k};
    return m;
}

/* cap: tokens of one decoder chunk */
static void vae_load(Vae *v, const char *path, unsigned cap)
{
    double t0 = now_ms();
    st_open(&v->st, path);
    const SafeTensors *s = &v->st;
    char nm[160];
    for (int i = 0; i < VAE_LAYERS; i++) {
        VaeBlock *b = &v->blk[i];
#define P(x) (snprintf(nm, sizeof(nm), "decoder.transformer_blocks.%d.%s", i, x), nm)
        b->qkv = st_mat(s, P("attn.to_qkv.weight"), 3 * VAE_DIM, VAE_DIM);
        b->out = st_mat(s, P("attn.to_out.weight"), VAE_DIM, VAE_DIM);
        b->w1 = st_mat(s, P("ff.w1.weight"), 2 * VAE_FF, VAE_DIM);
        b->w2 = st_mat(s, P("ff.w2.weight"), VAE_DIM, VAE_FF);
        b->qkv_b = st_vec(s, P("attn.to_qkv.bias"), 3 * VAE_DIM, 1);
        b->out_b = st_vec(s, P("attn.to_out.bias"), VAE_DIM, 1);
        /* w1 runs with its output scaled by 1/128 (f16 range): so its bias */
        b->w1_b = st_vec(s, P("ff.w1.bias"), 2 * VAE_FF, 1.f / 128);
        b->w2_b = st_vec(s, P("ff.w2.bias"), VAE_DIM, 1);
        b->n1 = st_vec(s, P("norm1.weight"), VAE_DIM, 1);
        b->n2 = st_vec(s, P("norm2.weight"), VAE_DIM, 1);
        b->s1 = st_vec(s, P("scale1"), VAE_DIM, 1);
        b->s2 = st_vec(s, P("scale2"), VAE_DIM, 1);
#undef P
    }
    v->proj = st_mat(s, "decoder.proj_out.weight", 3 * 4 * 16 * 16, VAE_DIM);
    v->proj_b = st_vec(s, "decoder.proj_out.bias", 3 * 4 * 16 * 16, 1);
    v->norm_w = st_vec(s, "decoder.norm_out.weight", VAE_DIM, 1);
    v->norm_b = st_vec(s, "decoder.norm_out.bias", VAE_DIM, 1);
    v->xe_w = st_vec(s, "decoder.x_embedder.weight", (size_t)VAE_DIM * VIDEO_C, 1);
    v->xe_b = st_vec(s, "decoder.x_embedder.bias", VAE_DIM, 1);
    float *r = st_f32(s, "decoder.register_tokens", 4 * VAE_DIM);
    memcpy(v->reg, r, sizeof(v->reg));
    free(r);
    float *pw = st_f32(s, "post_quant_conv.weight", VIDEO_C * VIDEO_C);
    float *pb = st_f32(s, "post_quant_conv.bias", VIDEO_C);
    memcpy(v->pq_w, pw, sizeof(v->pq_w));
    memcpy(v->pq_b, pb, sizeof(v->pq_b));
    free(pw), free(pb);

    v->cap = cap;
    v->mp = (cap + 127) / 128 * 128;
    size_t mp = v->mp;
    v->x = dzeros(mp * VAE_DIM * 4);
    v->y16 = dzeros(mp * VAE_DIM * 2);
    v->qkv = dzeros(mp * 3 * VAE_DIM * 2);
    v->o = dzeros(mp * VAE_DIM * 2);
    v->h1 = dzeros(mp * 2 * VAE_FF * 2);
    v->g16 = dzeros(mp * VAE_FF * 2);
    v->cs = dzeros(mp * VAE_ROT * 4);
    v->tok = dalloc(mp * VIDEO_C * 4);
    v->out = dalloc(mp * 3072 * 4);
    scratch = dalloc(SCRATCH_BYTES);
    fprintf(stderr, "video VAE decoder loaded (%.1f s)\n", (now_ms() - t0) / 1e3);
}

static void vae_free(Vae *v)
{
    st_close(&v->st);
    arena_used = 0;
}

/* Decodes one chunk: z (VAE latents, post-denormalization) [24][T][H][W] ->
 * pixels [3][4T][16H][16W] in [0, 1]. */
static void vae_chunk(Vae *v, const float *z, unsigned T, unsigned H, unsigned W, float *pix)
{
    const unsigned np = T * H * W, n = np + 5, D = VAE_DIM;
    if (n > v->cap) {
        fprintf(stderr, "internal error: VAE chunk of %u tokens (cap %u)\n", n, v->cap);
        exit(1);
    }
    /* post_quant_conv (1x1x1) per token, tokens in (t, h, w) order */
    float *tok = xmalloc((size_t)np * VIDEO_C * 4);
    for (unsigned p = 0; p < np; p++)
        for (unsigned o = 0; o < VIDEO_C; o++) {
            float s = v->pq_b[o];
            for (unsigned c = 0; c < VIDEO_C; c++)
                s += v->pq_w[o * VIDEO_C + c] * z[(size_t)c * np + p];
            tok[(size_t)p * VIDEO_C + o] = s;
        }
    CHECK(cuMemcpyHtoD(v->tok, tok, (size_t)np * VIDEO_C * 4));
    free(tok);
    linear_f32(v->tok, VIDEO_C, v->xe_w, v->xe_b, v->x, D, np, D, VIDEO_C, 0);
    CHECK(cuMemcpyHtoD(v->x + (size_t)np * D * 4, v->reg, sizeof(v->reg)));
    CHECK(cuMemsetD32(v->x + (size_t)(np + 4) * D * 4, 0, D));

    /* RoPE: positions spread over [-2 pi, 2 pi] per axis, theta 100, the
     * 5 extra tokens at the origin */
    float omega[8];
    for (int j = 0; j < 8; j++)
        omega[j] = 1.0f / powf(100.f, (float)j * (14.f / 16.f) / 7.f);
    float *cs = xmalloc((size_t)n * VAE_ROT * 4);
    const float two_pi = 6.28318530717958647692f;
    for (unsigned i = 0; i < n; i++) {
        float pos[3] = {0, 0, 0};
        if (i < np) {
            unsigned t = i / (H * W), h = i / W % H, w = i % W;
            pos[0] = (2.f * ((t + 0.5f) / T) - 1.f) * two_pi;
            pos[1] = (2.f * ((h + 0.5f) / H) - 1.f) * two_pi;
            pos[2] = (2.f * ((w + 0.5f) / W) - 1.f) * two_pi;
        }
        for (int a = 0; a < 3; a++)
            for (int j = 0; j < 8; j++) {
                float ang = pos[a] * omega[j];
                cs[((size_t)i * 24 + a * 8 + j) * 2] = cosf(ang);
                cs[((size_t)i * 24 + a * 8 + j) * 2 + 1] = sinf(ang);
            }
    }
    CHECK(cuMemcpyHtoD(v->cs, cs, (size_t)n * VAE_ROT * 4));
    free(cs);

    for (int l = 0; l < VAE_LAYERS; l++) {
        const VaeBlock *b = &v->blk[l];
        rms_mod(v->x, b->n1, 0, 0, 0, 0, v->y16, 0, n, D, 1e-5f, 1);
        matmul(&b->qkv, 0, v->y16, n, v->qkv, 3 * D, G_F16, b->qkv_b, 0, 0, 0, 1);
        /* per head [q 64 | k 64 | v 64]; q and k RMS-normed without weight */
        qk_norm_rope(v->qkv, 3 * D, n, VAE_HEADS, VAE_HEADS, 0, 64, 192, 64, 0, 0, 1e-5f, v->cs,
                     VAE_ROT);
        attention(64, v->qkv, v->o, n, VAE_HEADS, 3 * D, D, 0, 64, 128, 192, 1, 0);
        matmul(&b->out, 0, v->o, n, v->x, D, G_GATED, b->out_b, b->s1, 0, 0, 1);
        rms_mod(v->x, b->n2, 0, 0, 0, 0, v->y16, 0, n, D, 1e-5f, 1);
        matmul(&b->w1, 0, v->y16, n, v->h1, 2 * VAE_FF, G_F16, b->w1_b, 0, 0, 0, 1.f / 128);
        swiglu(v->h1, 2 * VAE_FF, v->g16, VAE_FF, n, VAE_FF, 128, 1.f / 128);
        matmul(&b->w2, 0, v->g16, n, v->x, D, G_GATED, b->w2_b, b->s2, 0, 0, 128);
    }
    layernorm(v->x, v->norm_w, v->norm_b, v->y16, np, D, 1e-5f);
    matmul(&v->proj, 0, v->y16, np, v->out, 3072, G_F32, v->proj_b, 0, 0, 0, 1);
    float *o = download(v->out, (size_t)np * 3072);
    /* unpatchify: feature c * 1024 + (pt, ph, pw) of a 4 x 16 x 16 block */
    const unsigned FT = 4 * T, FH = 16 * H, FW = 16 * W;
    for (unsigned p = 0; p < np; p++) {
        unsigned t = p / (H * W), h = p / W % H, w = p % W;
        for (unsigned c = 0; c < 3; c++)
            for (unsigned q = 0; q < 1024; q++) {
                unsigned pt = q / 256, ph = q / 16 % 16, pw = q % 16;
                float val = o[(size_t)p * 3072 + c * 1024 + q] * pixel_std[c] + pixel_mean[c];
                val = val < 0 ? 0 : val > 1 ? 1 : val;
                pix[(((size_t)c * FT + 4 * t + pt) * FH + 16 * h + ph) * FW + 16 * w + pw] = val;
            }
    }
    free(o);
}

/* Number of output frames for lt latent frames (stable-diffusion.cpp). */
static unsigned video_frames(unsigned lt)
{
    return lt <= 1 ? 1 : ((lt - 2) / 5) * 17 + 5;
}

/* The whole video: diffusion latents x [24][lt][lh][lw] -> frames
 * [3][F][16 lh][16 lw]. Chunks of 7 latent frames (5 new + 2 overlap),
 * as the reference decodes them: each chunk drops its first 3 frames,
 * keeps 17, and cross-fades 5 more into the next chunk. */
static float *vae_decode(Vae *v, const float *x, unsigned lt, unsigned lh, unsigned lw,
                         unsigned *frames_out)
{
    const unsigned sp = lh * lw;
    unsigned pseudo = lt + 3, pad = (5 - pseudo % 5) % 5;
    pseudo += pad;
    int chunks = (int)pseudo / 5 - 1;
    if (chunks < 1)
        pad += 5, chunks++;
    unsigned T = lt + pad;
    float *z = xmalloc((size_t)VIDEO_C * T * sp * 4); /* VAE latents, last frame repeated */
    for (unsigned c = 0; c < VIDEO_C; c++)
        for (unsigned t = 0; t < T; t++)
            for (unsigned i = 0; i < sp; i++) {
                unsigned ts = t < lt ? t : lt - 1;
                z[((size_t)c * T + t) * sp + i] =
                    x[((size_t)c * lt + ts) * sp + i] * latents_std[c] + latents_mean[c];
            }
    const unsigned FH = 16 * lh, FW = 16 * lw, fpix = FH * FW;
    unsigned cap_frames = 0;
    float *video = NULL, *overlap = NULL;
    unsigned nv = 0, nov = 0, nov_all = 0; /* blend extent, frames kept from the last chunk */
    float *chunk = xmalloc((size_t)VIDEO_C * 7 * sp * 4);
    float *pix = xmalloc((size_t)3 * 28 * fpix * 4);
    for (unsigned start = 0; start + 2 < T; start += 5) {
        unsigned end = start + 7 < T ? start + 7 : T, ct = end - start;
        int last = end == T;
        for (unsigned c = 0; c < VIDEO_C; c++)
            memcpy(chunk + (size_t)c * ct * sp, z + ((size_t)c * T + start) * sp,
                   (size_t)ct * sp * 4);
        vae_chunk(v, chunk, ct, lh, lw, pix);
        unsigned df = 4 * ct, first_end = df < 20 ? df : 20, f0 = 3 < first_end ? 3 : first_end;
        unsigned keep = first_end - f0;
        unsigned extra = df > 23 ? df - 23 : 0;
        if (nv + keep + extra > cap_frames) {
            cap_frames = (nv + keep + extra) * 2;
            video = realloc(video, (size_t)3 * cap_frames * fpix * 4);
        }
        /* append frames [f0, first_end), the first ones cross-faded with the
         * previous chunk's tail (frames are [frame][c][pixels] until the end) */
        for (unsigned f = 0; f < keep; f++)
            for (unsigned c = 0; c < 3; c++) {
                float *dst = video + ((size_t)(nv + f) * 3 + c) * fpix;
                const float *src = pix + ((size_t)c * df + f0 + f) * fpix;
                unsigned ext = nov < keep ? nov : keep;
                if (overlap && f < ext) {
                    float wb = (float)f / ext, wa = 1.f - wb;
                    const float *pv = overlap + ((size_t)(nov_all - ext + f) * 3 + c) * fpix;
                    for (unsigned i = 0; i < fpix; i++)
                        dst[i] = pv[i] * wa + src[i] * wb;
                } else {
                    memcpy(dst, src, (size_t)fpix * 4);
                }
            }
        nv += keep;
        free(overlap);
        overlap = NULL, nov = 0;
        if (extra) {
            nov = extra < 5 ? extra : 5; /* blend extent: min(5, |prev|, |cur|) */
            nov_all = extra;
            overlap = xmalloc((size_t)extra * 3 * fpix * 4);
            for (unsigned f = 0; f < extra; f++)
                for (unsigned c = 0; c < 3; c++)
                    memcpy(overlap + ((size_t)f * 3 + c) * fpix,
                           pix + ((size_t)c * df + 23 + f) * fpix, (size_t)fpix * 4);
            if (last) {
                for (unsigned f = 0; f < extra; f++)
                    memcpy(video + (size_t)(nv + f) * 3 * fpix, overlap + (size_t)f * 3 * fpix,
                           (size_t)3 * fpix * 4);
                nv += extra;
                free(overlap);
                overlap = NULL;
            }
        }
    }
    free(overlap), free(chunk), free(pix), free(z);
    unsigned want = video_frames(lt);
    if (nv > want)
        nv = want;
    *frames_out = nv;
    /* [frame][c] -> [c][frame] */
    float *out = xmalloc((size_t)3 * nv * fpix * 4);
    for (unsigned f = 0; f < nv; f++)
        for (unsigned c = 0; c < 3; c++)
            memcpy(out + ((size_t)c * nv + f) * fpix, video + ((size_t)f * 3 + c) * fpix,
                   (size_t)fpix * 4);
    free(video);
    return out;
}

/* ------------------------------------------------------------------------ */
/* Audio VAE decoder: BigVGAN, once per stereo channel                       */

#define AUDIO_UPS 7
#define AUDIO_RATE 32000
static const int ups_rate[AUDIO_UPS] = {5, 5, 2, 2, 2, 2, 2};
static const int ups_kernel[AUDIO_UPS] = {9, 9, 4, 4, 4, 4, 4};
static const int res_kernel[3] = {3, 7, 11};
static const int res_dil[3] = {1, 3, 5};

/* A 1-D convolution as a GEMM: weights f16 [npad][kpad] (rows >= the real
 * output count and columns >= cin * K are zero), bias f32 [npad]. */
typedef struct {
    Mat w;
    CUdeviceptr b;
    unsigned cin, cout, k;
} Conv;

/* BigVGAN's anti-aliased SnakeBeta: log-scale alpha / beta per channel and
 * the two 12-tap filters (the upsampling one times 2, its gain) */
typedef struct {
    CUdeviceptr alpha, beta, up, dn;
} Snake;

typedef struct {
    SafeTensors st;
    float mean[AUDIO_C], std[AUDIO_C];
    CUdeviceptr in_w, in_b; /* dec_in_proj, f32 [2048][32] */
    Conv pre, ups[AUDIO_UPS], c1[AUDIO_UPS * 3][3], c2[AUDIO_UPS * 3][3], post;
    Snake a1[AUDIO_UPS * 3][3], a2[AUDIO_UPS * 3][3], apost;
    unsigned cap; /* samples */
    CUdeviceptr x, xb, sum, a, h, col, lat;
} Avae;

static unsigned round_up(unsigned n, unsigned m)
{
    return (n + m - 1) / m * m;
}

static const float *st_f32_raw(const SafeTensors *s, const char *name, size_t n)
{
    return (const float *)st_get(s, name, ST_F32, n)->data;
}

/* weights [cout][cin][k] (transposed = 0) or [cin][cout][k] (a transposed
 * conv, whose GEMM rows are (co, k) pairs over cin columns) */
static Conv conv_load(const SafeTensors *s, const char *wname, const char *bname, unsigned cin,
                      unsigned cout, unsigned k, int transposed)
{
    Conv c = {{GGML_F16, 0, 0, 0}, 0, cin, cout, k};
    const float *w = st_f32_raw(s, wname, (size_t)cin * cout * k);
    unsigned n = transposed ? cout * k : cout, kk = transposed ? cin : cin * k;
    unsigned npad = round_up(n, 128), kpad = round_up(kk, 32);
    float *m = calloc((size_t)npad * kpad, 4);
    if (transposed) {
        for (unsigned ci = 0; ci < cin; ci++)
            for (unsigned j = 0; j < cout * k; j++)
                m[(size_t)j * kpad + ci] = w[(size_t)ci * cout * k + j];
    } else {
        for (unsigned co = 0; co < cout; co++)
            memcpy(m + (size_t)co * kpad, w + (size_t)co * cin * k, (size_t)cin * k * 4);
    }
    c.w.w = upload_f16(m, (size_t)npad * kpad);
    c.w.n = npad, c.w.k = kpad;
    free(m);
    float *b = calloc(npad, 4);
    if (bname)
        memcpy(b, st_f32_raw(s, bname, cout), (size_t)cout * 4);
    c.b = upload(b, (size_t)npad * 4);
    free(b);
    return c;
}

static Snake snake_load(const SafeTensors *s, const char *prefix, unsigned c)
{
    char nm[200];
    Snake a;
    snprintf(nm, sizeof(nm), "%s.act.alpha", prefix);
    a.alpha = upload(st_f32_raw(s, nm, c), c * 4);
    snprintf(nm, sizeof(nm), "%s.act.beta", prefix);
    a.beta = upload(st_f32_raw(s, nm, c), c * 4);
    snprintf(nm, sizeof(nm), "%s.upsample.filter", prefix);
    float up[12];
    memcpy(up, st_f32_raw(s, nm, 12), 48);
    for (int i = 0; i < 12; i++)
        up[i] *= 2;
    a.up = upload(up, 48);
    snprintf(nm, sizeof(nm), "%s.downsample.lowpass.filter", prefix);
    a.dn = upload(st_f32_raw(s, nm, 12), 48);
    return a;
}

/* cap: latent frames per channel */
static void avae_load(Avae *v, const char *path, unsigned cap)
{
    double t0 = now_ms();
    st_open(&v->st, path);
    const SafeTensors *s = &v->st;
    char nm[200], nb[200];
    memcpy(v->mean, st_f32_raw(s, "latents_mean", AUDIO_C), sizeof(v->mean));
    memcpy(v->std, st_f32_raw(s, "latents_std", AUDIO_C), sizeof(v->std));
    v->in_w = upload(st_f32_raw(s, "dec_in_proj.weight", 2048 * AUDIO_C), 2048 * AUDIO_C * 4);
    v->in_b = upload(st_f32_raw(s, "dec_in_proj.bias", 2048), 2048 * 4);
    v->pre = conv_load(s, "decoder.conv_pre.weight", "decoder.conv_pre.bias", 2048, 1024, 7, 0);
    unsigned ch = 1024;
    for (int i = 0; i < AUDIO_UPS; i++) {
        unsigned nc = ch / 2;
        snprintf(nm, sizeof(nm), "decoder.ups.%d.0.weight", i);
        snprintf(nb, sizeof(nb), "decoder.ups.%d.0.bias", i);
        v->ups[i] = conv_load(s, nm, nb, ch, nc, ups_kernel[i], 1);
        for (int j = 0; j < 3; j++) {
            int r = i * 3 + j;
            for (int l = 0; l < 3; l++) {
                snprintf(nm, sizeof(nm), "decoder.resblocks.%d.convs1.%d.weight", r, l);
                snprintf(nb, sizeof(nb), "decoder.resblocks.%d.convs1.%d.bias", r, l);
                v->c1[r][l] = conv_load(s, nm, nb, nc, nc, res_kernel[j], 0);
                snprintf(nm, sizeof(nm), "decoder.resblocks.%d.convs2.%d.weight", r, l);
                snprintf(nb, sizeof(nb), "decoder.resblocks.%d.convs2.%d.bias", r, l);
                v->c2[r][l] = conv_load(s, nm, nb, nc, nc, res_kernel[j], 0);
                snprintf(nm, sizeof(nm), "decoder.resblocks.%d.activations.%d", r, 2 * l);
                v->a1[r][l] = snake_load(s, nm, nc);
                snprintf(nm, sizeof(nm), "decoder.resblocks.%d.activations.%d", r, 2 * l + 1);
                v->a2[r][l] = snake_load(s, nm, nc);
            }
        }
        ch = nc;
    }
    v->apost = snake_load(s, "decoder.activation_post", ch);
    v->post = conv_load(s, "decoder.conv_post.weight", NULL, ch, 1, 7, 0);

    v->cap = cap * 800;
    size_t big = 0; /* the largest activation, [samples][max(channels, 128)] f32 */
    for (unsigned l = cap, c = 1024, i = 0; i <= AUDIO_UPS; i++) {
        size_t sz = (size_t)l * (c > 128 ? c : 128);
        if (sz > big)
            big = sz;
        if (i < AUDIO_UPS)
            l *= ups_rate[i], c /= 2;
    }
    size_t col = 0; /* the largest im2col / transposed-conv matrix */
    for (unsigned l = cap, c = 1024, i = 0; i <= AUDIO_UPS; i++) {
        size_t a = (size_t)round_up(l, 128) * round_up(c * 11, 32);
        if (i == 0)
            a = (size_t)round_up(l, 128) * 2048 * 7;
        if (a > col)
            col = a;
        if (i < AUDIO_UPS) {
            size_t z = (size_t)l * round_up(c / 2 * ups_kernel[i], 128);
            if (z > big)
                big = z;
            l *= ups_rate[i], c /= 2;
        }
    }
    v->x = dalloc(big * 4);
    v->xb = dalloc(big * 4);
    v->sum = dalloc(big * 4);
    v->a = dalloc(big * 4);
    v->h = dalloc(big * 4);
    v->col = dzeros(col * 2);
    v->lat = dalloc((size_t)cap * 2048 * 4);
    fprintf(stderr, "audio VAE decoder loaded (%.1f s)\n", (now_ms() - t0) / 1e3);
}

static void avae_free(Avae *v)
{
    st_close(&v->st);
    arena_used = 0;
}

/* y [l][ld] (mode G_F32 / G_ACC) = conv(x [l][ldx], dilation dil, "same"
 * padding pad) */
static void conv1d(const Conv *c, CUdeviceptr x, unsigned ldx, unsigned l, unsigned dil,
                   unsigned pad, CUdeviceptr y, unsigned ld, unsigned mode, CUdeviceptr col)
{
    im2col1d(x, ldx, col, l, c->cin, c->k, dil, pad, c->w.k);
    gemm_f16(col, c->w.w, c->b, y, l, c->w.n, c->w.k, ld, mode, 0, 0, 0, 1);
}

/* One channel: latents [t][32] (normalized) -> samples [t * 800] */
static void avae_channel(Avae *v, const float *lat, unsigned t, float *wave)
{
    float *z = xmalloc((size_t)t * AUDIO_C * 4);
    for (unsigned i = 0; i < t; i++)
        for (unsigned c = 0; c < AUDIO_C; c++)
            z[i * AUDIO_C + c] = lat[i * AUDIO_C + c] * v->std[c] + v->mean[c];
    CHECK(cuMemcpyHtoD(v->a, z, (size_t)t * AUDIO_C * 4));
    free(z);
    linear_f32(v->a, AUDIO_C, v->in_w, v->in_b, v->lat, 2048, t, 2048, AUDIO_C, 0);
    conv1d(&v->pre, v->lat, 2048, t, 1, 3, v->x, 1024, G_F32, v->col);
    unsigned l = t, ch = 1024, ld = 1024;
    for (int i = 0; i < AUDIO_UPS; i++) {
        const Conv *u = &v->ups[i];
        unsigned nc = ch / 2, nl = l * ups_rate[i], nld = nc > 128 ? nc : 128;
        /* transposed conv: z [l][(co, k)] = x . W, then the overlap-add */
        if (ch < u->w.k) /* the padding columns must be zero */
            CHECK(cuMemsetD8(v->col, 0, (size_t)round_up(l, 128) * u->w.k * 2));
        cast_f16(v->x, ld, v->col, u->w.k, l, ch, 1);
        gemm_f16(v->col, u->w.w, 0, v->a, l, u->w.n, u->w.k, u->w.n, G_F32, 0, 0, 0, 1);
        col2im_t(v->a, u->w.n, u->b, v->x, nld, l, nl, nc, u->k, ups_rate[i],
                 (u->k - ups_rate[i]) / 2);
        l = nl, ch = nc, ld = nld;
        /* 3 residual stacks (kernels 3, 7, 11) on the same input, averaged */
        for (int j = 0; j < 3; j++) {
            int r = i * 3 + j;
            CHECK(cuMemcpyDtoD(v->xb, v->x, (size_t)l * ld * 4));
            for (int s = 0; s < 3; s++) {
                const Conv *c1 = &v->c1[r][s], *c2 = &v->c2[r][s];
                unsigned k = res_kernel[j], d = res_dil[s];
                aa_snake(v->xb, v->a, ld, l, ch, v->a1[r][s].alpha, v->a1[r][s].beta,
                         v->a1[r][s].up, v->a1[r][s].dn);
                conv1d(c1, v->a, ld, l, d, (k * d - d) / 2, v->h, ld, G_F32, v->col);
                aa_snake(v->h, v->a, ld, l, ch, v->a2[r][s].alpha, v->a2[r][s].beta,
                         v->a2[r][s].up, v->a2[r][s].dn);
                conv1d(c2, v->a, ld, l, 1, k / 2, v->xb, ld, G_ACC, v->col);
            }
            if (j == 0)
                CHECK(cuMemcpyDtoD(v->sum, v->xb, (size_t)l * ld * 4));
            else
                axpby(v->sum, v->xb, 1, 1, l * ld);
        }
        axpby(v->x, v->sum, 0, 1.f / 3, l * ld);
    }
    aa_snake(v->x, v->a, ld, l, ch, v->apost.alpha, v->apost.beta, v->apost.up, v->apost.dn);
    conv1d(&v->post, v->a, ld, l, 1, 3, v->h, 128, G_F32, v->col);
    float *o = download(v->h, (size_t)l * 128);
    for (unsigned i = 0; i < l; i++) {
        float s = o[(size_t)i * 128];
        wave[i] = s < -1 ? -1 : s > 1 ? 1 : s;
    }
    free(o);
}

/* audio latents as the sampler leaves them ([32][2][t], t fastest) ->
 * left[t * 800], right[t * 800] */
static void avae_decode(Avae *v, const float *audio, unsigned t, float *left, float *right)
{
    float *lat = xmalloc((size_t)t * AUDIO_C * 4);
    for (int lr = 0; lr < 2; lr++) {
        for (unsigned i = 0; i < t; i++)
            for (unsigned c = 0; c < AUDIO_C; c++)
                lat[i * AUDIO_C + c] = audio[i + (size_t)t * lr + 2 * (size_t)t * c];
        avae_channel(v, lat, t, lr ? right : left);
    }
    free(lat);
}

/* ------------------------------------------------------------------------ */
/* check: every stage against stable-diffusion.cpp's dumps                   */

static int check_dit(const char *dir, const char *path)
{
    int64_t ne[8];
    int nd;
    float *ctx = read_dump(dir, "dit_ctx_0", ne, &nd);
    if (!ctx) {
        fprintf(stderr, "%s: no dit_ctx_0.bin\n", dir);
        return 1;
    }
    unsigned text = (unsigned)ne[1];
    float *x0 = read_dump(dir, "dit_x_0", ne, &nd);
    unsigned lw = (unsigned)ne[0], lh = (unsigned)ne[1], lt = (unsigned)ne[2];
    size_t npk = (size_t)ne[0] * ne[1] * ne[2] * ne[3];
    float *ai = read_dump(dir, "audio_in", ne, &nd);
    unsigned audio_t = (unsigned)ne[0];
    free(ai);
    Layout L;
    make_layout(&L, text, audio_t, lt, lh, lw);
    printf("diffusion transformer: %u tokens (%u text, %u audio, %u video)\n", L.n, text,
           L.n_audio, L.n_video);
    Dit d;
    memset(&d, 0, sizeof(d));
    dit_load(&d, path, L.n);
    dit_refine(&d, ctx, text);
    float *out = xmalloc(npk * 4);
    int bad = 0;
    for (int step = 0;; step++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "dit_x_%d", step);
        float *x = step ? read_dump(dir, nm, NULL, NULL) : x0;
        if (!x)
            break;
        snprintf(nm, sizeof(nm), "dit_t_%d", step);
        float *tv = read_dump(dir, nm, NULL, NULL);
        snprintf(nm, sizeof(nm), "dit_out_%d", step);
        float *ref = read_dump(dir, nm, NULL, NULL);
        memset(out, 0, npk * 4);
        double t0 = now_ms();
        dit_forward(&d, &L, tv[0] / 1000.f, x, out);
        double ms = now_ms() - t0;
        double me, re;
        snprintf(nm, sizeof(nm), "step %d (t=%.1f, %.0f ms)", step, tv[0], ms);
        compare(nm, out, ref, npk, &me, &re);
        if (getenv("MINIMAXH3_SAVE")) { /* debugging: our output next to the reference */
            char fn[1100];
            snprintf(fn, sizeof(fn), "%s/ours_dit_out_%d.f32", getenv("MINIMAXH3_SAVE"), step);
            FILE *f = fopen(fn, "wb");
            if (f)
                fwrite(out, 4, npk, f), fclose(f);
        }
        bad |= re > 0.08; /* sd.cpp CUDA vs CPU differ by ~4%: README */
        free(x), free(tv), free(ref);
    }
    free(out), free(ctx), free(L.pos);
    dit_free(&d);
    return bad;
}

static int check_vae(const char *dir, const char *path)
{
    int64_t ne[8];
    float *x = read_dump(dir, "vae_in", ne, NULL);
    if (!x) {
        fprintf(stderr, "%s: no vae_in.bin\n", dir);
        return 1;
    }
    unsigned lw = (unsigned)ne[0], lh = (unsigned)ne[1], lt = (unsigned)ne[2];
    float *ref = read_dump(dir, "vae_out", ne, NULL);
    printf("video VAE decoder: %u latent frames of %ux%u\n", lt, lh, lw);
    Vae v;
    memset(&v, 0, sizeof(v));
    vae_load(&v, path, 7 * lh * lw + 5);
    double t0 = now_ms();
    unsigned nf;
    float *pix = vae_decode(&v, x, lt, lh, lw, &nf);
    double ms = now_ms() - t0;
    int bad = 0;
    if (nf != (unsigned)ne[2]) {
        printf("  %u frames, the reference has %u\n", nf, (unsigned)ne[2]);
        bad = 1;
    } else {
        /* reference layout: [3][frames][h][w], as ours */
        double me, re;
        char nm[64];
        snprintf(nm, sizeof(nm), "%u frames (%.0f ms)", nf, ms);
        compare(nm, pix, ref, (size_t)3 * nf * 256 * lh * lw, &me, &re);
        size_t n8 = 0, n1 = 0, tot = (size_t)3 * nf * 256 * lh * lw;
        for (size_t i = 0; i < tot; i++) { /* as 8-bit pixels */
            int a = (int)lrintf(pix[i] * 255), b = (int)lrintf(ref[i] * 255);
            n8 += a != b, n1 += abs(a - b) > 1;
        }
        printf("  8-bit pixels: %.2f%% differ, %.3f%% by more than 1\n", 100.0 * n8 / tot,
               100.0 * n1 / tot);
        bad = re > 0.02;
    }
    free(x), free(ref), free(pix);
    vae_free(&v);
    return bad;
}

static int check_audio(const char *dir, const char *path)
{
    int64_t ne[8];
    float *a = read_dump(dir, "audio_in", ne, NULL);
    if (!a) {
        fprintf(stderr, "%s: no audio_in.bin\n", dir);
        return 1;
    }
    unsigned t = (unsigned)ne[0];
    float *ref = read_dump(dir, "audio_out", ne, NULL);
    printf("audio VAE decoder: %u latents per channel\n", t);
    Avae v;
    memset(&v, 0, sizeof(v));
    avae_load(&v, path, t);
    float *out = xmalloc((size_t)t * 800 * 2 * 4);
    double t0 = now_ms();
    avae_decode(&v, a, t, out, out + (size_t)t * 800);
    double ms = now_ms() - t0, me, re;
    char nm[64];
    snprintf(nm, sizeof(nm), "%u samples x 2 (%.0f ms)", t * 800, ms);
    compare(nm, out, ref, (size_t)t * 800 * 2, &me, &re);
    if (getenv("MINIMAXH3_SAVE")) { /* debugging: our waveform */
        char fn[1100];
        snprintf(fn, sizeof(fn), "%s/ours_audio_out.f32", getenv("MINIMAXH3_SAVE"));
        FILE *f = fopen(fn, "wb");
        if (f)
            fwrite(out, 4, (size_t)t * 800 * 2, f), fclose(f);
    }
    free(a), free(ref), free(out);
    avae_free(&v);
    return re > 0.03; /* the reference itself is ~2% off (f16 im2col): README */
}

static int check_te(const char *dir, const char *path)
{
    int64_t ne[8];
    float *tf = read_dump(dir, "llm_tokens", ne, NULL);
    if (!tf) {
        fprintf(stderr, "%s: no llm_tokens.bin\n", dir);
        return 1;
    }
    unsigned T = (unsigned)ne[0];
    int32_t *tok = xmalloc(T * 4);
    for (unsigned i = 0; i < T; i++)
        tok[i] = (int32_t)tf[i];
    float *ref = read_dump(dir, "dit_ctx_0", ne, NULL);
    printf("text encoder: %u tokens\n", T);
    float *ctx = xmalloc((size_t)T * TE_DIM * 4);
    double t0 = now_ms();
    te_encode(path, tok, T, ctx);
    double me, re;
    char nm[64];
    snprintf(nm, sizeof(nm), "hidden states (%.1f s)", (now_ms() - t0) / 1e3);
    compare(nm, ctx, ref, (size_t)T * TE_DIM, &me, &re);
    free(tf), free(tok), free(ref), free(ctx);
    return re > 0.03;
}

/* the whole sampler from the dumped noise and condition -> x_final */
static int check_sampler(const char *dir, const char *path, unsigned steps)
{
    int64_t ne[8];
    float *ctx = read_dump(dir, "dit_ctx_0", ne, NULL);
    unsigned text = (unsigned)ne[1];
    float *x = read_dump(dir, "x_init", ne, NULL);
    unsigned lw = (unsigned)ne[0], lh = (unsigned)ne[1], lt = (unsigned)ne[2];
    size_t npk = (size_t)ne[0] * ne[1] * ne[2] * ne[3];
    float *ref = read_dump(dir, "x_final", NULL, NULL);
    float *ai = read_dump(dir, "audio_in", ne, NULL);
    unsigned audio_t = (unsigned)ne[0];
    Layout L;
    make_layout(&L, text, audio_t, lt, lh, lw);
    Dit d;
    memset(&d, 0, sizeof(d));
    dit_load(&d, path, L.n);
    dit_refine(&d, ctx, text);
    double t0 = now_ms(), me, re;
    sample(&d, &L, steps, x, npk, 0);
    char nm[64];
    snprintf(nm, sizeof(nm), "%u-step sampler (%.1f s)", steps, (now_ms() - t0) / 1e3);
    printf("sampler: x_init -> x_final\n");
    size_t sp = (size_t)lt * lh * lw;
    compare(nm, x, ref, (size_t)VIDEO_C * sp, &me, &re);
    double me2, re2;
    compare("  audio latents", x + (size_t)VIDEO_C * sp, ref + (size_t)VIDEO_C * sp,
            (size_t)audio_t * 2 * AUDIO_C, &me2, &re2);
    if (getenv("MINIMAXH3_SAVE")) { /* debugging: our final latent */
        char fn[1100];
        snprintf(fn, sizeof(fn), "%s/ours_x_final.f32", getenv("MINIMAXH3_SAVE"));
        FILE *f = fopen(fn, "wb");
        if (f)
            fwrite(x, 4, npk, f), fclose(f);
    }
    free(ctx), free(x), free(ref), free(ai), free(L.pos);
    dit_free(&d);
    /* free-running: sd.cpp's 8-bit activations alone move this ~13% (README) */
    return re > 0.2 || re2 > 0.2;
}

/* the prompt's tokens and the starting noise */
static int check_inputs(const char *dir, const char *prompt)
{
    int64_t ne[8];
    int bad = 0;
    float *tf = read_dump(dir, "llm_tokens", ne, NULL);
    if (tf && prompt) {
        Tokenizer *tk = tok_load(DEFAULT_MERGES);
        int32_t *ids;
        int n = tk ? tok_encode(tk, prompt, &ids) : -1;
        int same = n == (int)ne[0];
        for (int i = 0; same && i < n; i++)
            same = ids[i] == (int32_t)tf[i];
        printf("tokenizer: %d tokens, %s the reference's %d\n", n, same ? "same as" : "DIFFERENT from",
               (int)ne[0]);
        bad |= !same;
        if (tk)
            tok_free(tk), free(ids);
    }
    free(tf);
    float *x = read_dump(dir, "x_init", ne, NULL);
    if (x) {
        size_t n = (size_t)ne[0] * ne[1] * ne[2] * ne[3];
        float *z = xmalloc(n * 4);
        philox_randn(42, 0, z, n);
        size_t diff = 0;
        for (size_t i = 0; i < n; i++)
            diff += z[i] != x[i];
        printf("noise (seed 42): %zu of %zu values differ from x_init\n", diff, n);
        bad |= diff != 0;
        free(z), free(x);
    }
    return bad;
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *t = xmalloc(n + 1);
    if (fread(t, 1, n, f) != (size_t)n)
        n = 0;
    t[n] = 0;
    fclose(f);
    while (n > 0 && (t[n - 1] == '\n' || t[n - 1] == '\r'))
        t[--n] = 0;
    return t;
}

/* Every stage, each from the reference's own inputs (so errors don't
 * compound): tokens and noise, text encoder, each DiT call, the whole
 * sampler, both decoders. DIR/prompt.txt holds the prompt. */
static int cmd_check(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/prompt.txt", dir);
    char *prompt = read_text(path);
    int64_t ne[8];
    unsigned steps = 0;
    for (;; steps++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "dit_t_%u", steps);
        float *t = read_dump(dir, nm, ne, NULL);
        if (!t)
            break;
        free(t);
    }
    const char *only = getenv("MINIMAXH3_ONLY"); /* inputs, te, dit, sampler, vae, audio */
#define RUN(name) (!only || !strcmp(only, name))
    int bad = 0;
    if (RUN("inputs"))
        bad |= check_inputs(dir, prompt);
    if (RUN("te"))
        bad |= check_te(dir, DEFAULT_TE);
    if (RUN("dit"))
        bad |= check_dit(dir, DEFAULT_DIT);
    if (RUN("sampler") && steps)
        bad |= check_sampler(dir, DEFAULT_DIT, steps);
    if (RUN("vae"))
        bad |= check_vae(dir, DEFAULT_VAE);
    if (RUN("audio"))
        bad |= check_audio(dir, DEFAULT_AVAE);
#undef RUN
    free(prompt);
    printf(bad ? "SOME STAGES DIFFER\n" : "all stages match\n");
    return bad;
}

/* ------------------------------------------------------------------------ */

static void init_cuda(const char *cubin)
{
    CUdevice dev;
    CUcontext ctx;
    char name[256];
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
    /* the kernels use no local memory or device printf/malloc: shrink the
     * per-context reservations to leave room for the weights */
    CHECK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 0));
    CHECK(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 0));
    fprintf(stderr, "device: %s\n", name);
    load_kernels(cubin);
    arena_init();
}

/* ------------------------------------------------------------------------ */
/* generation                                                                */

typedef struct {
    const char *prompt, *out, *png_dir, *wav, *dit, *te, *vae, *avae, *merges;
    unsigned w, h, frames, fps, steps;
    uint64_t seed;
} GenOptions;

static int cmd_generate(const GenOptions *o)
{
    double t_all = now_ms();
    if (o->w % 32 || o->h % 32 || !o->w || !o->h) {
        fprintf(stderr, "width and height must be multiples of 32\n");
        return 1;
    }
    if (o->w > 256 || o->h > 256) {
        fprintf(stderr, "this engine decodes video up to 256 x 256 (the VAE's tile size)\n");
        return 1;
    }
    /* frames: 5 + 17 k (the VAE's temporal chunks) */
    unsigned frames = o->frames < 5 ? 5 : o->frames;
    while (frames % 17 != 5)
        frames++;
    const unsigned lt = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    const unsigned lh = o->h / 16, lw = o->w / 16;
    const unsigned audio_t = (unsigned)lround((double)frames * 40.0 / o->fps);
    if (audio_t < 1) {
        fprintf(stderr, "no audio latents for %u frames at %u fps\n", frames, o->fps);
        return 1;
    }

    Tokenizer *tk = tok_load(o->merges);
    if (!tk)
        return 1;
    int32_t *ids;
    int text = tok_encode(tk, o->prompt, &ids);
    tok_free(tk);
    if (text < 1) {
        fprintf(stderr, "empty prompt\n");
        return 1;
    }
    fprintf(stderr, "prompt: %d tokens; video %ux%u, %u frames at %u fps; audio %u latents\n", text,
            o->w, o->h, frames, o->fps, audio_t);

    /* 1. text encoder */
    float *ctx = xmalloc((size_t)text * TE_DIM * 4);
    te_encode(o->te, ids, (unsigned)text, ctx);
    free(ids);

    /* 2. diffusion */
    Layout L;
    make_layout(&L, (unsigned)text, audio_t, lt, lh, lw);
    size_t npk = packed_size(lt, lh, lw, audio_t);
    float *x = xmalloc(npk * 4);
    philox_randn(o->seed, 0, x, npk);
    Dit d;
    memset(&d, 0, sizeof(d));
    dit_load(&d, o->dit, L.n);
    double t0 = now_ms();
    dit_refine(&d, ctx, (unsigned)text);
    sample(&d, &L, o->steps, x, npk, 1);
    fprintf(stderr, "sampling: %u steps over %u tokens (%.1f s)\n", o->steps, L.n,
            (now_ms() - t0) / 1e3);
    dit_free(&d);
    free(ctx);

    /* 3. video */
    Vae v;
    memset(&v, 0, sizeof(v));
    vae_load(&v, o->vae, 7 * lh * lw + 5);
    t0 = now_ms();
    unsigned nf;
    float *pix = vae_decode(&v, x, lt, lh, lw, &nf);
    vae_free(&v);
    fprintf(stderr, "video decoded: %u frames (%.1f s)\n", nf, (now_ms() - t0) / 1e3);

    /* 4. audio */
    Avae a;
    memset(&a, 0, sizeof(a));
    avae_load(&a, o->avae, audio_t);
    const unsigned ns = audio_t * 800;
    float *left = xmalloc((size_t)ns * 4), *right = xmalloc((size_t)ns * 4);
    t0 = now_ms();
    avae_decode(&a, x + (size_t)VIDEO_C * lt * lh * lw, audio_t, left, right);
    avae_free(&a);
    fprintf(stderr, "audio decoded: %u samples at %d Hz (%.1f s)\n", ns, AUDIO_RATE,
            (now_ms() - t0) / 1e3);

    /* 5. files: 8-bit RGB as the reference rounds it */
    const unsigned W = o->w, H = o->h;
    uint8_t *rgb = xmalloc((size_t)nf * H * W * 3);
    for (unsigned f = 0; f < nf; f++)
        for (unsigned c = 0; c < 3; c++)
            for (size_t i = 0; i < (size_t)H * W; i++) {
                float val = pix[((size_t)c * nf + f) * H * W + i];
                rgb[((size_t)f * H * W + i) * 3 + c] =
                    val <= 0 ? 0 : val >= 1 ? 255 : (uint8_t)(val * 255.0f + 0.5f);
            }
    int ok = avi_write(o->out, rgb, W, H, nf, o->fps, left, right, ns, AUDIO_RATE);
    if (ok)
        fprintf(stderr, "wrote %s (%.1f s of video and audio)\n", o->out, (double)nf / o->fps);
    else
        fprintf(stderr, "cannot write %s\n", o->out);
    if (o->wav && !wav_write(o->wav, left, right, ns, AUDIO_RATE))
        fprintf(stderr, "cannot write %s\n", o->wav), ok = 0;
    if (o->png_dir)
        mkdir(o->png_dir, 0777); /* may exist already */
    if (o->png_dir)
        for (unsigned f = 0; f < nf; f++) {
            char fn[1024];
            snprintf(fn, sizeof(fn), "%s/frame_%03u.png", o->png_dir, f);
            if (!png_write(fn, rgb + (size_t)f * H * W * 3, W, H)) {
                fprintf(stderr, "cannot write %s\n", fn), ok = 0;
                break;
            }
        }
    fprintf(stderr, "total %.1f s\n", (now_ms() - t_all) / 1e3);
    free(rgb), free(pix), free(left), free(right), free(x), free(L.pos);
    return ok ? 0 : 1;
}

static int cmd_tokenize(const char *merges, const char *text)
{
    Tokenizer *tk = tok_load(merges);
    if (!tk)
        return 1;
    int32_t *ids;
    int n = tok_encode(tk, text, &ids);
    for (int i = 0; i < n; i++) {
        size_t len;
        const char *p = tok_piece(tk, ids[i], &len);
        printf("%6d  '%.*s'\n", ids[i], (int)len, p);
    }
    free(ids);
    tok_free(tk);
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [options] PROMPT        generate a video with sound\n"
            "       %s test                    kernel self-tests\n"
            "       %s check DIR               every stage against stable-diffusion.cpp's\n"
            "                                  dumps in DIR (README.md)\n"
            "       %s tokenize TEXT           show the tokens of TEXT\n"
            "options:\n"
            "  -o FILE         output AVI (default out.avi)\n"
            "  -W N, -H N      size in pixels, multiples of 32, at most 256 (default 256)\n"
            "  -f N            frames (rounded up to 5 + 17 k; default 22)\n"
            "  --fps N         frame rate (default 24)\n"
            "  --steps N       sampling steps (default 20)\n"
            "  -s N            seed (default 42)\n"
            "  --png DIR       also write every frame as DIR/frame_NNN.png\n"
            "  --wav FILE      also write the soundtrack as a WAV file\n"
            "  --dit FILE      diffusion model (default " DEFAULT_DIT ")\n"
            "  --te FILE       text encoder (default " DEFAULT_TE ")\n"
            "  --vae FILE      video VAE (default " DEFAULT_VAE ")\n"
            "  --audio-vae F   audio VAE (default " DEFAULT_AVAE ")\n"
            "  --merges FILE   tokenizer merges (default " DEFAULT_MERGES ")\n"
            "  --cubin FILE    kernels (default " DEFAULT_CUBIN ")\n",
            prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
    const char *cubin = DEFAULT_CUBIN;
    GenOptions o = {NULL, "out.avi", NULL, NULL, DEFAULT_DIT, DEFAULT_TE, DEFAULT_VAE,
                    DEFAULT_AVAE, DEFAULT_MERGES, 256, 256, 22, 24, 20, 42};
    emulate_q8 = getenv("MINIMAXH3_EMU_Q8") != NULL;
    if (argc >= 2 && !strcmp(argv[1], "test")) {
        init_cuda(cubin);
        return self_test();
    }
    if (argc >= 3 && !strcmp(argv[1], "check")) {
        init_cuda(cubin);
        return cmd_check(argv[2]);
    }
    if (argc >= 3 && !strcmp(argv[1], "tokenize"))
        return cmd_tokenize(DEFAULT_MERGES, argv[2]);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
#define OPT(name) (!strcmp(a, name) && more)
        if (OPT("-o"))
            o.out = argv[++i];
        else if (OPT("-W"))
            o.w = (unsigned)atoi(argv[++i]);
        else if (OPT("-H"))
            o.h = (unsigned)atoi(argv[++i]);
        else if (OPT("-f"))
            o.frames = (unsigned)atoi(argv[++i]);
        else if (OPT("--fps"))
            o.fps = (unsigned)atoi(argv[++i]);
        else if (OPT("--steps"))
            o.steps = (unsigned)atoi(argv[++i]);
        else if (OPT("-s"))
            o.seed = strtoull(argv[++i], NULL, 10);
        else if (OPT("--png"))
            o.png_dir = argv[++i];
        else if (OPT("--wav"))
            o.wav = argv[++i];
        else if (OPT("--dit"))
            o.dit = argv[++i];
        else if (OPT("--te"))
            o.te = argv[++i];
        else if (OPT("--vae"))
            o.vae = argv[++i];
        else if (OPT("--audio-vae"))
            o.avae = argv[++i];
        else if (OPT("--merges"))
            o.merges = argv[++i];
        else if (OPT("--cubin"))
            cubin = argv[++i];
        else if (a[0] == '-' || o.prompt) {
            usage(argv[0]);
            return 1;
        } else
            o.prompt = a;
#undef OPT
    }
    if (!o.prompt || !o.steps || !o.fps) {
        usage(argv[0]);
        return 1;
    }
    init_cuda(cubin);
    return cmd_generate(&o);
}
