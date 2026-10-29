/*
 * sam3.c - SAM 3 (Segment Anything 3, image model) inference on the GPU with
 * hand-written PTX kernels (sam3.ptx) driven through the CUDA Driver API. No
 * other libraries. Given an image and a noun phrase ("dog", "red car") it
 * finds every instance of the phrase and returns a box, a score and a mask
 * for each (Meta's "promptable concept segmentation", text prompts only).
 *
 * The weights come from a GGUF file converted from the MLX port
 * (sam3-image-f32.gguf: 1368 f32 tensors). Its tensor names are
 * "t_" + the first 12 hex digits of SHA-1(PyTorch parameter name), with
 * nn.MultiheadAttention's in_proj split into query_proj / key_proj /
 * value_proj, and convolutions stored [out][kh][kw][in] (MLX); the engine
 * hashes the names it needs. The text tokenizer's merges are not in the
 * GGUF: they are read from clip_merges.txt (CLIP's list).
 *
 * Pipeline (sam3.ptx's header has the kernel for each step):
 *   image -> 1008 x 1008 -> ViT-L (32 blocks, windowed + 4 global, 2-D RoPE)
 *   -> 72 x 72 x 1024 -> FPN neck: 288^2, 144^2, 72^2 x 256
 *   prompt -> CLIP tokens (32) -> text transformer (24 blocks) -> 32 x 256
 *   geometry encoder (no boxes given: one CLS token attending to the image)
 *   prompt = text tokens + CLS (33 x 256)
 *   fusion encoder (6 layers): image tokens attend to themselves and the prompt
 *   decoder (6 layers): 200 object queries + a presence token, boxes refined
 *   per layer; score = sigmoid(query . prompt) * sigmoid(presence)
 *   masks: pixel decoder over the FPN levels, mask = query embedding . pixels
 *   (288 x 288), kept queries (score > threshold) resized to the image.
 * Every matrix product runs in f16 on the tensor cores with f32 accumulation.
 * The image features are computed once per image; each prompt then costs
 * the text encoder and everything after it.
 *
 * Usage (see usage() for every option):
 *   ./sam3_demo -i IMAGE -p PROMPT [-p PROMPT2 ...] [-o out.png] [-t 0.5]
 *   ./sam3_demo test                  kernel self-tests against the CPU
 *   ./sam3_demo tokenize TEXT         CLIP token ids of TEXT
 *   ./sam3_demo tokenize-ids          one line of ids per stdin line
 * Environment (debugging): SAM3_DUMP=DIR writes intermediate tensors as raw
 * f32 files (DIR/<name>.f32, row-major [rows][channels]) for
 * analysis_tools/verification/compare_ref.py; SAM3_INPUT=FILE replaces the
 * preprocessed image by FILE (f32 [1008][1008][3], as ref_sam3.py writes).
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "clip_tokenizer.h"
#include "gguf.h"
#include "image.h"

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

#define DEFAULT_MODEL "sam3-image-f32.gguf"
#define DEFAULT_CUBIN "sam3.cubin"
#define DEFAULT_MERGES "clip_merges.txt"

/* model shape (the GGUF has a single configuration) */
#define IMG 1008                /* input side */
#define PATCH 14
#define GRID 72                 /* IMG / PATCH */
#define NTOK (GRID * GRID)      /* 5184 image tokens */
#define PATCH_K 588             /* 14 * 14 * 3 */
#define PATCH_KPAD 608          /* padded to a multiple of 32 */
#define VIT_D 1024
#define VIT_H 16
#define VIT_FF 4736
#define VIT_LAYERS 32
#define WIN 24                  /* window side; also the pretraining grid */
#define TXT_LEN 32
#define TXT_D 1024
#define TXT_H 16
#define TXT_FF 4096
#define TXT_LAYERS 24
#define D 256                   /* width of everything after the backbones */
#define NH 8                    /* heads of 32 */
#define FF 2048
#define ENC_LAYERS 6
#define DEC_LAYERS 6
#define GEO_LAYERS 3
#define NQ 200                  /* object queries */
#define NPROMPT (TXT_LEN + 1)   /* text tokens + the geometry CLS token */
#define S0 288                  /* FPN level 0 side (mask resolution) */
#define S1 144
#define LN_EPS 1e-5f
#define ROWS(n) (((n) + 127) / 128 * 128) /* gemm_f16 reads A in 128-row tiles */
#define PAD64(n) (((n) + 63) / 64 * 64)

/* Host wall-clock time in milliseconds. */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
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
/* Kernel handles and launch helpers                                         */

static CUfunction k_layernorm, k_add_pos, k_im2col_patch, k_gemm, k_rope_vit, k_split, k_flash,
    k_add_cast, k_d2s, k_im2col3, k_upadd, k_gn_stats, k_gn_apply, k_linear, k_box_sine, k_rpb,
    k_box_update, k_masked_mean, k_mask_resize, k_f2h;
static unsigned max_blocks; /* cap for grid-stride kernels: 8 blocks per SM */

static void load_kernels(const char *cubin)
{
    CUmodule mod;
    if (cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) {
        fprintf(stderr, "cannot load %s (run `make` in the sam3 directory)\n", cubin);
        exit(1);
    }
    struct { CUfunction *fn; const char *name; } k[] = {
        {&k_layernorm, "layernorm"}, {&k_add_pos, "add_pos"}, {&k_im2col_patch, "im2col_patch"},
        {&k_gemm, "gemm_f16"}, {&k_rope_vit, "rope_vit"}, {&k_split, "split_heads"},
        {&k_flash, "flash_attn"}, {&k_add_cast, "add_cast"}, {&k_d2s, "depth2space"},
        {&k_im2col3, "im2col3"}, {&k_upadd, "upsample_add"}, {&k_gn_stats, "gn_stats"},
        {&k_gn_apply, "gn_apply"}, {&k_linear, "linear_small"}, {&k_box_sine, "box_sine"},
        {&k_rpb, "rpb_mlp"}, {&k_box_update, "box_update"}, {&k_masked_mean, "masked_mean"},
        {&k_mask_resize, "mask_resize"}, {&k_f2h, "f32_to_f16"},
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
static void launch_1d(CUfunction fn, unsigned total, void **args)
{
    const unsigned threads = 256;
    unsigned blocks = (total + threads - 1) / threads;
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks == 0)
        blocks = 1;
    CHECK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL));
}

static size_t dev_bytes; /* everything dalloc() has allocated */

static CUdeviceptr dalloc(size_t bytes)
{
    CUdeviceptr p;
    CHECK(cuMemAlloc(&p, bytes));
    dev_bytes += bytes;
    CHECK(cuMemsetD8(p, 0, bytes));
    return p;
}

/* A linear layer: f16 weight [n][k] (k contiguous), f32 bias [n] or 0. */
typedef struct {
    CUdeviceptr w, b;
    unsigned n, k;
} Lin;

/* A LayerNorm / GroupNorm affine: f32 weight and bias. */
typedef struct {
    CUdeviceptr w, b;
} Norm;

enum { G_F32 = 0, G_F16 = 1, G_GELU = 2, G_RELU = 3, G_ADD = 4, G_RELU32 = 5 };

/* C[M][ldc] (op) = A[M][K] . B[N][K]^T + bias (see gemm_f16) */
static void gemm_raw(CUdeviceptr a, CUdeviceptr b, CUdeviceptr bias, CUdeviceptr c, unsigned m,
                     unsigned n, unsigned k, unsigned ldc, unsigned mode)
{
    if (n % 128 || k % 32) {
        fprintf(stderr, "gemm_f16: N %% 128 and K %% 32 must be 0 (N %u, K %u)\n", n, k);
        exit(1);
    }
    void *args[] = {&a, &b, &bias, &c, &m, &n, &k, &ldc, &mode};
    CHECK(cuLaunchKernel(k_gemm, n / 128, (m + 127) / 128, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void gemm(CUdeviceptr a, const Lin *w, CUdeviceptr c, unsigned m, unsigned mode)
{
    gemm_raw(a, w->w, w->b, c, m, w->n, w->k, w->n, mode);
}

static void layernorm(CUdeviceptr x, const Norm *nm, CUdeviceptr y32, CUdeviceptr y16,
                      CUdeviceptr pos, CUdeviceptr yp16, unsigned rows, unsigned n)
{
    float eps = LN_EPS;
    void *args[] = {&x, (void *)&nm->w, (void *)&nm->b, &y32, &y16, &pos, &yp16, &rows, &n, &eps};
    CHECK(cuLaunchKernel(k_layernorm, (rows + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void add_cast(CUdeviceptr a, CUdeviceptr b, CUdeviceptr ys, CUdeviceptr ya, unsigned n)
{
    void *args[] = {&a, &b, &ys, &ya, &n};
    launch_1d(k_add_cast, n / 2, args);
}

/* Attention launch parameters beyond Q, K, V, O (see flash_attn). */
typedef struct {
    unsigned lq, lk, lq_pad, lk_pad, bh, h, dout, ostride, flags;
    float scale;
    CUdeviceptr kmask, bx, by;
    unsigned bgw, wsz, nwx, gw;
} Att;

static void flash(CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr o, const Att *a)
{
    void *args[] = {&q, &k, &v, &o, (void *)&a->lq, (void *)&a->lk, (void *)&a->lq_pad,
                    (void *)&a->lk_pad, (void *)&a->h, (void *)&a->dout, (void *)&a->ostride,
                    (void *)&a->scale, (void *)&a->flags, (void *)&a->kmask, (void *)&a->bx,
                    (void *)&a->by, (void *)&a->bgw, (void *)&a->wsz, (void *)&a->nwx,
                    (void *)&a->gw};
    CHECK(cuLaunchKernel(k_flash, a->lq_pad / 64, a->bh, 1, 128, 1, 1, 0, NULL, args, NULL));
}

static void split_heads(CUdeviceptr src, unsigned ld, unsigned col0, CUdeviceptr dst, unsigned l,
                        unsigned lpad, unsigned h, unsigned hd)
{
    void *args[] = {&src, &ld, &col0, &dst, &l, &lpad, &h, &hd};
    launch_1d(k_split, l * h * 32, args);
}

static void linear_small(CUdeviceptr x, unsigned ldx, CUdeviceptr w, CUdeviceptr b,
                         CUdeviceptr y, unsigned ldy, unsigned m, unsigned n, unsigned k,
                         unsigned act)
{
    void *args[] = {&x, &ldx, &w, &b, &y, &ldy, &m, &n, &k, &act};
    CHECK(cuLaunchKernel(k_linear, (m * n + 7) / 8, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static void im2col3(CUdeviceptr src, CUdeviceptr dst, unsigned h, unsigned w, unsigned c)
{
    void *args[] = {&src, &dst, &h, &w, &c};
    launch_1d(k_im2col3, h * w * 9 * c / 8, args);
}

static void depth2space(CUdeviceptr src, CUdeviceptr dst, unsigned h, unsigned w, unsigned o)
{
    void *args[] = {&src, &dst, &h, &w, &o};
    launch_1d(k_d2s, h * w * 4 * o / 2, args);
}

/* ------------------------------------------------------------------------ */
/* SHA-1, for the GGUF's tensor names                                       */

static void sha1(const uint8_t *msg, size_t n, uint8_t out[20])
{
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    size_t total = ((n + 8) / 64 + 1) * 64;
    uint8_t *buf = calloc(total, 1);
    memcpy(buf, msg, n);
    buf[n] = 0x80;
    uint64_t bits = (uint64_t)n * 8;
    for (int i = 0; i < 8; i++)
        buf[total - 1 - i] = (uint8_t)(bits >> (8 * i));
    for (size_t off = 0; off < total; off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)buf[off + 4 * i] << 24 | (uint32_t)buf[off + 4 * i + 1] << 16 |
                   (uint32_t)buf[off + 4 * i + 2] << 8 | buf[off + 4 * i + 3];
        for (int i = 16; i < 80; i++) {
            uint32_t v = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = v << 1 | v >> 31;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)
                f = (b & c) | (~b & d), k = 0x5A827999;
            else if (i < 40)
                f = b ^ c ^ d, k = 0x6ED9EBA1;
            else if (i < 60)
                f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC;
            else
                f = b ^ c ^ d, k = 0xCA62C1D6;
            uint32_t t = (a << 5 | a >> 27) + f + e + k + w[i];
            e = d, d = c, c = b << 30 | b >> 2, b = a, a = t;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e;
    }
    free(buf);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++)
            out[4 * i + j] = (uint8_t)(h[i] >> (24 - 8 * j));
}

/* ------------------------------------------------------------------------ */
/* Weights                                                                  */

static Gguf gg;
static size_t weight_bytes;
static CUdeviceptr stage;      /* f32 staging for the f16 conversion */
static size_t stage_bytes;

/* The tensor of a PyTorch parameter name (f32), checking its element count. */
static const GgufTensor *tensor(const char *name, size_t count)
{
    uint8_t dg[20];
    sha1((const uint8_t *)name, strlen(name), dg);
    char key[16] = "t_";
    for (int i = 0; i < 6; i++)
        sprintf(key + 2 + 2 * i, "%02x", dg[i]);
    const GgufTensor *t = gguf_tensor(&gg, key, GGML_F32, 0, 0, 0);
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++)
        n *= t->ne[i];
    if (count && n != count) {
        fprintf(stderr, "%s (%s): %zu elements, expected %zu\n", name, key, n, count);
        exit(1);
    }
    return t;
}

static const float *host_f32(const char *name, size_t count)
{
    return (const float *)tensor(name, count)->data;
}

static CUdeviceptr up_f32(const char *name, size_t count)
{
    const GgufTensor *t = tensor(name, count);
    CUdeviceptr p = dalloc(t->nbytes);
    CHECK(cuMemcpyHtoD(p, t->data, t->nbytes));
    weight_bytes += t->nbytes;
    return p;
}

/* host f32 -> device f16 */
static void to_f16(CUdeviceptr dst, const float *src, size_t count)
{
    for (size_t off = 0; off < count;) {
        size_t n = count - off;
        if (n > stage_bytes / 4)
            n = stage_bytes / 4;
        CHECK(cuMemcpyHtoD(stage, src + off, n * 4));
        CUdeviceptr d = dst + off * 2;
        unsigned nn = (unsigned)n;
        void *args[] = {&stage, &d, &nn};
        launch_1d(k_f2h, nn / 2, args);
        CHECK(cuCtxSynchronize()); /* before the next copy reuses the staging buffer */
        off += n;
    }
}

/* The linear layer prefix.weight [n][k] (+ prefix.bias when bias). */
static Lin lin(const char *prefix, unsigned n, unsigned k, int bias)
{
    char name[700];
    Lin l = {0, 0, n, k};
    snprintf(name, sizeof(name), "%s.weight", prefix);
    l.w = dalloc((size_t)n * k * 2);
    to_f16(l.w, host_f32(name, (size_t)n * k), (size_t)n * k);
    weight_bytes += (size_t)n * k * 2;
    if (bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        l.b = up_f32(name, n);
    }
    return l;
}

/* Several linear layers stacked by rows into one (e.g. query + key). */
static Lin lin_cat(const char *const *prefixes, int count, unsigned n_each, unsigned k)
{
    Lin l = {dalloc((size_t)count * n_each * k * 2), dalloc((size_t)count * n_each * 4),
             count * n_each, k};
    char name[700];
    for (int i = 0; i < count; i++) {
        snprintf(name, sizeof(name), "%s.weight", prefixes[i]);
        to_f16(l.w + (size_t)i * n_each * k * 2, host_f32(name, (size_t)n_each * k),
               (size_t)n_each * k);
        snprintf(name, sizeof(name), "%s.bias", prefixes[i]);
        CHECK(cuMemcpyHtoD(l.b + (size_t)i * n_each * 4, host_f32(name, n_each), n_each * 4));
    }
    weight_bytes += (size_t)count * n_each * (k * 2 + 4);
    return l;
}

static Norm norm(const char *prefix, unsigned n)
{
    char name[700];
    Norm m;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    m.w = up_f32(name, n);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    m.b = up_f32(name, n);
    return m;
}

/* nn.MultiheadAttention (256 wide): separate and stacked projections */
typedef struct {
    Lin q, k, v, o, qk, kv;
} Mha;

static Mha mha(const char *prefix)
{
    char q[600], k[600], v[600], o[600];
    snprintf(q, sizeof(q), "%s.query_proj", prefix);
    snprintf(k, sizeof(k), "%s.key_proj", prefix);
    snprintf(v, sizeof(v), "%s.value_proj", prefix);
    snprintf(o, sizeof(o), "%s.out_proj", prefix);
    Mha m;
    m.q = lin(q, D, D, 1);
    m.k = lin(k, D, D, 1);
    m.v = lin(v, D, D, 1);
    m.o = lin(o, D, D, 1);
    const char *qk[] = {q, k}, *kv[] = {k, v};
    m.qk = lin_cat(qk, 2, D, D);
    m.kv = lin_cat(kv, 2, D, D);
    return m;
}

typedef struct {
    Norm n1, n2;
    Lin qkv, proj, fc1, fc2;
} VitBlock;

typedef struct {
    Norm ln1, ln2;
    Lin qkv, o, fc, proj;
} TxtBlock;

/* pre-norm layer: self attention, cross attention, FFN (fusion and geometry
 * encoders); post-norm decoder layer (norm2 self, catext text, norm1 image,
 * norm3 FFN) */
typedef struct {
    Norm n1, n2, n3, ncat;
    Mha sa, ca, cat;
    Lin l1, l2;
} Layer;

static struct {
    /* vision */
    Lin patch;                 /* [1024][608], no bias */
    CUdeviceptr pos_embed;     /* f32 [577][1024] */
    Norm ln_pre;
    VitBlock vit[VIT_LAYERS];
    Lin dc0a, dc0b, c0_1, c0_3; /* neck level 0 (x4): deconv, gelu, deconv, 1x1, 3x3 */
    Lin dc1, c1_1, c1_3;        /* level 1 (x2) */
    Lin c2_1, c2_3;             /* level 2 (x1) */
    /* text */
    const float *tok_embed;    /* host f32 [49408][1024] (in the mapping) */
    const float *txt_pos;      /* host f32 [32][1024] */
    TxtBlock txt[TXT_LAYERS];
    Norm ln_final;
    Lin resizer;
    /* geometry encoder (CLS only) */
    const float *cls;          /* host f32 [256] */
    Lin geo_proj;
    Norm geo_norm, geo_out_norm;
    Layer geo[GEO_LAYERS];
    /* fusion encoder, decoder */
    Layer enc[ENC_LAYERS], dec[DEC_LAYERS];
    Norm dec_norm, pres_norm;
    const float *query_embed;  /* host f32 [200][256] */
    const float *ref_points;   /* host f32 [200][4] */
    const float *presence_tok; /* host f32 [256] */
    Lin rph1, rph2;            /* ref_point_head 512 -> 256 -> 256 */
    Lin bb1, bb2;              /* bbox_embed layers 0, 1 */
    CUdeviceptr bb3_w, bb3_b;  /* f32 [4][256], [4] */
    CUdeviceptr ph_w[3], ph_b[3]; /* presence head f32: 256 -> 256 -> 256 -> 1 */
    CUdeviceptr rpbx_w1, rpbx_b1, rpbx_w2, rpbx_b2, rpby_w1, rpby_b1, rpby_w2, rpby_b2;
    /* scoring */
    Lin pm1, pm2;              /* prompt_mlp 256 -> 2048 -> 256 */
    Norm pm_norm;
    CUdeviceptr pp_w, pp_b;    /* prompt_proj f32 */
    Lin hs_proj;
    /* segmentation head */
    Mha seg_ca;
    Norm seg_norm;
    Lin pd_conv[2];
    Norm pd_gn[2];
    Lin inst;                  /* instance_seg_head 1x1 */
    Lin me[3];                 /* mask_embed */
} W;

static void load_model(const char *path)
{
    double t0 = now_ms();
    gguf_open(&gg, path);
    const GgufKV *arch = gguf_kv(&gg, "general.architecture");
    uint64_t len = 0;
    const uint8_t *p = arch ? arch->val : NULL;
    const char *a = p ? gguf_str(&p, &len) : "";
    if (!arch || len != 4 || memcmp(a, "sam3", 4)) {
        fprintf(stderr, "%s: not a sam3 GGUF\n", path);
        exit(1);
    }
    if (gguf_u32(&gg, "sam3.image_size") != IMG || gguf_u32(&gg, "sam3.patch_size") != PATCH ||
        gguf_u32(&gg, "sam3.vision_layers") != VIT_LAYERS ||
        gguf_u32(&gg, "sam3.text_layers") != TXT_LAYERS) {
        fprintf(stderr, "%s: unexpected model configuration\n", path);
        exit(1);
    }
    stage_bytes = 64u << 20;
    stage = dalloc(stage_bytes);
    char n[512], m[600];
    const char *vb = "backbone.vision_backbone";

    /* patch embedding [1024][14][14][3] -> [1024][608] zero-padded */
    {
        snprintf(n, sizeof(n), "%s.trunk.patch_embed.proj.weight", vb);
        const float *w = host_f32(n, (size_t)VIT_D * PATCH_K);
        float *pad = calloc((size_t)VIT_D * PATCH_KPAD, 4);
        for (int o = 0; o < VIT_D; o++)
            memcpy(pad + (size_t)o * PATCH_KPAD, w + (size_t)o * PATCH_K, PATCH_K * 4);
        W.patch = (Lin){dalloc((size_t)VIT_D * PATCH_KPAD * 2), 0, VIT_D, PATCH_KPAD};
        to_f16(W.patch.w, pad, (size_t)VIT_D * PATCH_KPAD);
        free(pad);
        weight_bytes += (size_t)VIT_D * PATCH_KPAD * 2;
    }
    snprintf(n, sizeof(n), "%s.trunk.pos_embed", vb);
    W.pos_embed = up_f32(n, (size_t)(WIN * WIN + 1) * VIT_D);
    snprintf(n, sizeof(n), "%s.trunk.ln_pre", vb);
    W.ln_pre = norm(n, VIT_D);
    for (int i = 0; i < VIT_LAYERS; i++) {
        VitBlock *b = &W.vit[i];
#define VB(field, suffix) snprintf(n, sizeof(n), "%s.trunk.blocks.%d.%s", vb, i, suffix)
        VB(n1, "norm1"); b->n1 = norm(n, VIT_D);
        VB(n2, "norm2"); b->n2 = norm(n, VIT_D);
        VB(qkv, "attn.qkv"); b->qkv = lin(n, 3 * VIT_D, VIT_D, 1);
        VB(proj, "attn.proj"); b->proj = lin(n, VIT_D, VIT_D, 1);
        VB(fc1, "mlp.fc1"); b->fc1 = lin(n, VIT_FF, VIT_D, 1);
        VB(fc2, "mlp.fc2"); b->fc2 = lin(n, VIT_D, VIT_FF, 1);
#undef VB
    }
    /* neck: a 2x2 stride-2 deconvolution is a GEMM to [4 * out] columns
     * (o, dy, dx); its bias repeats 4 times */
    struct { Lin *l; const char *name; unsigned o, i, taps; int deconv; } neck[] = {
        {&W.dc0a, "convs.0.dconv_2x2_0", 512, 1024, 1, 1},
        {&W.dc0b, "convs.0.dconv_2x2_1", 256, 512, 1, 1},
        {&W.c0_1, "convs.0.conv_1x1", 256, 256, 1, 0},
        {&W.c0_3, "convs.0.conv_3x3", 256, 256, 9, 0},
        {&W.dc1, "convs.1.dconv_2x2", 512, 1024, 1, 1},
        {&W.c1_1, "convs.1.conv_1x1", 256, 512, 1, 0},
        {&W.c1_3, "convs.1.conv_3x3", 256, 256, 9, 0},
        {&W.c2_1, "convs.2.conv_1x1", 256, 1024, 1, 0},
        {&W.c2_3, "convs.2.conv_3x3", 256, 256, 9, 0},
    };
    for (size_t i = 0; i < sizeof(neck) / sizeof(neck[0]); i++) {
        snprintf(n, sizeof(n), "%s.%s", vb, neck[i].name);
        if (!neck[i].deconv) {
            *neck[i].l = lin(n, neck[i].o, neck[i].i * neck[i].taps, 1);
            continue;
        }
        *neck[i].l = lin(n, neck[i].o * 4, neck[i].i, 0);
        snprintf(m, sizeof(m), "%s.bias", n);
        const float *b = host_f32(m, neck[i].o);
        float *b4 = malloc(neck[i].o * 16);
        for (unsigned o = 0; o < neck[i].o * 4; o++)
            b4[o] = b[o / 4];
        neck[i].l->b = dalloc(neck[i].o * 16);
        CHECK(cuMemcpyHtoD(neck[i].l->b, b4, neck[i].o * 16));
        free(b4);
    }

    /* text encoder */
    const char *te = "backbone.language_backbone.encoder";
    snprintf(n, sizeof(n), "%s.token_embedding.weight", te);
    W.tok_embed = host_f32(n, (size_t)CLIP_VOCAB * TXT_D);
    snprintf(n, sizeof(n), "%s.positional_embedding", te);
    W.txt_pos = host_f32(n, (size_t)TXT_LEN * TXT_D);
    for (int i = 0; i < TXT_LAYERS; i++) {
        TxtBlock *b = &W.txt[i];
        char q[600], k[600], v[600];
        snprintf(n, sizeof(n), "%s.transformer.resblocks.%d", te, i);
        snprintf(m, sizeof(m), "%s.ln_1", n); b->ln1 = norm(m, TXT_D);
        snprintf(m, sizeof(m), "%s.ln_2", n); b->ln2 = norm(m, TXT_D);
        snprintf(q, sizeof(q), "%s.attn.query_proj", n);
        snprintf(k, sizeof(k), "%s.attn.key_proj", n);
        snprintf(v, sizeof(v), "%s.attn.value_proj", n);
        const char *qkv[] = {q, k, v};
        b->qkv = lin_cat(qkv, 3, TXT_D, TXT_D);
        snprintf(m, sizeof(m), "%s.attn.out_proj", n); b->o = lin(m, TXT_D, TXT_D, 1);
        snprintf(m, sizeof(m), "%s.mlp.c_fc", n); b->fc = lin(m, TXT_FF, TXT_D, 1);
        snprintf(m, sizeof(m), "%s.mlp.c_proj", n); b->proj = lin(m, TXT_D, TXT_FF, 1);
    }
    snprintf(n, sizeof(n), "%s.ln_final", te);
    W.ln_final = norm(n, TXT_D);
    W.resizer = lin("backbone.language_backbone.resizer", D, TXT_D, 1);

    /* geometry encoder, fusion encoder, decoder layers */
    W.cls = host_f32("geometry_encoder.cls_embed.weight", D);
    W.geo_proj = lin("geometry_encoder.final_proj", D, D, 1);
    W.geo_norm = norm("geometry_encoder.norm", D);
    W.geo_out_norm = norm("geometry_encoder.encode_norm", D);
    for (int s = 0; s < 3; s++) {
        int nl = s == 0 ? GEO_LAYERS : s == 1 ? ENC_LAYERS : DEC_LAYERS;
        for (int i = 0; i < nl; i++) {
            Layer *L = s == 0 ? &W.geo[i] : s == 1 ? &W.enc[i] : &W.dec[i];
            if (s == 0)
                snprintf(n, sizeof(n), "geometry_encoder.encode.%d", i);
            else
                snprintf(n, sizeof(n), "transformer.%s.layers.%d", s == 1 ? "encoder" : "decoder",
                         i);
            snprintf(m, sizeof(m), "%s.norm1", n); L->n1 = norm(m, D);
            snprintf(m, sizeof(m), "%s.norm2", n); L->n2 = norm(m, D);
            snprintf(m, sizeof(m), "%s.norm3", n); L->n3 = norm(m, D);
            snprintf(m, sizeof(m), "%s.self_attn", n); L->sa = mha(m);
            snprintf(m, sizeof(m), "%s.%s", n, s == 2 ? "cross_attn" : "cross_attn_image");
            L->ca = mha(m);
            if (s == 2) {
                snprintf(m, sizeof(m), "%s.ca_text", n); L->cat = mha(m);
                snprintf(m, sizeof(m), "%s.catext_norm", n); L->ncat = norm(m, D);
            }
            snprintf(m, sizeof(m), "%s.linear1", n); L->l1 = lin(m, FF, D, 1);
            snprintf(m, sizeof(m), "%s.linear2", n); L->l2 = lin(m, D, FF, 1);
        }
    }
    const char *dc = "transformer.decoder";
    snprintf(n, sizeof(n), "%s.norm", dc); W.dec_norm = norm(n, D);
    snprintf(n, sizeof(n), "%s.presence_token_out_norm", dc); W.pres_norm = norm(n, D);
    snprintf(n, sizeof(n), "%s.query_embed.weight", dc); W.query_embed = host_f32(n, NQ * D);
    snprintf(n, sizeof(n), "%s.reference_points.weight", dc); W.ref_points = host_f32(n, NQ * 4);
    snprintf(n, sizeof(n), "%s.presence_token.weight", dc); W.presence_tok = host_f32(n, D);
    snprintf(n, sizeof(n), "%s.ref_point_head.layers.0", dc); W.rph1 = lin(n, D, 2 * D, 1);
    snprintf(n, sizeof(n), "%s.ref_point_head.layers.1", dc); W.rph2 = lin(n, D, D, 1);
    snprintf(n, sizeof(n), "%s.bbox_embed.layers.0", dc); W.bb1 = lin(n, D, D, 1);
    snprintf(n, sizeof(n), "%s.bbox_embed.layers.1", dc); W.bb2 = lin(n, D, D, 1);
    snprintf(n, sizeof(n), "%s.bbox_embed.layers.2.weight", dc); W.bb3_w = up_f32(n, 4 * D);
    snprintf(n, sizeof(n), "%s.bbox_embed.layers.2.bias", dc); W.bb3_b = up_f32(n, 4);
    for (int i = 0; i < 3; i++) {
        snprintf(n, sizeof(n), "%s.presence_token_head.layers.%d.weight", dc, i);
        W.ph_w[i] = up_f32(n, i < 2 ? D * D : D);
        snprintf(n, sizeof(n), "%s.presence_token_head.layers.%d.bias", dc, i);
        W.ph_b[i] = up_f32(n, i < 2 ? D : 1);
    }
    CUdeviceptr *rpb[] = {&W.rpbx_w1, &W.rpbx_b1, &W.rpbx_w2, &W.rpbx_b2,
                          &W.rpby_w1, &W.rpby_b1, &W.rpby_w2, &W.rpby_b2};
    for (int i = 0; i < 8; i++) {
        snprintf(n, sizeof(n), "%s.boxRPB_embed_%c.layers.%d.%s", dc, i < 4 ? 'x' : 'y',
                 (i / 2) & 1, i & 1 ? "bias" : "weight");
        static const size_t cnt[4] = {256 * 2, 256, 8 * 256, 8};
        *rpb[i] = up_f32(n, cnt[i & 3]);
    }

    /* scoring and segmentation heads */
    W.pm1 = lin("dot_prod_scoring.prompt_mlp.layers.0", FF, D, 1);
    W.pm2 = lin("dot_prod_scoring.prompt_mlp.layers.1", D, FF, 1);
    W.pm_norm = norm("dot_prod_scoring.prompt_mlp.out_norm", D);
    W.pp_w = up_f32("dot_prod_scoring.prompt_proj.weight", D * D);
    W.pp_b = up_f32("dot_prod_scoring.prompt_proj.bias", D);
    W.hs_proj = lin("dot_prod_scoring.hs_proj", D, D, 1);
    W.seg_ca = mha("segmentation_head.cross_attend_prompt");
    W.seg_norm = norm("segmentation_head.cross_attn_norm", D);
    for (int i = 0; i < 2; i++) {
        snprintf(n, sizeof(n), "segmentation_head.pixel_decoder.conv_layers.%d", i);
        W.pd_conv[i] = lin(n, D, 9 * D, 1);
        snprintf(n, sizeof(n), "segmentation_head.pixel_decoder.norms.%d", i);
        W.pd_gn[i] = norm(n, D);
    }
    W.inst = lin("segmentation_head.instance_seg_head", D, D, 1);
    for (int i = 0; i < 3; i++) {
        snprintf(n, sizeof(n), "segmentation_head.mask_predictor.mask_embed.layers.%d", i);
        W.me[i] = lin(n, D, D, 1);
    }
    CHECK(cuCtxSynchronize());
    fprintf(stderr, "loaded %s: %.2f GB on the GPU in %.1f s\n", path, weight_bytes / 1e9,
            (now_ms() - t0) / 1e3);
}

/* ------------------------------------------------------------------------ */
/* Activations                                                              */

static struct {
    CUdeviceptr img;           /* f32 [1008][1008][3] */
    CUdeviceptr col;           /* f16 im2col: [S0*S0][9*256] (the largest) */
    CUdeviceptr x;             /* f32 [ROWS(NTOK)][1024] ViT / text stream */
    CUdeviceptr xn;            /* f16 [ROWS(NTOK)][1024] normed / cast */
    CUdeviceptr qkv;           /* f32 [NTOK][3072]; neck and encoder scratch */
    CUdeviceptr qh, kh, vh;    /* f16 [16][NTOK][64] attention heads */
    CUdeviceptr att;           /* f16 [ROWS(NTOK)][1024] attention output */
    CUdeviceptr hid;           /* f16 [ROWS(NTOK)][4736]; neck / pixel scratch */
    CUdeviceptr cs_win, cs_glob; /* RoPE cos/sin [576][32][2], [5184][32][2] */
    CUdeviceptr fpn0, fpn1, fpn2; /* f32 [288^2], [144^2], [ROWS(72^2)] x 256 */
    CUdeviceptr pos2;          /* f32 [NTOK][256] sine position of the 72^2 level */
    CUdeviceptr img16, imgpos16; /* f16 [ROWS(NTOK)][256] fpn2, fpn2 + pos2 */
    CUdeviceptr geo_row;       /* f32 [256] the geometry CLS token */
    CUdeviceptr prompt;        /* f32 [128][256]: text rows 0..31, CLS row 32 */
    CUdeviceptr prompt16;      /* f16 [128][256] */
    CUdeviceptr pmask;         /* u8 [64] 1 = padding token */
    CUdeviceptr s32a, s32b, s32c; /* f32 [256][512] small scratch */
    CUdeviceptr s16a, s16b, s16c; /* f16 [256][2048] small scratch */
    CUdeviceptr pkv;           /* f32 [128][512] prompt keys | values */
    CUdeviceptr tgt;           /* f32 [256][256] decoder: presence + queries */
    CUdeviceptr t16, tp16;     /* f16 [256][256] tgt, tgt + qpos */
    CUdeviceptr qpos;          /* f32 [256][256] row 0 (presence) stays 0 */
    CUdeviceptr ref, delta;    /* f32 [200][4] */
    CUdeviceptr rbx, rby;      /* f32 [8][201][72] box bias */
    CUdeviceptr hs32, hs16;    /* the normed last decoder output [200(256)][256] */
    CUdeviceptr pd32;          /* f32 [S0*S0][256] pixel decoder */
    CUdeviceptr pix16, inst16; /* f16 [S0*S0][256] */
    CUdeviceptr masks;         /* f32 [200][S0*S0] mask logits */
    CUdeviceptr logits, plogit; /* f32 [200], [1] */
    CUdeviceptr gn_acc;        /* f64 [8][2] */
} B;

static const char *dump_dir;

/* SAM3_DUMP: device f32 [n] -> DIR/name.f32 */
static void dump(const char *name, CUdeviceptr p, size_t n)
{
    if (!dump_dir)
        return;
    float *h = malloc(n * 4);
    CHECK(cuMemcpyDtoH(h, p, n * 4));
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", dump_dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(h, 4, n, f) != n)
        fprintf(stderr, "cannot write %s\n", path);
    if (f)
        fclose(f);
    free(h);
}

/* 2-D axial RoPE table for a side x side grid of positions scaled by s:
 * cs[(y*side + x)*32 + j] = cos, sin of (j < 16 ? x : y) * s * freq_(j % 16),
 * freq_i = 1 / 10000^(4i / 64) (computed in f32 like the PyTorch code) */
static CUdeviceptr rope_table(int side, float s)
{
    size_t n = (size_t)side * side * 32 * 2;
    float *cs = malloc(n * 4);
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++)
            for (int j = 0; j < 32; j++) {
                float freq = 1.0f / powf(10000.0f, (float)(4 * (j % 16)) / 64.0f);
                float t = (float)(j < 16 ? x : y) * s;
                float a = t * freq;
                size_t o = (((size_t)y * side + x) * 32 + j) * 2;
                cs[o] = (float)cos((double)a);
                cs[o + 1] = (float)sin((double)a);
            }
    CUdeviceptr p = dalloc(n * 4);
    CHECK(cuMemcpyHtoD(p, cs, n * 4));
    free(cs);
    return p;
}

/* PositionEmbeddingSine(256, normalize) of a side x side map: channels
 * 0..127 encode the row, 128..255 the column; value i of coordinate v
 * (1-based, normalized to 2 pi) is sin(v / T_i) for even i, cos for odd,
 * T_i = 10000^(2 floor(i / 2) / 128) */
static CUdeviceptr sine_pos(int side)
{
    float *p = malloc((size_t)side * side * D * 4);
    const float two_pi = 6.283185307179586f;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++)
            for (int i = 0; i < 128; i++) {
                float dim_t = powf(10000.0f, (float)(2 * (i / 2)) / 128.0f);
                float ey = (float)(y + 1) / ((float)side + 1e-6f) * two_pi;
                float ex = (float)(x + 1) / ((float)side + 1e-6f) * two_pi;
                double vy = (double)(ey / dim_t), vx = (double)(ex / dim_t);
                float *o = p + ((size_t)y * side + x) * D;
                o[i] = (float)(i % 2 ? cos(vy) : sin(vy));
                o[128 + i] = (float)(i % 2 ? cos(vx) : sin(vx));
            }
    CUdeviceptr d = dalloc((size_t)side * side * D * 4);
    CHECK(cuMemcpyHtoD(d, p, (size_t)side * side * D * 4));
    free(p);
    return d;
}

static void alloc_buffers(void)
{
    size_t rows = ROWS(NTOK);
    B.img = dalloc((size_t)IMG * IMG * 3 * 4);
    B.col = dalloc((size_t)S0 * S0 * 9 * D * 2);
    B.x = dalloc(rows * VIT_D * 4);
    B.xn = dalloc(rows * VIT_D * 2);
    B.qkv = dalloc((size_t)NTOK * 3 * VIT_D * 4);
    B.qh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
    B.kh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
    B.vh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
    B.att = dalloc(rows * VIT_D * 2);
    B.hid = dalloc(rows * VIT_FF * 2);
    B.cs_win = rope_table(WIN, 1.0f);
    B.cs_glob = rope_table(GRID, (float)WIN / GRID);
    B.fpn0 = dalloc((size_t)S0 * S0 * D * 4);
    B.fpn1 = dalloc((size_t)S1 * S1 * D * 4);
    B.fpn2 = dalloc(rows * D * 4);
    B.pos2 = sine_pos(GRID);
    B.img16 = dalloc(rows * D * 2);
    B.imgpos16 = dalloc(rows * D * 2);
    B.geo_row = dalloc(D * 4);
    B.prompt = dalloc(128 * D * 4);
    B.prompt16 = dalloc(128 * D * 2);
    B.pmask = dalloc(64);
    B.s32a = dalloc(256 * 512 * 4);
    B.s32b = dalloc(256 * 512 * 4);
    B.s32c = dalloc(256 * 512 * 4);
    B.s16a = dalloc(256 * FF * 2);
    B.s16b = dalloc(256 * FF * 2);
    B.s16c = dalloc(256 * FF * 2);
    B.pkv = dalloc(128 * 512 * 4);
    B.tgt = dalloc(256 * D * 4);
    B.t16 = dalloc(256 * D * 2);
    B.tp16 = dalloc(256 * D * 2);
    B.qpos = dalloc(256 * D * 4);
    B.ref = dalloc(NQ * 4 * 4);
    B.delta = dalloc(NQ * 4 * 4);
    B.rbx = dalloc(NH * (NQ + 1) * GRID * 4);
    B.rby = dalloc(NH * (NQ + 1) * GRID * 4);
    B.hs32 = dalloc(256 * D * 4);
    B.hs16 = dalloc(256 * D * 2);
    B.pd32 = dalloc((size_t)S0 * S0 * D * 4);
    B.pix16 = dalloc((size_t)S0 * S0 * D * 2);
    B.inst16 = dalloc((size_t)S0 * S0 * D * 2);
    B.masks = dalloc((size_t)NQ * S0 * S0 * 4);
    B.logits = dalloc(256 * 4);
    B.plogit = dalloc(16);
    B.gn_acc = dalloc(16 * 8);
}

/* Multi-head attention of the 256-wide modules (8 heads of 32): q, k, v are
 * f32 rows (row stride ld, starting at column col) of lq / lk tokens;
 * out f16 [lq][256]. */
static void attn256(CUdeviceptr q, unsigned qld, unsigned qcol, CUdeviceptr k, unsigned kld,
                    unsigned kcol, CUdeviceptr v, unsigned vld, unsigned vcol, unsigned lq,
                    unsigned lk, CUdeviceptr out, unsigned flags, CUdeviceptr kmask,
                    CUdeviceptr bx, CUdeviceptr by)
{
    unsigned lqp = PAD64(lq), lkp = PAD64(lk);
    split_heads(q, qld, qcol, B.qh, lq, lqp, NH, 32);
    split_heads(k, kld, kcol, B.kh, lk, lkp, NH, 32);
    split_heads(v, vld, vcol, B.vh, lk, lkp, NH, 32);
    Att a = {lq, lk, lqp, lkp, NH, NH, 32, D, flags, 0.17677669529663687f, kmask, bx, by, GRID,
             0, 1, 1};
    flash(B.qh, B.kh, B.vh, out, &a);
}

/* GroupNorm(8, 256) + ReLU of x [n][256] (f32), into y32 / y16 */
static void group_norm_relu(CUdeviceptr x, unsigned n, const Norm *nm, CUdeviceptr y32,
                            CUdeviceptr y16)
{
    CHECK(cuMemsetD8(B.gn_acc, 0, 16 * 8));
    void *a1[] = {&x, &B.gn_acc, &n};
    CHECK(cuLaunchKernel(k_gn_stats, max_blocks, 1, 1, 256, 1, 1, 0, NULL, a1, NULL));
    float eps = 1e-5f;
    void *a2[] = {&x, &B.gn_acc, (void *)&nm->w, (void *)&nm->b, &y32, &y16, &n, &eps};
    CHECK(cuLaunchKernel(k_gn_apply, max_blocks, 1, 1, 256, 1, 1, 0, NULL, a2, NULL));
}

/* ------------------------------------------------------------------------ */
/* Forward stages                                                           */

static int is_global(int i)
{
    return i == 7 || i == 15 || i == 23 || i == 31;
}

/* B.img -> fpn0..2 (+ img16, imgpos16, geo_row): everything that depends on
 * the image only */
static void vision(void)
{
    unsigned side = IMG, p = PATCH, c = 3, gw = GRID, kpad = PATCH_KPAD;
    void *a0[] = {&B.img, &B.col, &side, &p, &c, &gw, &kpad};
    launch_1d(k_im2col_patch, NTOK * PATCH_KPAD / 2, a0);
    gemm_raw(B.col, W.patch.w, 0, B.x, NTOK, VIT_D, PATCH_KPAD, VIT_D, G_F32);
    dump("patch_embed", B.x, (size_t)NTOK * VIT_D);
    unsigned t = NTOK, cc = VIT_D, ps = WIN;
    void *a1[] = {&B.x, &W.pos_embed, &t, &cc, &gw, &ps};
    launch_1d(k_add_pos, NTOK * VIT_D / 4, a1);
    layernorm(B.x, &W.ln_pre, B.x, 0, 0, 0, NTOK, VIT_D);
    dump("ln_pre", B.x, (size_t)NTOK * VIT_D);
    for (int i = 0; i < VIT_LAYERS; i++) {
        const VitBlock *b = &W.vit[i];
        int g = is_global(i);
        layernorm(B.x, &b->n1, 0, B.xn, 0, 0, NTOK, VIT_D);
        gemm(B.xn, &b->qkv, B.qkv, NTOK, G_F32);
        unsigned wsz = g ? 0 : WIN, h = VIT_H;
        CUdeviceptr cs = g ? B.cs_glob : B.cs_win;
        void *ar[] = {&B.qkv, &cs, &B.qh, &B.kh, &B.vh, &t, &gw, &wsz, &h};
        launch_1d(k_rope_vit, NTOK * VIT_H * 32, ar);
        unsigned l = g ? NTOK : WIN * WIN;
        Att a = {l, l, l, l, g ? VIT_H : VIT_H * 9, VIT_H, 64, VIT_D, 0, 0.125f, 0, 0, 0, 0,
                 wsz, GRID / WIN, GRID};
        flash(B.qh, B.kh, B.vh, B.att, &a);
        gemm(B.att, &b->proj, B.x, NTOK, G_ADD);
        layernorm(B.x, &b->n2, 0, B.xn, 0, 0, NTOK, VIT_D);
        gemm(B.xn, &b->fc1, B.hid, NTOK, G_GELU);
        gemm(B.hid, &b->fc2, B.x, NTOK, G_ADD);
        if (dump_dir && (i == 0 || g)) {
            char name[32];
            snprintf(name, sizeof(name), "vit_block%d", i);
            dump(name, B.x, (size_t)NTOK * VIT_D);
        }
    }

    /* neck; scratch: hid (A) and qkv (B), both >= 21.2M halves */
    CUdeviceptr nA = B.hid, nB = B.qkv;
    add_cast(B.x, 0, B.xn, 0, NTOK * VIT_D);
    gemm(B.xn, &W.dc0a, nA, NTOK, G_GELU);                 /* 72^2 x (512 x 4) */
    depth2space(nA, nB, GRID, GRID, 512);                   /* 144^2 x 512 */
    gemm(nB, &W.dc0b, nA, S1 * S1, G_F16);                  /* 144^2 x (256 x 4) */
    depth2space(nA, nB, S1, S1, 256);                       /* 288^2 x 256 */
    gemm(nB, &W.c0_1, nA, S0 * S0, G_F16);
    im2col3(nA, B.col, S0, S0, D);
    gemm(B.col, &W.c0_3, B.fpn0, S0 * S0, G_F32);
    gemm(B.xn, &W.dc1, nA, NTOK, G_F16);
    depth2space(nA, nB, GRID, GRID, 512);
    gemm(nB, &W.c1_1, nA, S1 * S1, G_F16);
    im2col3(nA, B.col, S1, S1, D);
    gemm(B.col, &W.c1_3, B.fpn1, S1 * S1, G_F32);
    gemm(B.xn, &W.c2_1, nA, NTOK, G_F16);
    im2col3(nA, B.col, GRID, GRID, D);
    gemm(B.col, &W.c2_3, B.fpn2, NTOK, G_F32);
    add_cast(B.fpn2, B.pos2, B.imgpos16, B.img16, NTOK * D);
    dump("fpn0", B.fpn0, (size_t)S0 * S0 * D);
    dump("fpn1", B.fpn1, (size_t)S1 * S1 * D);
    dump("fpn2", B.fpn2, (size_t)NTOK * D);

    /* geometry encoder without boxes: the CLS token alone, through 3
     * pre-norm layers (self attention over one token is its value
     * projection; cross attention to the image features + position) */
    CUdeviceptr gx = B.s32a, g32 = B.s32b, g16 = B.s16a, gv16 = B.s16b, gh16 = B.s16c;
    CHECK(cuMemcpyHtoD(g32, W.cls, D * 4));
    add_cast(g32, 0, g16, 0, D);
    gemm(g16, &W.geo_proj, g32, 1, G_F32);
    layernorm(g32, &W.geo_norm, gx, 0, 0, 0, 1, D);
    for (int l = 0; l < GEO_LAYERS; l++) {
        const Layer *L = &W.geo[l];
        layernorm(gx, &L->n1, 0, g16, 0, 0, 1, D);
        gemm(g16, &L->sa.v, gv16, 1, G_F16);
        gemm(gv16, &L->sa.o, gx, 1, G_ADD);
        layernorm(gx, &L->n2, 0, g16, 0, 0, 1, D);
        gemm(g16, &L->ca.q, g32, 1, G_F32);
        CUdeviceptr kf = B.qkv, vf = B.qkv + (size_t)NTOK * D * 4;
        gemm(B.imgpos16, &L->ca.k, kf, NTOK, G_F32);
        gemm(B.img16, &L->ca.v, vf, NTOK, G_F32);
        attn256(g32, D, 0, kf, D, 0, vf, D, 0, 1, NTOK, gv16, 0, 0, 0, 0);
        gemm(gv16, &L->ca.o, gx, 1, G_ADD);
        layernorm(gx, &L->n3, 0, g16, 0, 0, 1, D);
        gemm(g16, &L->l1, gh16, 1, G_RELU);
        gemm(gh16, &L->l2, gx, 1, G_ADD);
    }
    layernorm(gx, &W.geo_out_norm, B.geo_row, 0, 0, 0, 1, D);
    dump("geo_feats", B.geo_row, D);
}

/* prompt text -> B.prompt rows 0..31 (+ the CLS row 32), B.pmask; returns
 * the host mask too */
static void text(const int32_t *tok, uint8_t *mask)
{
    float *x0 = malloc((size_t)TXT_LEN * TXT_D * 4);
    for (int t = 0; t < TXT_LEN; t++)
        for (int i = 0; i < TXT_D; i++)
            x0[t * TXT_D + i] =
                W.tok_embed[(size_t)tok[t] * TXT_D + i] + W.txt_pos[(size_t)t * TXT_D + i];
    CHECK(cuMemcpyHtoD(B.x, x0, (size_t)TXT_LEN * TXT_D * 4));
    free(x0);
    for (int i = 0; i < TXT_LAYERS; i++) {
        const TxtBlock *b = &W.txt[i];
        layernorm(B.x, &b->ln1, 0, B.xn, 0, 0, TXT_LEN, TXT_D);
        gemm(B.xn, &b->qkv, B.qkv, TXT_LEN, G_F32);
        split_heads(B.qkv, 3 * TXT_D, 0, B.qh, TXT_LEN, 64, TXT_H, 64);
        split_heads(B.qkv, 3 * TXT_D, TXT_D, B.kh, TXT_LEN, 64, TXT_H, 64);
        split_heads(B.qkv, 3 * TXT_D, 2 * TXT_D, B.vh, TXT_LEN, 64, TXT_H, 64);
        Att a = {TXT_LEN, TXT_LEN, 64, 64, TXT_H, TXT_H, 64, TXT_D, 2, 0.125f, 0, 0, 0, 0, 0,
                 1, 1};
        flash(B.qh, B.kh, B.vh, B.att, &a);
        gemm(B.att, &b->o, B.x, TXT_LEN, G_ADD);
        layernorm(B.x, &b->ln2, 0, B.xn, 0, 0, TXT_LEN, TXT_D);
        gemm(B.xn, &b->fc, B.hid, TXT_LEN, G_GELU);
        gemm(B.hid, &b->proj, B.x, TXT_LEN, G_ADD);
    }
    layernorm(B.x, &W.ln_final, 0, B.xn, 0, 0, TXT_LEN, TXT_D);
    gemm(B.xn, &W.resizer, B.prompt, TXT_LEN, G_F32);
    dump("text_feats", B.prompt, (size_t)TXT_LEN * D);
    CHECK(cuMemcpyDtoD(B.prompt + (size_t)TXT_LEN * D * 4, B.geo_row, D * 4));
    for (int t = 0; t < TXT_LEN; t++)
        mask[t] = tok[t] == 0;
    mask[TXT_LEN] = 0;
    CHECK(cuMemcpyHtoD(B.pmask, mask, NPROMPT));
    add_cast(B.prompt, 0, B.prompt16, 0, NPROMPT * D);
}

/* fusion encoder: image tokens (fpn2) attend to each other (with position)
 * and to the prompt; result in B.x as f32 [NTOK][256] (the "memory") */
static void encoder(void)
{
    CUdeviceptr ex = B.x, en16 = B.xn, enp16 = B.xn + (size_t)ROWS(NTOK) * D * 2;
    CUdeviceptr qk = B.qkv, v = B.qkv + (size_t)NTOK * 2 * D * 4;
    CHECK(cuMemcpyDtoD(ex, B.fpn2, (size_t)NTOK * D * 4));
    for (int l = 0; l < ENC_LAYERS; l++) {
        const Layer *L = &W.enc[l];
        layernorm(ex, &L->n1, 0, en16, B.pos2, enp16, NTOK, D);
        gemm(enp16, &L->sa.qk, qk, NTOK, G_F32);
        gemm(en16, &L->sa.v, v, NTOK, G_F32);
        attn256(qk, 2 * D, 0, qk, 2 * D, D, v, D, 0, NTOK, NTOK, B.att, 0, 0, 0, 0);
        gemm(B.att, &L->sa.o, ex, NTOK, G_ADD);
        layernorm(ex, &L->n2, 0, en16, 0, 0, NTOK, D);
        gemm(en16, &L->ca.q, v, NTOK, G_F32);
        gemm(B.prompt16, &L->ca.kv, B.pkv, NPROMPT, G_F32);
        attn256(v, D, 0, B.pkv, 2 * D, 0, B.pkv, 2 * D, D, NTOK, NPROMPT, B.att, 1, B.pmask, 0,
                0);
        gemm(B.att, &L->ca.o, ex, NTOK, G_ADD);
        layernorm(ex, &L->n3, 0, en16, 0, 0, NTOK, D);
        gemm(en16, &L->l1, B.hid, NTOK, G_RELU);
        gemm(B.hid, &L->l2, ex, NTOK, G_ADD);
        if (l == 0)
            dump("enc_layer0", ex, (size_t)NTOK * D);
    }
    dump("memory", ex, (size_t)NTOK * D);
}

/* decoder: 200 queries + presence token over the memory; leaves B.ref (the
 * final boxes), B.hs32 / B.hs16 (normed queries) and B.plogit */
static void decoder(void)
{
    CUdeviceptr mem16 = B.xn, mempos16 = B.xn + (size_t)ROWS(NTOK) * D * 2;
    CUdeviceptr dk = B.qkv, dv = B.qkv + (size_t)NTOK * D * 4;
    add_cast(B.x, B.pos2, mempos16, mem16, NTOK * D);

    float *t0 = malloc((NQ + 1) * D * 4), r0[NQ * 4];
    memcpy(t0, W.presence_tok, D * 4);
    memcpy(t0 + D, W.query_embed, NQ * D * 4);
    CHECK(cuMemcpyHtoD(B.tgt, t0, (NQ + 1) * D * 4));
    free(t0);
    for (int i = 0; i < NQ * 4; i++)
        r0[i] = 1.0f / (1.0f + expf(-W.ref_points[i]));
    CHECK(cuMemcpyHtoD(B.ref, r0, sizeof(r0)));

    const unsigned nt = NQ + 1;
    for (int l = 0; l < DEC_LAYERS; l++) {
        const Layer *L = &W.dec[l];
        /* query position from the current boxes, and the box bias */
        unsigned nq = NQ, g = GRID;
        void *as[] = {&B.ref, &B.s16a, &nq};
        launch_1d(k_box_sine, NQ * 512, as);
        gemm(B.s16a, &W.rph1, B.s16b, NQ, G_RELU);
        gemm_raw(B.s16b, W.rph2.w, W.rph2.b, B.qpos + D * 4, NQ, D, D, D, G_F32);
        void *ab[] = {&B.ref, &W.rpbx_w1, &W.rpbx_b1, &W.rpbx_w2, &W.rpbx_b2, &W.rpby_w1,
                      &W.rpby_b1, &W.rpby_w2, &W.rpby_b2, &B.rbx, &B.rby, &nq, &g};
        CHECK(cuLaunchKernel(k_rpb, (NQ * GRID * 2 + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, ab,
                             NULL));
        add_cast(B.tgt, B.qpos, B.tp16, B.t16, nt * D);
        /* self attention */
        gemm(B.tp16, &L->sa.qk, B.s32a, nt, G_F32);
        gemm(B.t16, &L->sa.v, B.s32b, nt, G_F32);
        attn256(B.s32a, 2 * D, 0, B.s32a, 2 * D, D, B.s32b, D, 0, nt, nt, B.s16a, 0, 0, 0, 0);
        gemm(B.s16a, &L->sa.o, B.tgt, nt, G_ADD);
        layernorm(B.tgt, &L->n2, B.tgt, 0, B.qpos, B.tp16, nt, D);
        /* text cross attention */
        gemm(B.tp16, &L->cat.q, B.s32a, nt, G_F32);
        gemm(B.prompt16, &L->cat.kv, B.pkv, NPROMPT, G_F32);
        attn256(B.s32a, D, 0, B.pkv, 2 * D, 0, B.pkv, 2 * D, D, nt, NPROMPT, B.s16a, 1, B.pmask,
                0, 0);
        gemm(B.s16a, &L->cat.o, B.tgt, nt, G_ADD);
        layernorm(B.tgt, &L->ncat, B.tgt, 0, B.qpos, B.tp16, nt, D);
        /* image cross attention with the box bias */
        gemm(B.tp16, &L->ca.q, B.s32a, nt, G_F32);
        gemm(mempos16, &L->ca.k, dk, NTOK, G_F32);
        gemm(mem16, &L->ca.v, dv, NTOK, G_F32);
        attn256(B.s32a, D, 0, dk, D, 0, dv, D, 0, nt, NTOK, B.s16a, 4, 0, B.rbx, B.rby);
        gemm(B.s16a, &L->ca.o, B.tgt, nt, G_ADD);
        layernorm(B.tgt, &L->n1, B.tgt, B.t16, 0, 0, nt, D);
        /* feed-forward */
        gemm(B.t16, &L->l1, B.s16b, nt, G_RELU);
        gemm(B.s16b, &L->l2, B.tgt, nt, G_ADD);
        layernorm(B.tgt, &L->n3, B.tgt, 0, 0, 0, nt, D);
        /* box refinement from the normed queries (rows 1..200) */
        layernorm(B.tgt + D * 4, &W.dec_norm, B.hs32, B.hs16, 0, 0, NQ, D);
        gemm(B.hs16, &W.bb1, B.s16a, NQ, G_RELU);
        gemm(B.s16a, &W.bb2, B.s32a, NQ, G_RELU32);
        linear_small(B.s32a, D, W.bb3_w, W.bb3_b, B.delta, 4, NQ, 4, D, 0);
        unsigned n4 = NQ * 4;
        void *au[] = {&B.ref, &B.delta, &n4};
        CHECK(cuLaunchKernel(k_box_update, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, au, NULL));
    }
    dump("dec_hs", B.hs32, (size_t)NQ * D);
    dump("pred_boxes", B.ref, NQ * 4);
    /* presence logit from the last layer's presence token */
    layernorm(B.tgt, &W.pres_norm, B.s32a, 0, 0, 0, 1, D);
    linear_small(B.s32a, D, W.ph_w[0], W.ph_b[0], B.s32b, D, 1, D, D, 1);
    linear_small(B.s32b, D, W.ph_w[1], W.ph_b[1], B.s32c, D, 1, D, D, 1);
    linear_small(B.s32c, D, W.ph_w[2], W.ph_b[2], B.plogit, 1, 1, 1, D, 0);
    dump("presence_logit", B.plogit, 1);
}

/* dot-product scores of the queries against the pooled prompt -> B.logits
 * (raw, before the / 16 and clamp done on the host) */
static void scoring(void)
{
    CUdeviceptr p2 = B.s32a, pooled = B.s32b, pp = B.s32c;
    CHECK(cuMemcpyDtoD(p2, B.prompt, NPROMPT * D * 4));
    gemm(B.prompt16, &W.pm1, B.s16a, NPROMPT, G_RELU);
    gemm(B.s16a, &W.pm2, p2, NPROMPT, G_ADD);
    layernorm(p2, &W.pm_norm, p2, 0, 0, 0, NPROMPT, D);
    unsigned l = NPROMPT, c = D;
    void *am[] = {&p2, &B.pmask, &pooled, &l, &c};
    CHECK(cuLaunchKernel(k_masked_mean, 1, 1, 1, 256, 1, 1, 0, NULL, am, NULL));
    linear_small(pooled, D, W.pp_w, W.pp_b, pp, D, 1, D, D, 0);
    gemm(B.hs16, &W.hs_proj, B.s32a, NQ, G_F32); /* p2 is no longer needed */
    linear_small(B.s32a, D, pp, 0, B.logits, 1, NQ, 1, D, 0);
    dump("pred_logits_raw", B.logits, NQ);
}

/* segmentation head: mask logits of every query at 288 x 288 -> B.masks */
static void masks(void)
{
    CUdeviceptr ex = B.x, en16 = B.xn, q = B.qkv;
    layernorm(ex, &W.seg_norm, 0, en16, 0, 0, NTOK, D);
    gemm(en16, &W.seg_ca.q, q, NTOK, G_F32);
    gemm(B.prompt16, &W.seg_ca.kv, B.pkv, NPROMPT, G_F32);
    attn256(q, D, 0, B.pkv, 2 * D, 0, B.pkv, 2 * D, D, NTOK, NPROMPT, B.att, 1, B.pmask, 0, 0);
    gemm(B.att, &W.seg_ca.o, ex, NTOK, G_ADD); /* memory + attention */
    /* pixel decoder: 72 -> 144 -> 288 */
    unsigned h = GRID, w = GRID, c = D;
    void *a1[] = {&ex, &B.fpn1, &B.hid, &h, &w, &c};
    launch_1d(k_upadd, S1 * S1 * D / 4, a1);
    im2col3(B.hid, B.col, S1, S1, D);
    gemm(B.col, &W.pd_conv[0], B.pd32, S1 * S1, G_F32);
    group_norm_relu(B.pd32, S1 * S1, &W.pd_gn[0], B.pd32, 0);
    h = w = S1;
    void *a2[] = {&B.pd32, &B.fpn0, &B.hid, &h, &w, &c};
    launch_1d(k_upadd, S0 * S0 * D / 4, a2);
    im2col3(B.hid, B.col, S0, S0, D);
    gemm(B.col, &W.pd_conv[1], B.pd32, S0 * S0, G_F32);
    group_norm_relu(B.pd32, S0 * S0, &W.pd_gn[1], dump_dir ? B.pd32 : 0, B.pix16);
    dump("pixel_embed", B.pd32, (size_t)S0 * S0 * D);
    gemm(B.pix16, &W.inst, B.inst16, S0 * S0, G_F16);
    /* mask embedding of each query, then its dot product with every pixel */
    gemm(B.hs16, &W.me[0], B.s16a, NQ, G_RELU);
    gemm(B.s16a, &W.me[1], B.s16b, NQ, G_RELU);
    gemm(B.s16b, &W.me[2], B.s16c, NQ, G_F16);
    gemm_raw(B.s16c, B.inst16, 0, B.masks, NQ, S0 * S0, D, S0 * S0, G_F32);
    dump("pred_masks", B.masks, (size_t)NQ * S0 * S0);
}

/* ------------------------------------------------------------------------ */
/* Detection                                                                */

typedef struct {
    int query;
    float score, box[4]; /* pixels: x0, y0, x1, y1 */
} Det;

static double t_stage[6]; /* ms: text, encoder, decoder, scoring, masks, resize */

static void sync_time(double *acc, double *t)
{
    CHECK(cuCtxSynchronize());
    double n = now_ms();
    *acc += n - *t;
    *t = n;
}

/* Runs one prompt on the current image features. Fills det (up to NQ) and,
 * when masks_out is given, *masks_out = u8 [n][ih][iw]. Returns n. */
static int detect(const ClipTokenizer *tok, const char *prompt, float thr, int iw, int ih,
                  Det *det, uint8_t **masks_out)
{
    int32_t ids[TXT_LEN];
    uint8_t pm[NPROMPT];
    clip_tokenize(tok, prompt, ids, TXT_LEN);
    double t = now_ms();
    text(ids, pm);
    sync_time(&t_stage[0], &t);
    encoder();
    sync_time(&t_stage[1], &t);
    decoder();
    sync_time(&t_stage[2], &t);
    scoring();
    sync_time(&t_stage[3], &t);
    masks();
    sync_time(&t_stage[4], &t);

    float logit[NQ], plogit, ref[NQ * 4];
    CHECK(cuMemcpyDtoH(logit, B.logits, sizeof(logit)));
    CHECK(cuMemcpyDtoH(&plogit, B.plogit, 4));
    CHECK(cuMemcpyDtoH(ref, B.ref, sizeof(ref)));
    float presence = 1.0f / (1.0f + expf(-plogit));
    int n = 0;
    uint32_t sel[NQ];
    for (int q = 0; q < NQ; q++) {
        float s = logit[q] / 16.0f; /* DotProductScoring: / sqrt(256), clamp 12 */
        s = s < -12 ? -12 : s > 12 ? 12 : s;
        float p = presence / (1.0f + expf(-s));
        if (p <= thr)
            continue;
        Det *d = &det[n];
        d->query = q;
        d->score = p;
        const float *r = ref + q * 4;
        d->box[0] = (r[0] - 0.5f * r[2]) * iw;
        d->box[1] = (r[1] - 0.5f * r[3]) * ih;
        d->box[2] = (r[0] + 0.5f * r[2]) * iw;
        d->box[3] = (r[1] + 0.5f * r[3]) * ih;
        sel[n++] = (uint32_t)q;
    }
    if (masks_out && n) {
        size_t bytes = (size_t)n * iw * ih;
        CUdeviceptr dsel = dalloc(n * 4), dout = dalloc(bytes);
        CHECK(cuMemcpyHtoD(dsel, sel, n * 4));
        unsigned nn = n, s = S0, h = ih, w = iw;
        void *a[] = {&B.masks, &dsel, &dout, &nn, &s, &h, &w};
        launch_1d(k_mask_resize, (unsigned)bytes, a);
        *masks_out = malloc(bytes);
        CHECK(cuMemcpyDtoH(*masks_out, dout, bytes));
        CHECK(cuMemFree(dsel));
        CHECK(cuMemFree(dout));
    } else if (masks_out) {
        *masks_out = NULL;
    }
    sync_time(&t_stage[5], &t);
    return n;
}

/* ------------------------------------------------------------------------ */
/* Self-tests: every kernel against a CPU reference (no model needed)       */

static uint16_t f2h(float f) /* round to nearest even */
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 0xFF)
        return (uint16_t)(sign | 0x7C00 | (m ? 0x200 : 0));
    int ee = (int)e - 127 + 15;
    if (ee >= 31)
        return (uint16_t)(sign | 0x7C00);
    if (ee <= 0) {
        if (ee < -10)
            return (uint16_t)sign;
        m |= 0x800000;
        int shift = 14 - ee;
        uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1)))
            h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)ee << 10) | (m >> 13), rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1)))
        h++;
    return (uint16_t)(sign | h);
}

static float h2f(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) {
        if (!m) {
            x = sign;
        } else {
            e = 127 - 15 + 1;
            while (!(m & 0x400))
                m <<= 1, e--;
            x = sign | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) {
        x = sign | 0x7F800000 | (m << 13);
    } else {
        x = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static float rf16(float f) { return h2f(f2h(f)); } /* f16 rounding */

static int n_fail, n_tests;

static void report(const char *name, double err, double tol)
{
    int ok = err <= tol && err == err;
    n_tests++;
    n_fail += !ok;
    printf("  %-44s max err %.3g (tol %.3g)  %s\n", name, err, tol, ok ? "ok" : "FAIL");
}

static float *rand_f32(size_t n, float lo, float hi)
{
    float *p = malloc(n * 4);
    for (size_t i = 0; i < n; i++)
        p[i] = rng_uniform(lo, hi);
    return p;
}

static CUdeviceptr up(const void *h, size_t bytes)
{
    CUdeviceptr d = dalloc(bytes);
    CHECK(cuMemcpyHtoD(d, h, bytes));
    return d;
}

static CUdeviceptr up16(const float *h, size_t n) /* f32 host -> f16 device */
{
    uint16_t *t = malloc(n * 2);
    for (size_t i = 0; i < n; i++)
        t[i] = f2h(h[i]);
    CUdeviceptr d = up(t, n * 2);
    free(t);
    return d;
}

static float *down(CUdeviceptr d, size_t n)
{
    float *h = malloc(n * 4);
    CHECK(cuMemcpyDtoH(h, d, n * 4));
    return h;
}

static float *down16(CUdeviceptr d, size_t n)
{
    uint16_t *t = malloc(n * 2);
    float *h = malloc(n * 4);
    CHECK(cuMemcpyDtoH(t, d, n * 2));
    for (size_t i = 0; i < n; i++)
        h[i] = h2f(t[i]);
    free(t);
    return h;
}

/* max |a - b| / (1 + |b|) */
static double cmp(const float *a, const float *b, size_t n)
{
    double e = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]) / (1.0 + fabs((double)b[i]));
        if (d > e || d != d)
            e = d != d ? INFINITY : d;
    }
    return e;
}

static float gelu_ref(float x) { return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f))); }

static void test_gemm(void)
{
    const unsigned M = 333, N = 384, K = 608;
    float *a = rand_f32((size_t)ROWS(M) * K, -1, 1), *b = rand_f32((size_t)N * K, -1, 1);
    float *bias = rand_f32(N, -1, 1), *c0 = rand_f32((size_t)M * N, -1, 1);
    for (size_t i = 0; i < (size_t)ROWS(M) * K; i++)
        a[i] = rf16(a[i]);
    for (size_t i = 0; i < (size_t)N * K; i++)
        b[i] = rf16(b[i]);
    float *ref = malloc((size_t)M * N * 4);
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            double s = 0;
            for (unsigned k = 0; k < K; k++)
                s += (double)a[(size_t)m * K + k] * b[(size_t)n * K + k];
            ref[(size_t)m * N + n] = (float)s + bias[n];
        }
    CUdeviceptr da = up16(a, (size_t)ROWS(M) * K), db = up16(b, (size_t)N * K);
    CUdeviceptr dbias = up(bias, N * 4), dc = dalloc((size_t)M * N * 4);
    float *want = malloc((size_t)M * N * 4);
    static const char *names[] = {"gemm_f16 f32 out", "gemm_f16 f16 out", "gemm_f16 gelu",
                                  "gemm_f16 relu", "gemm_f16 residual add", "gemm_f16 relu f32"};
    for (unsigned mode = 0; mode < 6; mode++) {
        CHECK(cuMemcpyHtoD(dc, c0, (size_t)M * N * 4));
        gemm_raw(da, db, dbias, dc, M, N, K, N, mode);
        int half = mode >= 1 && mode <= 3;
        float *got = half ? down16(dc, (size_t)M * N) : down(dc, (size_t)M * N);
        for (size_t i = 0; i < (size_t)M * N; i++) {
            float v = ref[i];
            if (mode == 2)
                v = gelu_ref(v);
            if (mode == 3 || mode == 5)
                v = v > 0 ? v : 0;
            if (mode == 4)
                v += c0[i];
            want[i] = half ? rf16(v) : v;
        }
        report(names[mode], cmp(got, want, (size_t)M * N), half ? 2e-3 : 1e-4);
        free(got);
    }
    /* rows past M are not written */
    float *c = down(dc, (size_t)M * N);
    free(c);
    free(a), free(b), free(bias), free(c0), free(ref), free(want);
    cuMemFree(da), cuMemFree(db), cuMemFree(dbias), cuMemFree(dc);
}

/* CPU attention with the same inputs as flash_attn (f16 Q, K, V) */
static void attn_ref(const float *q, const float *k, const float *v, float *o, unsigned lq,
                     unsigned lk, unsigned lqp, unsigned lkp, unsigned bh, unsigned dout,
                     float scale, int causal, const uint8_t *kmask, const float *bx,
                     const float *by, unsigned bgw, unsigned h, unsigned ostride, unsigned wsz,
                     unsigned nwx, unsigned gw)
{
    double *s = malloc(lk * sizeof(double));
    for (unsigned b = 0; b < bh; b++)
        for (unsigned i = 0; i < lq; i++) {
            double mx = -INFINITY;
            for (unsigned j = 0; j < lk; j++) {
                double d = 0;
                for (unsigned c = 0; c < 64; c++)
                    d += (double)q[((size_t)b * lqp + i) * 64 + c] *
                         k[((size_t)b * lkp + j) * 64 + c];
                d *= scale;
                if (bx)
                    d += by[((size_t)(b % h) * lq + i) * bgw + j / bgw] +
                         bx[((size_t)(b % h) * lq + i) * bgw + j % bgw];
                if ((causal && j > i) || (kmask && kmask[j]))
                    d = -INFINITY;
                s[j] = d;
                if (d > mx)
                    mx = d;
            }
            double l = 0;
            for (unsigned j = 0; j < lk; j++)
                l += s[j] = s[j] == -INFINITY ? 0 : exp(s[j] - mx);
            unsigned bb = b / h, hh = b % h, tok = i;
            if (wsz)
                tok = ((bb / nwx) * wsz + i / wsz) * gw + (bb % nwx) * wsz + i % wsz;
            for (unsigned c = 0; c < dout; c++) {
                double acc = 0;
                for (unsigned j = 0; j < lk; j++)
                    acc += s[j] * v[((size_t)b * lkp + j) * 64 + c];
                o[(size_t)tok * ostride + hh * dout + c] = (float)(acc / l);
            }
        }
    free(s);
}

static void test_flash_case(const char *name, unsigned lq, unsigned lk, unsigned bh, unsigned h,
                            unsigned dout, int flags, unsigned wsz, unsigned nwx, unsigned gw)
{
    unsigned lqp = PAD64(lq), lkp = PAD64(lk), bgw = 8;
    unsigned ostride = h * dout, tokens = wsz ? gw * gw : lq;
    float *q = rand_f32((size_t)bh * lqp * 64, -2, 2), *k = rand_f32((size_t)bh * lkp * 64, -2, 2);
    float *v = rand_f32((size_t)bh * lkp * 64, -1, 1);
    for (size_t i = 0; i < (size_t)bh * lqp * 64; i++)
        q[i] = rf16(q[i]);
    for (size_t i = 0; i < (size_t)bh * lkp * 64; i++)
        k[i] = rf16(k[i]), v[i] = rf16(v[i]);
    if (dout < 64) /* padded heads: the padding is zero */
        for (size_t r = 0; r < (size_t)bh * lkp; r++)
            for (unsigned c = dout; c < 64; c++)
                k[r * 64 + c] = v[r * 64 + c] = 0;
    uint8_t *km = malloc(lk);
    for (unsigned j = 0; j < lk; j++)
        km[j] = (flags & 1) && (rng_u32() % 3 == 0) && j > 0;
    float *bx = rand_f32((size_t)h * lq * bgw, -3, 3), *by = rand_f32((size_t)h * lq * bgw, -3, 3);
    float scale = 1.0f / sqrtf((float)dout);
    float *want = calloc((size_t)tokens * ostride, 4);
    attn_ref(q, k, v, want, lq, lk, lqp, lkp, bh, dout, scale, flags & 2, flags & 1 ? km : NULL,
             flags & 4 ? bx : NULL, by, bgw, h, ostride, wsz, nwx, gw);
    CUdeviceptr dq = up16(q, (size_t)bh * lqp * 64), dk = up16(k, (size_t)bh * lkp * 64);
    CUdeviceptr dv = up16(v, (size_t)bh * lkp * 64), dm = up(km, lk);
    CUdeviceptr dbx = up(bx, (size_t)h * lq * bgw * 4), dby = up(by, (size_t)h * lq * bgw * 4);
    CUdeviceptr dout_ = dalloc((size_t)tokens * ostride * 2);
    Att a = {lq, lk, lqp, lkp, bh, h, dout, ostride, (unsigned)flags, scale, dm, dbx, dby, bgw,
             wsz, nwx, gw};
    flash(dq, dk, dv, dout_, &a);
    float *got = down16(dout_, (size_t)tokens * ostride);
    report(name, cmp(got, want, (size_t)tokens * ostride), 4e-3);
    free(q), free(k), free(v), free(km), free(bx), free(by), free(want), free(got);
    cuMemFree(dq), cuMemFree(dk), cuMemFree(dv), cuMemFree(dm), cuMemFree(dbx), cuMemFree(dby);
    cuMemFree(dout_);
}

static void test_flash(void)
{
    test_flash_case("flash_attn plain (lq 100, lk 150, 3 heads)", 100, 150, 3, 3, 64, 0, 0, 1, 1);
    test_flash_case("flash_attn windows (4 windows of 16, 2 heads)", 16, 16, 8, 2, 64, 0, 4, 2, 8);
    test_flash_case("flash_attn key mask, 32-dim heads", 70, 33, 8, 8, 32, 1, 0, 1, 1);
    test_flash_case("flash_attn causal", 32, 32, 4, 4, 64, 2, 0, 1, 1);
    test_flash_case("flash_attn box bias (8 x 8 keys), 1 query", 1, 64, 8, 8, 32, 4, 0, 1, 1);
    test_flash_case("flash_attn box bias, 201 queries", 201, 64, 8, 8, 32, 4, 0, 1, 1);
}

static void test_layernorm(void)
{
    const unsigned R = 37, N = 256;
    float *x = rand_f32(R * N, -3, 5), *w = rand_f32(N, 0.5f, 1.5f), *b = rand_f32(N, -1, 1);
    float *pos = rand_f32(R * N, -1, 1), *want = malloc(R * N * 4), *wantp = malloc(R * N * 4);
    for (unsigned r = 0; r < R; r++) {
        double m = 0, v = 0;
        for (unsigned i = 0; i < N; i++)
            m += x[r * N + i];
        m /= N;
        for (unsigned i = 0; i < N; i++)
            v += (x[r * N + i] - m) * (x[r * N + i] - m);
        v /= N;
        for (unsigned i = 0; i < N; i++) {
            want[r * N + i] = (float)((x[r * N + i] - m) / sqrt(v + 1e-5) * w[i] + b[i]);
            wantp[r * N + i] = rf16(want[r * N + i] + pos[r * N + i]);
        }
    }
    CUdeviceptr dx = up(x, R * N * 4), dp = up(pos, R * N * 4), y32 = dalloc(R * N * 4);
    CUdeviceptr y16 = dalloc(R * N * 2), yp16 = dalloc(R * N * 2);
    Norm nm = {up(w, N * 4), up(b, N * 4)};
    layernorm(dx, &nm, y32, y16, dp, yp16, R, N);
    float *g32 = down(y32, R * N), *g16 = down16(y16, R * N), *gp = down16(yp16, R * N);
    report("layernorm f32", cmp(g32, want, R * N), 1e-5);
    for (unsigned i = 0; i < R * N; i++)
        want[i] = rf16(want[i]);
    report("layernorm f16, f16 + position", fmax(cmp(g16, want, R * N), cmp(gp, wantp, R * N)),
           2e-3);
    free(x), free(w), free(b), free(pos), free(want), free(wantp), free(g32), free(g16), free(gp);
}

static void test_rope(void)
{
    /* the real tables at the real sizes; compare a sample of rows */
    for (int g = 0; g < 2; g++) {
        float *qkv = rand_f32((size_t)NTOK * 3 * VIT_D, -2, 2);
        CUdeviceptr d = up(qkv, (size_t)NTOK * 3 * VIT_D * 4);
        unsigned t = NTOK, gw = GRID, wsz = g ? 0 : WIN, h = VIT_H;
        CUdeviceptr cs = g ? B.cs_glob : B.cs_win;
        void *a[] = {&d, &cs, &B.qh, &B.kh, &B.vh, &t, &gw, &wsz, &h};
        launch_1d(k_rope_vit, NTOK * VIT_H * 32, a);
        float *gq = down16(B.qh, (size_t)VIT_H * NTOK * 64);
        float *gk = down16(B.kh, (size_t)VIT_H * NTOK * 64);
        float *gv = down16(B.vh, (size_t)VIT_H * NTOK * 64);
        double err = 0;
        for (int s = 0; s < 4000; s++) {
            unsigned tok = rng_u32() % NTOK, hh = rng_u32() % VIT_H, j = rng_u32() % 32;
            unsigned y = tok / GRID, x = tok % GRID, b = 0, l = tok, L = NTOK;
            float px = g ? x * (1.0f / 3) : (float)(x % WIN), py = g ? y * (1.0f / 3) : (float)(y % WIN);
            if (!g)
                b = (y / WIN) * 3 + x / WIN, l = (y % WIN) * WIN + x % WIN, L = WIN * WIN;
            double ang = (j < 16 ? px : py) * pow(10000.0, -(double)(j % 16) / 16);
            double c = cos(ang), sn = sin(ang);
            const float *r = qkv + (size_t)tok * 3 * VIT_D + hh * 64 + 2 * j;
            size_t o = (((size_t)b * VIT_H + hh) * L + l) * 64 + 2 * j;
            float wq0 = rf16((float)(r[0] * c - r[1] * sn)), wq1 = rf16((float)(r[0] * sn + r[1] * c));
            float wk0 = rf16((float)(r[VIT_D] * c - r[VIT_D + 1] * sn));
            float wk1 = rf16((float)(r[VIT_D] * sn + r[VIT_D + 1] * c));
            float wv0 = rf16(r[2 * VIT_D]), wv1 = rf16(r[2 * VIT_D + 1]);
            float gg[6] = {gq[o], gq[o + 1], gk[o], gk[o + 1], gv[o], gv[o + 1]};
            float ww[6] = {wq0, wq1, wk0, wk1, wv0, wv1};
            double e = cmp(gg, ww, 6);
            if (e > err)
                err = e;
        }
        report(g ? "rope_vit global (positions / 3)" : "rope_vit 24 x 24 windows", err, 2e-3);
        free(qkv), free(gq), free(gk), free(gv);
        cuMemFree(d);
    }
}

static void test_conv_helpers(void)
{
    /* im2col3 + depth2space + upsample_add + split_heads: exact data movement */
    const unsigned H = 6, Wd = 5, C = 16;
    float *src = rand_f32(H * Wd * C, -2, 2);
    for (unsigned i = 0; i < H * Wd * C; i++)
        src[i] = rf16(src[i]);
    CUdeviceptr ds = up16(src, H * Wd * C), dd = dalloc(H * Wd * 9 * C * 2);
    im2col3(ds, dd, H, Wd, C);
    float *got = down16(dd, H * Wd * 9 * C), *want = malloc(H * Wd * 9 * C * 4);
    for (unsigned y = 0; y < H; y++)
        for (unsigned x = 0; x < Wd; x++)
            for (unsigned t = 0; t < 9; t++)
                for (unsigned c = 0; c < C; c++) {
                    int sy = (int)y + (int)(t / 3) - 1, sx = (int)x + (int)(t % 3) - 1;
                    want[((y * Wd + x) * 9 + t) * C + c] =
                        sy < 0 || sx < 0 || sy >= (int)H || sx >= (int)Wd
                            ? 0
                            : src[((unsigned)sy * Wd + (unsigned)sx) * C + c];
                }
    report("im2col3", cmp(got, want, H * Wd * 9 * C), 0);
    free(got), free(want);
    cuMemFree(dd);
    /* depth2space: src [H*W][4*O], O = C/4 */
    unsigned O = C / 4;
    dd = dalloc(H * Wd * 4 * O * 2);
    depth2space(ds, dd, H, Wd, O);
    got = down16(dd, H * Wd * 4 * O);
    want = malloc(H * Wd * 4 * O * 4);
    for (unsigned Y = 0; Y < 2 * H; Y++)
        for (unsigned X = 0; X < 2 * Wd; X++)
            for (unsigned o = 0; o < O; o++)
                want[(Y * 2 * Wd + X) * O + o] =
                    src[((Y / 2) * Wd + X / 2) * C + o * 4 + (Y % 2) * 2 + X % 2];
    report("depth2space", cmp(got, want, H * Wd * 4 * O), 0);
    free(got), free(want);
    cuMemFree(dd);
    /* upsample_add */
    float *prev = rand_f32(H * Wd * C, -2, 2), *cur = rand_f32(4 * H * Wd * C, -2, 2);
    CUdeviceptr dp = up(prev, H * Wd * C * 4), dc = up(cur, 4 * H * Wd * C * 4);
    dd = dalloc(4 * H * Wd * C * 2);
    unsigned h = H, w = Wd, c = C;
    void *a[] = {&dp, &dc, &dd, &h, &w, &c};
    launch_1d(k_upadd, H * Wd * C, a);
    got = down16(dd, 4 * H * Wd * C);
    want = malloc(4 * H * Wd * C * 4);
    for (unsigned Y = 0; Y < 2 * H; Y++)
        for (unsigned X = 0; X < 2 * Wd; X++)
            for (unsigned k = 0; k < C; k++)
                want[(Y * 2 * Wd + X) * C + k] =
                    rf16(cur[(Y * 2 * Wd + X) * C + k] + prev[((Y / 2) * Wd + X / 2) * C + k]);
    report("upsample_add", cmp(got, want, 4 * H * Wd * C), 0);
    free(got), free(want), free(prev), free(cur), free(src);
    cuMemFree(dd), cuMemFree(dp), cuMemFree(dc), cuMemFree(ds);
    /* split_heads with 32-dim heads padded to 64 */
    const unsigned L = 70, Hh = 8, hd = 32, ld = 600, col0 = 40;
    float *s = rand_f32(L * ld, -2, 2);
    ds = up(s, L * ld * 4);
    dd = dalloc(Hh * 128 * 64 * 2);
    split_heads(ds, ld, col0, dd, L, 128, Hh, hd);
    got = down16(dd, Hh * 128 * 64);
    double e = 0;
    for (unsigned hh = 0; hh < Hh; hh++)
        for (unsigned l = 0; l < L; l++)
            for (unsigned d = 0; d < 64; d++) {
                float wv = d < hd ? rf16(s[l * ld + col0 + hh * hd + d]) : 0;
                double x = fabs(got[(hh * 128 + l) * 64 + d] - wv);
                e = x > e ? x : e;
            }
    report("split_heads (32 of 64 dims)", e, 0);
    free(s), free(got);
    cuMemFree(ds), cuMemFree(dd);
}

static void test_small(void)
{
    /* patch im2col + add_pos at a small size: side 28, patch 14, C 3 */
    {
        const unsigned side = 28, p = 14, c = 3, gw = 2, kpad = 608;
        float *img = rand_f32(side * side * c, -1, 1);
        CUdeviceptr di = up(img, side * side * c * 4), dd = dalloc(gw * gw * kpad * 2);
        void *a[] = {&di, &dd, (void *)&side, (void *)&p, (void *)&c, (void *)&gw, (void *)&kpad};
        launch_1d(k_im2col_patch, gw * gw * kpad / 2, a);
        float *got = down16(dd, gw * gw * kpad), *want = calloc(gw * gw * kpad, 4);
        for (unsigned t = 0; t < gw * gw; t++)
            for (unsigned k = 0; k < p * p * c; k++) {
                unsigned ky = k / (p * c), kx = (k % (p * c)) / c, ch = k % c;
                want[t * kpad + k] =
                    rf16(img[(((t / gw) * p + ky) * side + (t % gw) * p + kx) * c + ch]);
            }
        report("im2col_patch", cmp(got, want, gw * gw * kpad), 0);
        free(img), free(got), free(want);
        cuMemFree(di), cuMemFree(dd);
    }
    {
        const unsigned g = 6, ps = 4, C = 8, T = g * g;
        float *x = rand_f32(T * C, -1, 1), *pos = rand_f32((ps * ps + 1) * C, -1, 1);
        CUdeviceptr dx = up(x, T * C * 4), dp = up(pos, (ps * ps + 1) * C * 4);
        void *a[] = {&dx, &dp, (void *)&T, (void *)&C, (void *)&g, (void *)&ps};
        launch_1d(k_add_pos, T * C / 4, a);
        float *got = down(dx, T * C);
        for (unsigned t = 0; t < T; t++)
            for (unsigned i = 0; i < C; i++)
                x[t * C + i] += pos[(1 + ((t / g) % ps) * ps + (t % g) % ps) * C + i];
        report("add_pos (tiled)", cmp(got, x, T * C), 0);
        free(x), free(pos), free(got);
        cuMemFree(dx), cuMemFree(dp);
    }
    /* linear_small */
    {
        const unsigned M = 7, N = 5, K = 300;
        float *x = rand_f32(M * K, -1, 1), *w = rand_f32(N * K, -1, 1), *b = rand_f32(N, -1, 1);
        CUdeviceptr dx = up(x, M * K * 4), dw = up(w, N * K * 4), db = up(b, N * 4);
        CUdeviceptr dy = dalloc(M * N * 4);
        linear_small(dx, K, dw, db, dy, N, M, N, K, 1);
        float *got = down(dy, M * N), want[7 * 5];
        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < N; n++) {
                double s = b[n];
                for (unsigned k = 0; k < K; k++)
                    s += (double)x[m * K + k] * w[n * K + k];
                want[m * N + n] = s > 0 ? (float)s : 0;
            }
        report("linear_small (relu)", cmp(got, want, M * N), 1e-5);
        free(x), free(w), free(b), free(got);
        cuMemFree(dx), cuMemFree(dw), cuMemFree(db), cuMemFree(dy);
    }
    /* group norm + relu */
    {
        const unsigned n = 999;
        float *x = rand_f32(n * 256, -2, 3), *w = rand_f32(256, 0.5f, 1.5f), *b = rand_f32(256, -1, 1);
        CUdeviceptr dx = up(x, n * 256 * 4), y16 = dalloc(n * 256 * 2), y32 = dalloc(n * 256 * 4);
        Norm nm = {up(w, 1024), up(b, 1024)};
        group_norm_relu(dx, n, &nm, y32, y16);
        float *got = down(y32, n * 256), *want = malloc(n * 256 * 4);
        for (unsigned g = 0; g < 8; g++) {
            double m = 0, v = 0;
            for (unsigned r = 0; r < n; r++)
                for (unsigned c = g * 32; c < g * 32 + 32; c++)
                    m += x[r * 256 + c];
            m /= n * 32.0;
            for (unsigned r = 0; r < n; r++)
                for (unsigned c = g * 32; c < g * 32 + 32; c++)
                    v += (x[r * 256 + c] - m) * (x[r * 256 + c] - m);
            v /= n * 32.0;
            for (unsigned r = 0; r < n; r++)
                for (unsigned c = g * 32; c < g * 32 + 32; c++) {
                    double y = (x[r * 256 + c] - m) / sqrt(v + 1e-5) * w[c] + b[c];
                    want[r * 256 + c] = y > 0 ? (float)y : 0;
                }
        }
        report("gn_stats + gn_apply (GroupNorm 8 + relu)", cmp(got, want, n * 256), 1e-5);
        free(x), free(w), free(b), free(got), free(want);
        cuMemFree(dx), cuMemFree(y16), cuMemFree(y32);
    }
    /* decoder box helpers: box_sine, rpb_mlp, box_update; masked_mean */
    {
        const unsigned Q = 5, G = 72;
        float ref[5 * 4];
        for (int i = 0; i < 20; i++)
            ref[i] = rng_uniform(0.05f, 0.6f);
        CUdeviceptr dr = up(ref, sizeof(ref)), ds = dalloc(Q * 512 * 2);
        unsigned q = Q;
        void *a[] = {&dr, &ds, &q};
        launch_1d(k_box_sine, Q * 512, a);
        float *got = down16(ds, Q * 512), *want = malloc(Q * 512 * 4);
        static const int coord[4] = {1, 0, 2, 3};
        for (unsigned i = 0; i < Q * 512; i++) {
            unsigned qq = i / 512, blk = (i % 512) / 128, k = i % 128;
            double v = ref[qq * 4 + coord[blk]] * 2 * M_PI / pow(10000.0, 2.0 * (k / 2) / 128);
            want[i] = rf16((float)(k % 2 ? cos(v) : sin(v)));
        }
        report("box_sine", cmp(got, want, Q * 512), 2e-3);
        free(got), free(want);
        float *w1 = rand_f32(512, -1, 1), *b1 = rand_f32(256, -1, 1), *w2 = rand_f32(2048, -1, 1);
        float *b2 = rand_f32(8, -1, 1);
        CUdeviceptr dw1 = up(w1, 2048), db1 = up(b1, 1024), dw2 = up(w2, 8192), db2 = up(b2, 32);
        CUdeviceptr dbx = dalloc(8 * (Q + 1) * G * 4), dby = dalloc(8 * (Q + 1) * G * 4);
        unsigned g = G;
        void *ab[] = {&dr, &dw1, &db1, &dw2, &db2, &dw1, &db1, &dw2, &db2, &dbx, &dby, &q, &g};
        CHECK(cuLaunchKernel(k_rpb, (Q * G * 2 + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, ab, NULL));
        float *gx = down(dbx, 8 * (Q + 1) * G), *gy = down(dby, 8 * (Q + 1) * G);
        float *wx = calloc(8 * (Q + 1) * G, 4), *wy = calloc(8 * (Q + 1) * G, 4);
        for (unsigned axis = 0; axis < 2; axis++)
            for (unsigned qq = 0; qq < Q; qq++)
                for (unsigned i = 0; i < G; i++) {
                    float c = ref[qq * 4 + axis], s = ref[qq * 4 + 2 + axis] * 0.5f;
                    float d[2] = {(float)i / G - (c - s), (float)i / G - (c + s)};
                    for (int e = 0; e < 2; e++) {
                        double t = d[e] * 8;
                        d[e] = (float)((t > 0) - (t < 0)) * (float)(log2(fabs(t) + 1) / 3);
                    }
                    double o[8];
                    for (int hh = 0; hh < 8; hh++)
                        o[hh] = b2[hh];
                    for (int k = 0; k < 256; k++) {
                        double hk = w1[k * 2] * d[0] + w1[k * 2 + 1] * d[1] + b1[k];
                        hk = hk > 0 ? hk : 0;
                        for (int hh = 0; hh < 8; hh++)
                            o[hh] += w2[hh * 256 + k] * hk;
                    }
                    for (int hh = 0; hh < 8; hh++)
                        (axis ? wy : wx)[(hh * (Q + 1) + qq + 1) * G + i] = (float)o[hh];
                }
        report("rpb_mlp (box bias MLPs)",
               fmax(cmp(gx, wx, 8 * (Q + 1) * G), cmp(gy, wy, 8 * (Q + 1) * G)), 1e-4);
        free(gx), free(gy), free(wx), free(wy), free(w1), free(b1), free(w2), free(b2);
        cuMemFree(dw1), cuMemFree(db1), cuMemFree(dw2), cuMemFree(db2), cuMemFree(dbx);
        cuMemFree(dby);
        float delta[20];
        for (int i = 0; i < 20; i++)
            delta[i] = rng_uniform(-2, 2);
        CUdeviceptr dd = up(delta, sizeof(delta));
        unsigned n = 20;
        void *au[] = {&dr, &dd, &n};
        CHECK(cuLaunchKernel(k_box_update, 1, 1, 1, 256, 1, 1, 0, NULL, au, NULL));
        float *gu = down(dr, 20), wu[20];
        for (int i = 0; i < 20; i++) {
            double x = ref[i], z = log(fmax(x, 1e-3) / fmax(1 - x, 1e-3)) + delta[i];
            wu[i] = (float)(1 / (1 + exp(-z)));
        }
        report("box_update", cmp(gu, wu, 20), 1e-5);
        free(gu);
        cuMemFree(dd), cuMemFree(dr), cuMemFree(ds);
    }
    {
        const unsigned L = 33, C = 256;
        float *x = rand_f32(L * C, -1, 1);
        uint8_t m[33];
        for (unsigned i = 0; i < L; i++)
            m[i] = i > 3 && i < 32;
        CUdeviceptr dx = up(x, L * C * 4), dm = up(m, L), dy = dalloc(C * 4);
        unsigned l = L, c = C;
        void *a[] = {&dx, &dm, &dy, &l, &c};
        CHECK(cuLaunchKernel(k_masked_mean, 1, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
        float *got = down(dy, C), want[256];
        for (unsigned j = 0; j < C; j++) {
            double s = 0;
            int k = 0;
            for (unsigned i = 0; i < L; i++)
                if (!m[i])
                    s += x[i * C + j], k++;
            want[j] = (float)(s / k);
        }
        report("masked_mean", cmp(got, want, C), 1e-6);
        free(x), free(got);
        cuMemFree(dx), cuMemFree(dm), cuMemFree(dy);
    }
    /* mask_resize: bilinear (align_corners False), threshold 0 */
    {
        const unsigned s = 16, h = 23, w = 37;
        float *lg = rand_f32(3 * s * s, -1, 1);
        uint32_t sel[2] = {2, 0};
        CUdeviceptr dl = up(lg, 3 * s * s * 4), ds = up(sel, 8), dout = dalloc(2 * h * w);
        unsigned n = 2, ss = s, hh = h, ww = w;
        void *a[] = {&dl, &ds, &dout, &n, &ss, &hh, &ww};
        launch_1d(k_mask_resize, 2 * h * w, a);
        uint8_t *got = malloc(2 * h * w);
        CHECK(cuMemcpyDtoH(got, dout, 2 * h * w));
        int bad = 0, near = 0;
        for (unsigned m = 0; m < 2; m++)
            for (unsigned y = 0; y < h; y++)
                for (unsigned x = 0; x < w; x++) {
                    double sy = fmax(0, (y + 0.5) * s / h - 0.5), sx = fmax(0, (x + 0.5) * s / w - 0.5);
                    unsigned y0 = (unsigned)sy, x0 = (unsigned)sx;
                    unsigned y1 = y0 + (y0 < s - 1), x1 = x0 + (x0 < s - 1);
                    double ly = sy - y0, lx = sx - x0;
                    const float *L = lg + sel[m] * s * s;
                    double v = (1 - ly) * ((1 - lx) * L[y0 * s + x0] + lx * L[y0 * s + x1]) +
                               ly * ((1 - lx) * L[y1 * s + x0] + lx * L[y1 * s + x1]);
                    if ((v > 0) != got[(m * h + y) * w + x])
                        *(fabs(v) < 1e-5 ? &near : &bad) += 1;
                }
        report("mask_resize (wrong pixels)", bad, 0);
        free(lg), free(got);
        cuMemFree(dl), cuMemFree(ds), cuMemFree(dout);
    }
}

static int run_tests(void)
{
    printf("kernel self-tests (sam3.ptx vs CPU)\n");
    test_gemm();
    test_flash();
    test_layernorm();
    test_rope();
    test_conv_helpers();
    test_small();
    printf("%d / %d passed\n", n_tests - n_fail, n_tests);
    return n_fail != 0;
}

/* ------------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s -i IMAGE -p PROMPT [-p PROMPT ...] [options]\n"
            "       %s test | tokenize TEXT | tokenize-ids\n"
            "options:\n"
            "  -m FILE      model GGUF (default %s)\n"
            "  -k FILE      kernels (default %s)\n"
            "  -v FILE      CLIP merges (default %s)\n"
            "  -t X         score threshold (default 0.5)\n"
            "  -o FILE      write the image with masks and boxes (PNG)\n"
            "  --masks DIR  write each mask as DIR/<prompt>_<n>.png\n"
            "  -q           only the detections on stdout\n",
            prog, prog, DEFAULT_MODEL, DEFAULT_CUBIN, DEFAULT_MERGES);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *model = DEFAULT_MODEL, *cubin = DEFAULT_CUBIN, *merges = DEFAULT_MERGES;
    const char *image_path = NULL, *out_path = NULL, *mask_dir = NULL;
    const char *prompts[64];
    int n_prompts = 0, quiet = 0;
    float thr = 0.5f;
    if (argc >= 2 && (!strcmp(argv[1], "tokenize") || !strcmp(argv[1], "tokenize-ids"))) {
        ClipTokenizer *t = clip_load(merges);
        int32_t ids[256];
        char buf[256];
        if (!strcmp(argv[1], "tokenize")) {
            if (argc < 3)
                usage(argv[0]);
            int n = clip_tokenize(t, argv[2], ids, 77);
            for (int i = 0; i < n; i++)
                printf("%6d  '%s'\n", ids[i], clip_token_text(t, ids[i], buf, sizeof(buf)));
        } else {
            char line[65536];
            while (fgets(line, sizeof(line), stdin)) {
                line[strcspn(line, "\n")] = 0;
                int n = clip_encode(t, line, ids, 256);
                for (int i = 0; i < n; i++)
                    printf(i ? " %d" : "%d", ids[i]);
                printf("\n");
            }
        }
        clip_free(t);
        return 0;
    }
    int test = argc >= 2 && !strcmp(argv[1], "test");
    for (int i = test ? 2 : 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc)
            model = argv[++i];
        else if (!strcmp(a, "-k") && i + 1 < argc)
            cubin = argv[++i];
        else if (!strcmp(a, "-v") && i + 1 < argc)
            merges = argv[++i];
        else if (!strcmp(a, "-i") && i + 1 < argc)
            image_path = argv[++i];
        else if (!strcmp(a, "-p") && i + 1 < argc && n_prompts < 64)
            prompts[n_prompts++] = argv[++i];
        else if (!strcmp(a, "-t") && i + 1 < argc)
            thr = (float)atof(argv[++i]);
        else if (!strcmp(a, "-o") && i + 1 < argc)
            out_path = argv[++i];
        else if (!strcmp(a, "--masks") && i + 1 < argc)
            mask_dir = argv[++i];
        else if (!strcmp(a, "-q"))
            quiet = 1;
        else
            usage(argv[0]);
    }
    if (!test && (!image_path || !n_prompts))
        usage(argv[0]);

    CUdevice dev;
    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
    load_kernels(cubin);
    if (test) {
        B.cs_win = rope_table(WIN, 1.0f);
        B.cs_glob = rope_table(GRID, (float)WIN / GRID);
        B.qh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
        B.kh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
        B.vh = dalloc((size_t)VIT_H * NTOK * 64 * 2);
        B.gn_acc = dalloc(16 * 8);
        return run_tests();
    }
    dump_dir = getenv("SAM3_DUMP");
    ClipTokenizer *tok = clip_load(merges);
    Image img;
    if (!image_load(image_path, &img))
        return 1;
    load_model(model);
    size_t wb = dev_bytes;
    alloc_buffers();
    if (!quiet)
        fprintf(stderr, "GPU memory: %.2f GB weights + %.2f GB activations\n", wb / 1e9,
                (dev_bytes - wb) / 1e9);

    double t0 = now_ms();
    float *pre = malloc((size_t)IMG * IMG * 3 * 4);
    const char *raw = getenv("SAM3_INPUT");
    if (raw) {
        FILE *f = fopen(raw, "rb");
        if (!f || fread(pre, 4, (size_t)IMG * IMG * 3, f) != (size_t)IMG * IMG * 3) {
            fprintf(stderr, "cannot read %s\n", raw);
            return 1;
        }
        fclose(f);
    } else {
        image_preprocess(&img, IMG, pre);
    }
    CHECK(cuMemcpyHtoD(B.img, pre, (size_t)IMG * IMG * 3 * 4));
    dump("input", B.img, (size_t)IMG * IMG * 3);
    free(pre);
    double t1 = now_ms();
    vision();
    CHECK(cuCtxSynchronize());
    double t2 = now_ms();
    if (!quiet)
        fprintf(stderr, "image %s %dx%d: preprocess %.0f ms, backbone + neck %.0f ms\n",
                image_path, img.w, img.h, t1 - t0, t2 - t1);

    Det det[NQ];
    for (int p = 0; p < n_prompts; p++) {
        uint8_t *m = NULL;
        double ts = now_ms();
        int n = detect(tok, prompts[p], thr, img.w, img.h, det, &m);
        double te = now_ms();
        if (!quiet)
            fprintf(stderr, "prompt \"%s\": %d found in %.0f ms\n", prompts[p], n, te - ts);
        for (int i = 0; i < n; i++) {
            size_t area = 0;
            for (size_t k = 0; k < (size_t)img.w * img.h; k++)
                area += m[(size_t)i * img.w * img.h + k];
            printf("%s\t%.4f\t%.1f %.1f %.1f %.1f\tquery %d\tmask %zu px\n", prompts[p],
                   det[i].score, det[i].box[0], det[i].box[1], det[i].box[2], det[i].box[3],
                   det[i].query, area);
            if (mask_dir) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s_%d.png", mask_dir, prompts[p], i);
                uint8_t *rgb = malloc((size_t)img.w * img.h * 3);
                for (size_t k = 0; k < (size_t)img.w * img.h; k++)
                    rgb[3 * k] = rgb[3 * k + 1] = rgb[3 * k + 2] =
                        m[(size_t)i * img.w * img.h + k] ? 255 : 0;
                png_write(path, rgb, img.w, img.h);
                free(rgb);
            }
            if (out_path)
                image_overlay(&img, m + (size_t)i * img.w * img.h, det[i].box, p * 7 + i);
        }
        free(m);
    }
    if (!quiet && n_prompts)
        fprintf(stderr,
                "per prompt (ms): text %.1f, encoder %.1f, decoder %.1f, scoring %.1f, "
                "masks %.1f, resize %.1f\n",
                t_stage[0] / n_prompts, t_stage[1] / n_prompts, t_stage[2] / n_prompts,
                t_stage[3] / n_prompts, t_stage[4] / n_prompts, t_stage[5] / n_prompts);
    if (out_path && png_write(out_path, img.rgb, img.w, img.h) && !quiet)
        fprintf(stderr, "wrote %s\n", out_path);
    image_free(&img);
    clip_free(tok);
    gguf_close(&gg);
    return 0;
}
