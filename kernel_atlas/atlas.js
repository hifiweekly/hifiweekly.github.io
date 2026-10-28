// Kernel Atlas - shared helpers: theme colors, data generation, canvas grids,
// warp butterfly diagrams, code listings and the step-through driver.
(function () {
'use strict';
const A = {};
window.Atlas = A;

// Names follow one convention everywhere: a kernel is its PTX .entry name and its page is
// <name>.html; engine 'shared' means the PTX is identical in gptoss.ptx and gemma3.ptx.
// emb2.ptx (EmbeddingGemma 2) copies rmsnorm, add_rmsnorm, geglu and gemm_q8 from gemma3.ptx
// unchanged, so those kernels list 'gemma_emb2' among their models. qwen35.ptx (Qwen 3.5)
// copies rmsnorm, argmax and compact_logits unchanged ('qwen35') and adapts attention (a
// section on its page). llama31.ptx (Llama 3.1) copies rmsnorm, argmax, compact_logits,
// quant_q8 and swiglu of qwen35.ptx unchanged, and brings the generic-format mv_q / gemm_q
// (gemm_q's page has a section on Qwen 3.5's older version). mistral03.ptx (Mistral 7B
// v0.3) is llama31.ptx unchanged, so every Llama 3.1 kernel also lists 'mistral03'. glm46v.ptx
// (GLM-4.6V-Flash) keeps every llama31.ptx kernel but rope_kv, takes add_rmsnorm from gemma3.ptx
// and adds bias_rope_kv, so those kernels list 'glm46v'. Kernels that share an .entry
// name but not the code get pages of their own: qk_norm_rope_qwen, quant_q8_qwen,
// attention_oss, rope_kv_llama and attention_llama; `entry` gives the .entry name when the
// id differs from it.
A.MODELS = [
  { id: 'gptoss', name: 'gpt-oss-20b', engine: 'gpt-oss' },
  { id: 'gemma3', name: 'Gemma 3 12B', engine: 'Gemma 3' },
  { id: 'gemma_emb2', name: 'EmbeddingGemma 2', engine: 'EmbeddingGemma 2' },
  { id: 'qwen35', name: 'Qwen 3.5 9B', engine: 'Qwen 3.5' },
  { id: 'llama31', name: 'Llama 3.1 8B', engine: 'Llama 3.1' },
  { id: 'mistral03', name: 'Mistral 7B v0.3', engine: 'Mistral 7B' },
  { id: 'glm46v', name: 'GLM-4.6V-Flash 9.4B', engine: 'GLM-4.6V' },
];
A.KERNELS = [
  { id: 'rmsnorm', name: 'rmsnorm', engine: 'shared', models: ['gptoss', 'gemma3', 'gemma_emb2', 'qwen35', 'llama31', 'mistral03', 'glm46v'], family: 'Norms & activations' },
  { id: 'add_rmsnorm', name: 'add_rmsnorm', engine: 'Gemma 3', models: ['gemma3', 'gemma_emb2', 'glm46v'], family: 'Norms & activations' },
  { id: 'norm_add', name: 'norm_add', engine: 'EmbeddingGemma 2', models: ['gemma_emb2'], family: 'Norms & activations' },
  { id: 'qk_norm_rope', name: 'qk_norm_rope', engine: 'Gemma 3', models: ['gemma3'], family: 'Norms & activations' },
  { id: 'qk_norm_rope_qwen', name: 'qk_norm_rope (Qwen)', entry: 'qk_norm_rope', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Norms & activations' },
  { id: 'head_norm_rope', name: 'head_norm_rope', engine: 'EmbeddingGemma 2', models: ['gemma_emb2'], family: 'Norms & activations' },
  { id: 'swiglu_oai', name: 'swiglu_oai', engine: 'gpt-oss', models: ['gptoss'], family: 'Norms & activations' },
  { id: 'geglu', name: 'geglu', engine: 'Gemma 3', models: ['gemma3', 'gemma_emb2'], family: 'Norms & activations' },
  { id: 'gelu_mul', name: 'gelu_mul', engine: 'EmbeddingGemma 2', models: ['gemma_emb2'], family: 'Norms & activations' },
  { id: 'gated_norm', name: 'gated_norm', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Norms & activations' },
  { id: 'swiglu', name: 'swiglu', engine: 'Qwen 3.5', models: ['qwen35', 'llama31', 'mistral03', 'glm46v'], family: 'Norms & activations' },
  { id: 'quant_q8', name: 'quant_q8', engine: 'shared', models: ['gptoss', 'gemma3'], family: 'Matrix products' },
  { id: 'quant_q8_qwen', name: 'quant_q8 (Qwen)', entry: 'quant_q8', engine: 'Qwen 3.5', models: ['qwen35', 'llama31', 'mistral03', 'glm46v'], family: 'Matrix products' },
  { id: 'mv_q8', name: 'mv_q8', engine: 'shared', models: ['gptoss', 'gemma3'], family: 'Matrix products' },
  { id: 'mv_q5', name: 'mv_q5', engine: 'Gemma 3', models: ['gemma3'], family: 'Matrix products' },
  { id: 'mv_mxfp4', name: 'mv_mxfp4', engine: 'gpt-oss', models: ['gptoss'], family: 'Matrix products' },
  { id: 'mv_q4_1', name: 'mv_q4_1', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Matrix products' },
  { id: 'mv_q8_0', name: 'mv_q8_0', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Matrix products' },
  { id: 'mv_q5_k', name: 'mv_q5_k', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Matrix products' },
  { id: 'mv_q', name: 'mv_q', engine: 'Llama 3.1', models: ['llama31', 'mistral03', 'glm46v'], family: 'Matrix products' },
  { id: 'gemm_q', name: 'gemm_q', engine: 'Llama 3.1', models: ['llama31', 'mistral03', 'glm46v', 'qwen35'], family: 'Matrix products' },
  { id: 'gemm_q8', name: 'gemm_q8', engine: 'shared', models: ['gptoss', 'gemma3', 'gemma_emb2'], family: 'Matrix products' },
  { id: 'rope_kv', name: 'rope_kv', engine: 'gpt-oss', models: ['gptoss'], family: 'Attention' },
  { id: 'attention', name: 'attention', engine: 'Gemma 3', models: ['gemma3', 'qwen35'], family: 'Attention' },
  { id: 'attention_oss', name: 'attention (gpt-oss)', entry: 'attention', engine: 'gpt-oss', models: ['gptoss'], family: 'Attention' },
  { id: 'rope_kv_llama', name: 'rope_kv (Llama)', entry: 'rope_kv', engine: 'Llama 3.1', models: ['llama31', 'mistral03'], family: 'Attention' },
  { id: 'attention_llama', name: 'attention (Llama)', entry: 'attention', engine: 'Llama 3.1', models: ['llama31', 'mistral03', 'glm46v'], family: 'Attention' },
  { id: 'bias_rope_kv', name: 'bias_rope_kv', engine: 'GLM-4.6V', models: ['glm46v'], family: 'Attention' },
  { id: 'attention_enc', name: 'attention_enc', engine: 'EmbeddingGemma 2', models: ['gemma_emb2'], family: 'Attention' },
  { id: 'mv_f32', name: 'mv_f32', engine: 'gpt-oss', models: ['gptoss'], family: 'Routing & sampling' },
  { id: 'router_topk', name: 'router_topk', engine: 'gpt-oss', models: ['gptoss'], family: 'Routing & sampling' },
  { id: 'moe_sort', name: 'moe_sort', engine: 'gpt-oss', models: ['gptoss'], family: 'Routing & sampling' },
  { id: 'moe_combine', name: 'moe_combine', engine: 'gpt-oss', models: ['gptoss'], family: 'Routing & sampling' },
  { id: 'gdn_conv', name: 'gdn_conv', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Linear attention' },
  { id: 'gdn_recur', name: 'gdn_recur', engine: 'Qwen 3.5', models: ['qwen35'], family: 'Linear attention' },
  { id: 'argmax', name: 'argmax', engine: 'shared', models: ['gptoss', 'gemma3', 'qwen35', 'llama31', 'mistral03', 'glm46v'], family: 'Routing & sampling' },
  { id: 'compact_logits', name: 'compact_logits', engine: 'shared', models: ['gptoss', 'gemma3', 'qwen35', 'llama31', 'mistral03', 'glm46v'], family: 'Routing & sampling' },
];
A.engineLabel = e => e === 'shared' ? 'gpt-oss + Gemma 3' : e;

// <src-xref> generated by src/build.py: the PTX listings in src/ that hold each page's kernel
// (primary engine first), and each listing's functions.
A.SRC = {
  add_rmsnorm: [['gemma3', 'add_rmsnorm'], ['emb2', 'add_rmsnorm'], ['glm46v', 'add_rmsnorm']],
  argmax: [['gptoss', 'argmax'], ['gemma3', 'argmax'], ['qwen35', 'argmax'], ['qwen38', 'argmax'], ['llama31', 'argmax'], ['mistral03', 'argmax'], ['glm46v', 'argmax']],
  attention: [['gemma3', 'attention'], ['qwen35', 'attention'], ['qwen38', 'attention']],
  attention_enc: [['emb2', 'attention_enc']],
  attention_llama: [['llama31', 'attention'], ['mistral03', 'attention'], ['glm46v', 'attention']],
  attention_oss: [['gptoss', 'attention']],
  bias_rope_kv: [['glm46v', 'bias_rope_kv']],
  compact_logits: [['gptoss', 'compact_logits'], ['gemma3', 'compact_logits'], ['qwen35', 'compact_logits'], ['qwen38', 'compact_logits'], ['llama31', 'compact_logits'], ['mistral03', 'compact_logits'], ['glm46v', 'compact_logits']],
  gated_norm: [['qwen35', 'gated_norm'], ['qwen38', 'gated_norm']],
  gdn_conv: [['qwen35', 'gdn_conv'], ['qwen35', 'conv_save'], ['qwen38', 'gdn_conv'], ['qwen38', 'conv_save']],
  gdn_recur: [['qwen35', 'gdn_recur'], ['qwen38', 'gdn_recur']],
  geglu: [['gemma3', 'geglu'], ['emb2', 'geglu']],
  gelu_mul: [['emb2', 'gelu_mul']],
  gemm_q: [['llama31', 'gemm_q'], ['qwen35', 'gemm_q'], ['qwen38', 'gemm_q'], ['mistral03', 'gemm_q'], ['glm46v', 'gemm_q']],
  gemm_q8: [['gptoss', 'gemm_q8'], ['gemma3', 'gemm_q8'], ['emb2', 'gemm_q8']],
  head_norm_rope: [['emb2', 'head_norm_rope']],
  moe_combine: [['gptoss', 'moe_combine']],
  moe_sort: [['gptoss', 'moe_sort']],
  mv_f32: [['gptoss', 'mv_f32']],
  mv_mxfp4: [['gptoss', 'mv_mxfp4']],
  mv_q: [['llama31', 'mv_q'], ['qwen38', 'mv_q'], ['mistral03', 'mv_q'], ['glm46v', 'mv_q']],
  mv_q4_1: [['qwen35', 'mv_q4_1']],
  mv_q5: [['gemma3', 'mv_q5']],
  mv_q5_k: [['qwen35', 'mv_q5_k']],
  mv_q8: [['gptoss', 'mv_q8'], ['gemma3', 'mv_q8']],
  mv_q8_0: [['qwen35', 'mv_q8_0']],
  norm_add: [['emb2', 'norm_add']],
  qk_norm_rope: [['gemma3', 'qk_norm_rope']],
  qk_norm_rope_qwen: [['qwen35', 'qk_norm_rope'], ['qwen38', 'qk_norm_rope']],
  quant_q8: [['gptoss', 'quant_q8'], ['gemma3', 'quant_q8']],
  quant_q8_qwen: [['qwen35', 'quant_q8'], ['qwen38', 'quant_q8'], ['llama31', 'quant_q8'], ['mistral03', 'quant_q8'], ['glm46v', 'quant_q8']],
  rmsnorm: [['gptoss', 'rmsnorm'], ['gemma3', 'rmsnorm'], ['emb2', 'rmsnorm'], ['qwen35', 'rmsnorm'], ['qwen38', 'rmsnorm'], ['llama31', 'rmsnorm'], ['mistral03', 'rmsnorm'], ['glm46v', 'rmsnorm']],
  rope_kv: [['gptoss', 'rope_kv']],
  rope_kv_llama: [['llama31', 'rope_kv'], ['mistral03', 'rope_kv']],
  router_topk: [['gptoss', 'router_topk']],
  swiglu: [['qwen35', 'swiglu'], ['qwen38', 'swiglu'], ['llama31', 'swiglu'], ['mistral03', 'swiglu'], ['glm46v', 'swiglu']],
  swiglu_oai: [['gptoss', 'swiglu_oai']],
};
A.SRC_FILES = {
  gptoss: ['fp4x4', 'rmsnorm', 'quant_q8', 'dot4_s8', 'mv_q8', 'mv_mxfp4', 'mv_f32', 'router_topk', 'rope_kv', 'attention', 'swiglu_oai', 'moe_combine', 'st4_s8', 'st4_fp4', 'gemm_q8', 'moe_sort', 'gemm_mxfp4', 'argmax', 'compact_logits', 'sample_logits'],
  gemma3: ['rmsnorm', 'add_rmsnorm', 'mv_q5', 'qk_norm_rope', 'quant_q8', 'dot4_s8', 'mv_q8', 'attention', 'geglu', 'st4_s8', 'st4_q5', 'gemm_q8', 'gemm_q5', 'argmax', 'compact_logits', 'sample_logits'],
  emb2: ['rmsnorm', 'add_rmsnorm', 'geglu', 'st4_s8', 'gemm_q8', 'head_norm_rope', 'attention_enc', 'gelu_mul', 'norm_add', 'scale'],
  qwen35: ['rmsnorm', 'quant_q8', 'mv_q4_1', 'mv_q8_0', 'mv_q5_k', 'mv_q6_k', 'gm_out', 'gemm_q', 'gdn_conv', 'conv_save', 'gdn_recur', 'gated_norm', 'swiglu', 'qk_norm_rope', 'attention', 'argmax', 'compact_logits', 'sample_logits'],
  qwen38: ['rmsnorm', 'quant_q8', 'iq4_lut', 'iq3s_word', 'unpack16', 'mv_q', 'gm_out', 'gemm_q', 'gdn_conv', 'conv_save', 'gdn_recur', 'gated_norm', 'swiglu', 'qk_norm_rope', 'attention', 'argmax', 'compact_logits', 'sample_logits'],
  llama31: ['rmsnorm', 'quant_q8', 'iq4_lut', 'iq3s_word', 'unpack16', 'mv_q', 'gm_out', 'gemm_q', 'swiglu', 'rope_kv', 'attention', 'argmax', 'compact_logits', 'sample_logits'],
  mistral03: ['rmsnorm', 'quant_q8', 'iq4_lut', 'iq3s_word', 'unpack16', 'mv_q', 'gm_out', 'gemm_q', 'swiglu', 'rope_kv', 'attention', 'argmax', 'compact_logits', 'sample_logits'],
  glm46v: ['rmsnorm', 'add_rmsnorm', 'quant_q8', 'iq4_lut', 'iq3s_word', 'unpack16', 'mv_q', 'gm_out', 'gemm_q', 'swiglu', 'bias_rope_kv', 'attention', 'argmax', 'compact_logits', 'sample_logits'],
};
// </src-xref>
// Links to the full listings: one per file holding the page's kernel (a page that explains two
// kernels of one file, like gdn_conv and conv_save, links the first).
A.srcLinks = function (id) {
  const seen = new Set();
  return (A.SRC[id] || []).filter(([f]) => !seen.has(f) && seen.add(f));
};
A.srcHref = (file, fn) => `src/${file}.html#${fn}`;

// ---------- numbers ----------
A.fr = Math.fround;
A.h32 = function (a) { a |= 0; a = Math.imul(a ^ (a >>> 16), 0x7feb352d); a = Math.imul(a ^ (a >>> 15), 0x846ca68b); return (a ^ (a >>> 16)) >>> 0; };
A.rng = function (seed) { // mulberry32
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
};
A.gauss = function (r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
A.f16 = function (x) {
  if (!x || !isFinite(x)) return x;
  const a = Math.abs(x); let e = Math.floor(Math.log2(a)); if (e < -14) e = -14;
  const s = 2 ** (e - 10); return Math.sign(x) * Math.round(a / s) * s;
};
A.fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : (v === -Infinity ? '−∞' : (Math.abs(v) >= 1e5 ? v.toExponential(3) : v.toFixed(d)));
A.int = n => n.toLocaleString('en-US');
A.$ = id => document.getElementById(id);
A.esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ---------- theme colors ----------
A.C = {};
const themeHooks = [];
A.readColors = function () {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ['bg', 'panel', 'fg', 'muted', 'line', 'zero', 'w', 'x', 'y', 'sync', 'hl']) A.C[k] = cs.getPropertyValue('--' + k).trim();
};
A.onTheme = fn => themeHooks.push(fn);
A.readColors();
const recolor = () => { A.readColors(); themeHooks.forEach(f => f()); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', recolor);
new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// ---------- page chrome ----------
A.chrome = function (id) {
  A.page = id;
  const i = A.KERNELS.findIndex(k => k.id === id), k = A.KERNELS[i];
  const prev = A.KERNELS[(i + A.KERNELS.length - 1) % A.KERNELS.length], next = A.KERNELS[(i + 1) % A.KERNELS.length];
  const el = A.$('crumbs');
  const runs = k.models.map(m => A.MODELS.find(x => x.id === m)).map(m => `<a href="${m.id}.html#k=${k.id}">${m.name}</a>`).join(' ');
  if (el) el.innerHTML = `<a href="index.html" class="brand">kernel<i>_</i>atlas</a><span class="muted">/</span><span class="muted">${k.family} · ${A.engineLabel(k.engine)}</span>` +
    `<span class="muted">runs in</span>${runs}` +
    (A.srcLinks(id).length ? `<span class="muted">PTX source</span>` + A.srcLinks(id).map(([f, fn]) => `<a href="${A.srcHref(f, fn)}">${f}.ptx</a>`).join(' ') : '') +
    (A.TRITON && A.TRITON[id] ? `<a href="#s-triton">Triton version</a>` : '') +
    `<span style="margin-left:auto"></span><a href="${prev.id}.html">← ${prev.name}</a><a href="${next.id}.html">${next.name} →</a>`;
  const f = A.$('foot');
  if (f) f.innerHTML = `<span>${f.innerHTML}</span><span><a href="${prev.id}.html">← ${prev.name}</a> · <a href="index.html">all kernels</a> · <a href="${next.id}.html">${next.name} →</a></span>`;
  if (A.tritonSection) A.tritonSection(id);
};
A.facts = (el, list) => { el.innerHTML = list.map(([b, s]) => `<div class="fact"><b>${b}</b><span>${s}</span></div>`).join(''); };
A.legend = (el, list) => { el.innerHTML = list.map(([c, t]) => `<span><i class="sw" style="background: var(${c})"></i>${t}</span>`).join(''); };

// ---------- canvas grid ----------
A.grid = function (el, rows, cols) {
  const g = { el, rows, cols };
  el.style.aspectRatio = `${cols} / ${rows}`;
  g.resize = (r, c) => { g.rows = r; g.cols = c; el.style.aspectRatio = `${c} / ${r}`; };
  g.prep = function () {
    const w = el.clientWidth, h = el.clientHeight, dpr = window.devicePixelRatio || 1;
    if (el.width !== Math.round(w * dpr) || el.height !== Math.round(h * dpr)) { el.width = Math.round(w * dpr); el.height = Math.round(h * dpr); }
    const ctx = el.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = A.C.bg; ctx.fillRect(0, 0, w, h);
    g.ctx = ctx; g.w = w; g.h = h; g.cw = w / g.cols; g.ch = h / g.rows;
    return g;
  };
  g.cell = function (r, c, color, alpha = 1, pad = 0.12) {
    const p = Math.min(g.cw, g.ch) * pad, ctx = g.ctx;
    ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fillRect(c * g.cw + p / 2, r * g.ch + p / 2, g.cw - p, g.ch - p); ctx.globalAlpha = 1;
  };
  g.box = function (r, c, nr, nc, color, lw = 2) {
    const ctx = g.ctx; ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.strokeRect(c * g.cw + lw / 2 - 0.5, r * g.ch + lw / 2 - 0.5, nc * g.cw - lw + 1, nr * g.ch - lw + 1);
  };
  g.hatch = function (r, c, nr = 1, nc = 1) {
    const ctx = g.ctx, x = c * g.cw, y = r * g.ch, w = nc * g.cw, h = nr * g.ch;
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.strokeStyle = A.C.zero; ctx.lineWidth = 1;
    for (let o = -h; o < w; o += 4) { ctx.beginPath(); ctx.moveTo(x + o, y + h); ctx.lineTo(x + o + h, y); ctx.stroke(); }
    ctx.restore();
  };
  g.text = function (r, c, s, color, size) {
    const ctx = g.ctx; ctx.fillStyle = color || A.C.fg; ctx.font = `${size || Math.min(11, g.ch * 0.55)}px "IBM Plex Mono", monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(s, c * g.cw + g.cw / 2, r * g.ch + g.ch / 2);
  };
  g.hline = function (r, color, lw = 1) { g.ctx.fillStyle = color; g.ctx.fillRect(0, r * g.ch - lw / 2, g.w, lw); };
  g.vline = function (c, color, lw = 1) { g.ctx.fillStyle = color; g.ctx.fillRect(c * g.cw - lw / 2, 0, lw, g.h); };
  g.pos = function (e) {
    const b = el.getBoundingClientRect();
    return { r: Math.max(0, Math.min(g.rows - 1, Math.floor((e.clientY - b.top) / b.height * g.rows))), c: Math.max(0, Math.min(g.cols - 1, Math.floor((e.clientX - b.left) / b.width * g.cols))) };
  };
  g.hook = function (onHover, onClick, onLeave) {
    el.classList.add('pick');
    el.addEventListener('mousemove', e => onHover && onHover(g.pos(e)));
    el.addEventListener('mouseleave', () => onLeave && onLeave());
    el.addEventListener('click', e => onClick && onClick(g.pos(e)));
  };
  return g;
};

// ---------- warp butterfly (shfl.sync.bfly with offsets 16, 8, 4, 2, 1) ----------
// table[s][lane] = value after s exchange rounds; combine(a, b) gives the new value.
A.bflyTable = function (vals, combine) {
  const t = [vals.slice()];
  for (let s = 0, off = 16; off >= 1; s++, off >>= 1) {
    const prev = t[s], cur = [];
    for (let l = 0; l < 32; l++) cur.push(combine(prev[l], prev[l ^ off], l, l ^ off));
    t.push(cur);
  }
  return t;
};
// Draws rows 0..5 of the table; rounds <= stage are shown, the exchange of round `stage` highlighted.
A.butterfly = function (svg, table, stage, opts = {}) {
  const C = A.C, cw = 30, rh = 34, ox = 74, oy = 18, W = ox + 32 * cw + 6, H = oy + 6 * rh + 4;
  const fmt = opts.fmt || (v => A.fmt(v, 1)), sel = opts.sel ?? -1;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let s = '';
  for (let l = 0; l < 32; l += 1) if (l % 4 === 0) s += `<text x="${ox + l * cw + cw / 2}" y="11" font-size="9.5" text-anchor="middle" fill="${C.muted}">${l}</text>`;
  s += `<text x="4" y="11" font-size="9.5" fill="${C.muted}">lane</text>`;
  for (let r = 0; r <= 5; r++) {
    const y = oy + r * rh, shown = r <= stage, off = r ? 32 >> r : 0;
    s += `<text x="4" y="${y + 15}" font-size="10" fill="${shown ? C.fg : C.muted}">${r ? 'xor ' + off : 'start'}</text>`;
    if (r && r <= stage) {
      for (let l = 0; l < 32; l++) {
        const m = l ^ off; if (m < l) continue;
        const x1 = ox + l * cw + cw / 2, x2 = ox + m * cw + cw / 2, hot = r === stage;
        s += `<path d="M${x1} ${y - rh + 22} L${x2} ${y + 1} M${x2} ${y - rh + 22} L${x1} ${y + 1}" stroke="${hot ? C.sync : C.line}" stroke-width="${hot ? 1.1 : 0.8}" fill="none" opacity="${hot ? 0.9 : 0.7}"/>`;
      }
    }
    for (let l = 0; l < 32; l++) {
      const x = ox + l * cw;
      s += `<rect x="${x + 1.5}" y="${y + 2}" width="${cw - 3}" height="20" rx="2" fill="${shown ? (opts.color || C.y) : C.zero}" opacity="${shown ? (l === sel ? 0.95 : 0.28) : 0.25}" stroke="${l === sel ? C.fg : 'none'}" stroke-width="1.5"/>`;
      if (shown) s += `<text x="${x + cw / 2}" y="${y + 15.5}" font-size="8.5" text-anchor="middle" fill="${C.fg}">${A.esc(fmt(table[r][l]))}</text>`;
    }
  }
  svg.innerHTML = s;
};

// ---------- code listings ----------
// Source lines are "TAGS|text"; TAGS is a comma list matched against the step's tag.
const KW = {
  cpp: /\b(extern|__global__|__launch_bounds__|__shared__|__device__|const|unsigned|float|float2|float4|uint4|uint2|int|int8_t|uint32_t|__half|void|for|if|else|return|while|__syncthreads|__ldg|__half2float|__float2half|__shfl_xor_sync|atomicAdd|fmaf|fminf|fmaxf|exp2f|sqrtf|rsqrtf|copysignf|break|bool)\b/g,
  py: /@triton\.jit|\b(def|for|in|range|if|else|return|and|or|not|None|True|False)\b|\b(tl|triton|libdevice)\.[\w.]+/g,
  ptx: /\b(ld\.[\w.]+|st\.[\w.]+|fma\.rn\.f32|mul\.[\w.]+|bar\.sync|bra(\.uni)?|call(\.uni)?|setp\.[\w.]+|selp\.[\w.]+|add\.[\w.]+|sub\.[\w.]+|and\.[\w.]+|or\.pred|shr\.u32|shl\.b32|mov\.[\w.]+|cvt\.[\w.]+|shfl\.sync\.bfly\.b32|atom\.[\w.]+|ex2\.approx\.f32|div\.[\w.]+|rem\.u32|sqrt\.rn\.f32|rcp\.rn\.f32|max\.f32|min\.[\w.]+|abs\.f32|neg\.f32|copysign\.f32|mad\.lo\.u32|bfe\.s32|div\.u32|ret)\b/g,
};
function hl(src, lang) {
  let code = A.esc(src), comment = '';
  const ci = code.indexOf(lang === 'py' ? '#' : '//'); if (ci >= 0) { comment = code.slice(ci); code = code.slice(0, ci); }
  code = code.replace(KW[lang], '<span class="k">$&</span>');
  return code + (comment ? `<span class="c">${comment}</span>` : '');
}
// Pages whose kernel has a Triton port (triton.js) get it as a third listing. The chosen tab is
// remembered per viewer, so someone reading the Triton code keeps seeing it from page to page.
A.tritonListing = id => A.TRITON && A.TRITON[id] && { label: 'Triton (Python)', lang: 'py', src: A.TRITON[id].src };
A.codeTab = { get() { try { return localStorage.getItem('atlas-code'); } catch (e) { return null; } }, set(v) { try { localStorage.setItem('atlas-code', v); } catch (e) {} } };
A.hl = hl;
A.code = function (pre, tabsEl, listings, opts = {}) {
  const py = A.tritonListing(A.page);
  if (py && !listings.py) listings = { ...listings, py };
  const keep = opts.remember !== false, names = Object.keys(listings), saved = keep && A.codeTab.get(); let cur = names.includes(saved) ? saved : names[0], tag = '';
  const src = A.srcLinks(A.page)[0];
  tabsEl.innerHTML = names.map(n => `<button data-n="${n}" aria-pressed="${n === cur}">${listings[n].label}</button>`).join('') +
    (src && opts.srcLink !== false ? `<a class="srclink" href="${A.srcHref(src[0], src[1])}">whole kernel in ${src[0]}.ptx →</a>` : '');
  tabsEl.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; cur = b.dataset.n; if (keep) A.codeTab.set(cur); tabsEl.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(); });
  function draw() {
    const L = listings[cur];
    pre.innerHTML = L.src.split('\n').map(l => {
      const i = l.indexOf('|'), tags = l.slice(0, i).split(',').map(s => s.trim()), body = l.slice(i + 1);
      return `<span class="ln${tag && tags.includes(tag) ? ' on' : ''}">${hl(body, L.lang) || ' '}</span>`;
    }).join('');
  }
  return { show(t) { tag = t; draw(); } };
};

// ---------- step-through driver ----------
// steps: [{tag, title, text, delay?}]; render(step, index) draws everything else.
A.stepper = function (opts) {
  const host = A.$(opts.host || 'stepper');
  host.innerHTML = `<div class="btns"><button data-a="prev" aria-label="Previous step">◀</button><button data-a="play" aria-pressed="false">Play</button><button data-a="next" aria-label="Next step">▶</button></div>
    <label class="kctl">step <input type="range" min="0" value="0" step="1" aria-label="Step"> <span class="lbl"></span></label>`;
  const slider = host.querySelector('input'), lbl = host.querySelector('.lbl'), play = host.querySelector('[data-a=play]');
  let steps = opts.steps, i = 0, timer = null, playing = false;
  const cap = A.$(opts.caption || 'caption');
  function stop() { playing = false; clearTimeout(timer); play.textContent = 'Play'; play.setAttribute('aria-pressed', 'false'); }
  function go(n) {
    i = Math.max(0, Math.min(steps.length - 1, n));
    const st = steps[i];
    slider.max = steps.length - 1; slider.value = i;
    lbl.textContent = `${i + 1}/${steps.length} · ${st.short || st.title}`;
    if (cap) cap.innerHTML = `<h3>${st.title}</h3><p>${st.text}</p>`;
    if (opts.code) opts.code.show(st.tag);
    opts.render(st, i);
  }
  function tick() { if (!playing) return; if (i >= steps.length - 1) { stop(); return; } go(i + 1); timer = setTimeout(tick, steps[i].delay || opts.delay || 900); }
  host.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.a === 'prev') { stop(); go(i - 1); }
    if (b.dataset.a === 'next') { stop(); go(i + 1); }
    if (b.dataset.a === 'play') {
      if (playing) { stop(); return; }
      if (i >= steps.length - 1) go(0);
      playing = true; play.textContent = 'Pause'; play.setAttribute('aria-pressed', 'true'); timer = setTimeout(tick, 400);
    }
  });
  slider.addEventListener('input', () => { stop(); go(+slider.value); });
  document.addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === 'ArrowRight') { stop(); go(i + 1); }
    if (e.key === 'ArrowLeft') { stop(); go(i - 1); }
  });
  return {
    go, refresh: () => go(i), get index() { return i; }, get step() { return steps[i]; },
    setSteps(s, keep) { steps = s; stop(); go(keep ? Math.min(i, s.length - 1) : 0); },
  };
};

// redraw on resize / theme / font load
A.onResize = function (fn) {
  let raf = 0; const run = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(fn); };
  new ResizeObserver(run).observe(document.body);
  A.onTheme(run);
  if (document.fonts) document.fonts.ready.then(run);
};
})();
