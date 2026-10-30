// Kernel Atlas - model pages: the GGUF dataflow graphs (from model_htmls/), redrawn with
// the engine's kernel names. One spec per model drives the overview graph, the layer graph,
// the launch list and the tensor table, so all four use the same names.
(function () {
'use strict';
const A = window.Atlas, $ = A.$;
const covered = new Set(A.KERNELS.map(k => k.id));
// a node's walkthrough is <kernel>.html unless it names another page (two engines' kernels share an .entry name)
const pageOf = n => n.page || n.kernel;
// the engine's PTX listing in src/, and whether a kernel is defined there
const SRC_FILE = { gptoss: 'gptoss', gemma3: 'gemma3', gemma_emb2: 'emb2', qwen35: 'qwen35', llama31: 'llama31', mistral03: 'mistral03', glm46v: 'glm46v', sam3: 'sam3', minimaxh3: 'minimaxh3' };
const srcOf = (M, kernel) => { const f = SRC_FILE[M.id]; return f && kernel && (A.SRC_FILES[f] || []).includes(kernel) ? A.srcHref(f, kernel) : null; };

// node kinds: io (data in or out), w (GGUF tensor), op (kernel launch or host step), blk (layer)
function nodeText(n, hot) {
  const sub = n.sub ? `<br/><small style='font-size:11.5px;opacity:.8'>${n.sub}</small>` : '';
  if (n.kind === 'io') return `${n.id}(["${n.label}${sub}"]):::io`;
  if (n.kind === 'w') return `${n.id}[/"<b>${n.label}</b>${sub}"/]:::w`;
  if (n.kind === 'blk') return `${n.id}["<b>${n.label}</b>${sub}"]:::blk`;
  const cls = !n.kernel ? 'host' : hot ? 'hot' : covered.has(pageOf(n)) ? 'op' : 'opx';
  const head = n.kernel ? `<span class='kn' style='font-family:IBM Plex Mono,monospace;font-weight:500;font-size:13.5px'>${n.kernel}</span>` : `<i>${n.label}</i>`;
  return `${n.id}("${head}${sub}"):::${cls}`;
}

function toMermaid(g, hotKernel) {
  const C = A.C, L = ['flowchart TB'];
  L.push(`classDef io fill:${C.panel},stroke:${C.x},color:${C.fg},stroke-width:1.5px`);
  L.push(`classDef w fill:${C.bg},stroke:${C.w},color:${C.fg}`);
  L.push(`classDef op fill:${C.panel},stroke:${C.sync},color:${C.fg},stroke-width:1.5px`);
  L.push(`classDef opx fill:${C.panel},stroke:${C.muted},color:${C.fg},stroke-dasharray:4 3`);
  L.push(`classDef host fill:${C.bg},stroke:${C.line},color:${C.muted}`);
  L.push(`classDef hot fill:${C.hl},stroke:${C.y},color:${C.fg},stroke-width:2.5px`);
  L.push(`classDef blk fill:${C.panel},stroke:${C.sync},color:${C.fg},stroke-width:1.5px`);
  const emit = n => L.push(nodeText(n, hotKernel && n.kernel === hotKernel));
  for (const item of g.nodes) {
    if (item.group) { L.push(`subgraph ${item.group}["${item.label} &nbsp; &nbsp; &nbsp;"]`); item.nodes.forEach(emit); L.push('end'); }
    else emit(item);
  }
  for (const [a, b, lab] of g.edges) L.push(lab ? `${a} -->|${lab}| ${b}` : `${a} --> ${b}`);
  return L.join('\n');
}
const flat = g => g.nodes.flatMap(n => n.group ? n.nodes : [n]);

let seq = 0;
async function draw(host, g, hotKernel, onPick) {
  const { svg } = await mermaid.render('mm' + (++seq), toMermaid(g, hotKernel));
  host.innerHTML = svg;
  const byId = Object.fromEntries(flat(g).map(n => [n.id, n]));
  host.querySelectorAll('g.node').forEach(el => {
    const m = /^flowchart-(.+)-\d+$/.exec(el.id), n = m && byId[m[1]];
    if (!n) return;
    el.style.cursor = 'pointer';
    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', (n.kernel || n.label) + (n.sub ? ', ' + n.sub.replace(/<[^>]+>/g, ' ') : ''));
    const pick = () => onPick(n);
    el.addEventListener('click', pick);
    el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  });
}

A.model = function (M) {
  const passes = M.passes || [['decode', 'decode · 1 token'], ['prefill', 'prefill · T tokens']];
  const kinds = M.kinds || [['local', 'local layer'], ['global', 'global layer']];
  const st = { pass: passes[0][0], kind: kinds[0][0], hot: null, sel: null };
  const hm = /k=([a-z0-9_]+)/.exec(location.hash);
  if (hm) { const k = A.KERNELS.find(x => x.id === hm[1]); st.hot = (k && k.entry) || hm[1]; }

  // controls
  const seg = (el, key, opts) => {
    el.innerHTML = opts.map(([v, t]) => `<button type="button" data-v="${v}">${t}</button>`).join('');
    const sync = () => el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.v === st[key]));
    el.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; st[key] = b.dataset.v; sync(); renderLayer(); renderLaunches(); });
    sync();
  };
  seg($('passSeg'), 'pass', passes);
  seg($('kindSeg'), 'kind', kinds);

  const usedKernels = () => [...new Set([...flat(M.overview()), ...passes.flatMap(([p]) => kinds.flatMap(([k]) => flat(M.layer(p, k))))].map(n => n.kernel).filter(Boolean))];
  const hotSel = $('hotSel');
  hotSel.innerHTML = '<option value="">none</option>' + usedKernels().sort().map(k => `<option>${k}</option>`).join('');
  hotSel.value = st.hot || '';
  hotSel.addEventListener('change', () => { st.hot = hotSel.value || null; history.replaceState(null, '', st.hot ? '#k=' + st.hot : location.pathname); renderAll(); });

  const detail = $('detail');
  function showDetail(n) {
    st.sel = n;
    const k = n.kernel && A.KERNELS.find(x => x.id === pageOf(n));
    let h = `<h3>${n.kernel ? `<code>${n.kernel}</code>` : n.label}</h3>`;
    if (n.sub) h += `<p class="muted">${n.sub}</p>`;
    const rows = [];
    if (n.kind === "w") { const t = M.tensors.find(t => t.name === (n.t || n.label)); if (t) rows.push(['file', t.file], ['engine', t.engine], ['read by', t.by]); }
    if (n.launch) rows.push(['launch', n.launch[st.pass] || n.launch.any]);
    if (n.io) rows.push(['buffers', n.io]);
    if (n.note) rows.push(['note', n.note]);
    const src = srcOf(M, n.kernel);
    if (src) rows.push(['PTX', `<a href="${src}">${n.kernel} in ${SRC_FILE[M.id]}.ptx</a>`]);
    const tri = M.id === 'gemma3' && n.kernel && A.TRITON && (A.TRITON[n.kernel] || A.TRITON_EXTRA[n.kernel]);
    if (tri) rows.push(['Triton', A.TRITON[n.kernel] ? A.TRITON[n.kernel].fns.filter(f => f[0] === '_').map(f => `<code>${f}</code>`).join(' + ') : `<code>${tri.label}</code>`]);
    if (rows.length) h += `<dl class="kv">${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('')}</dl>`;
    if (k) h += `<p><a href="${k.id}.html">Open the ${k.name} walkthrough →</a>${k.engine !== 'shared' && k.engine !== M.engine ? ` <span class="muted">(the page shows the ${k.engine} version)</span>` : ''}</p>`;
    else if (n.kernel) h += `<p class="muted">No walkthrough page for <code>${n.kernel}</code> yet.</p>`;
    if (n.kind === 'blk' && n.layerKind) h += `<p class="muted">Selected below in the layer graph.</p>`;
    detail.innerHTML = h;
  }

  async function renderOverview() {
    await draw($('ovGraph'), M.overview(), st.hot, n => {
      showDetail(n);
      if (n.layerKind) { st.kind = n.layerKind; $('kindSeg').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.v === st.kind)); renderLayer(); renderLaunches(); }
    });
  }
  async function renderLayer() {
    await draw($('lyGraph'), M.layer(st.pass, st.kind), st.hot, showDetail);
    $('lyTitle').textContent = M.layerTitle(st.pass, st.kind);
  }
  function renderLaunches() {
    const ops = flat(M.layer(st.pass, st.kind)).filter(n => n.kind === 'op');
    let i = 0;
    $('launches').innerHTML = `<thead><tr><th>#</th><th>kernel</th><th>launch</th><th>buffers</th><th>does</th><th>PTX</th></tr></thead><tbody>` +
      ops.map(n => {
        const name = n.kernel ? (covered.has(pageOf(n)) ? `<a href="${pageOf(n)}.html">${n.kernel}</a>` : n.kernel) : `<i>${n.label}</i>`;
        return `<tr class="${n.kernel && n.kernel === st.hot ? 'on' : ''}"><td>${n.kernel ? ++i : ''}</td><td>${name}</td><td>${n.launch ? (n.launch[st.pass] || n.launch.any) : 'host'}</td><td>${n.io || ''}</td><td>${(n.sub || '').replace(/<br\/>/g, ' · ')}</td><td>${srcOf(M, n.kernel) ? `<a href="${srcOf(M, n.kernel)}">source</a>` : ''}</td></tr>`;
      }).join('') + '</tbody>';
    $('launchCount').textContent = `${i} launches per layer × ${M.layersOf ? M.layersOf(st.kind) : M.nLayers} layers`;
  }
  function renderTensors() {
    $('tensors').innerHTML = `<thead><tr><th>GGUF tensor</th><th>file (rows × cols · type)</th><th>engine format on the GPU</th><th>read by</th></tr></thead><tbody>` +
      M.tensors.map(t => `<tr><td>${t.name}</td><td>${t.file}</td><td>${t.engine}</td><td>${t.by}</td></tr>`).join('') + '</tbody>';
  }
  async function renderAll() { try { await document.fonts.ready; } catch (e) {} await renderOverview(); await renderLayer(); renderLaunches(); if (st.sel) showDetail(st.sel); }

  function init() {
    const C = A.C;
    mermaid.initialize({
      startOnLoad: false, securityLevel: 'loose', theme: 'base',
      flowchart: { htmlLabels: true, curve: 'basis', nodeSpacing: 22, rankSpacing: 34, padding: 8 },
      themeVariables: { fontFamily: getComputedStyle(document.body).fontFamily, fontSize: '13px', background: C.panel, primaryColor: C.panel, primaryTextColor: C.fg, primaryBorderColor: C.line, lineColor: C.muted, clusterBkg: C.bg, clusterBorder: C.line, edgeLabelBackground: C.panel, tertiaryColor: C.panel },
    });
  }
  A.facts($('facts'), M.facts);
  renderTensors();
  init();
  renderAll().catch(e => { console.error(e); $('ovGraph').textContent = 'Graph failed to render: ' + e.message; });
  A.onTheme(() => { init(); renderAll(); });
  const k = A.MODELS.find(m => m.id === M.id);
  $('crumbs').innerHTML = `<a href="index.html" class="brand">kernel<i>_</i>atlas</a><span class="muted">/</span><span class="muted">Models</span>` +
    (SRC_FILE[M.id] ? `<a href="src/${SRC_FILE[M.id]}.html">${SRC_FILE[M.id]}.ptx source</a>` : '') + `<span style="margin-left:auto"></span>` +
    A.MODELS.filter(m => m !== k).map(m => `<a href="${m.id}.html">${m.name} →</a>`).join('');
};
})();
