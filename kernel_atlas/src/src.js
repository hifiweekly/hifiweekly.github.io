// Kernel Atlas - PTX source listings: register highlighting, symbol search, the
// active entry of the side index, and opening collapsed blocks that a link points into.
(function () {
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ---------- registers: hover marks every use in the function, click keeps them ----------
// A register's scope is its function card (a .func's parameters are registers there too).
const scopeOf = el => el.closest('.card') || document.body;
const usesOf = (el, r) => [...scopeOf(el).querySelectorAll('.r')].filter(x => x.dataset.r === r);
let hov = [], sel = null; // sel: {r, card, list, i}
function unhover() { hov.forEach(x => x.classList.remove('hov')); hov = []; }
document.addEventListener('mouseover', e => {
  const el = e.target.closest('.r');
  unhover();
  if (!el) return;
  hov = usesOf(el, el.dataset.r);
  hov.forEach(x => x.classList.add('hov'));
});
const box = $('regbox');
function clearSel() {
  if (!sel) return;
  sel.list.forEach(x => x.classList.remove('sel', 'cur'));
  sel = null;
  if (box) box.hidden = true;
}
function showSel() {
  if (!box || !sel) return;
  const fn = sel.card.dataset.fn;
  const ln = el => el.closest('.ln') ? el.closest('.ln').id.slice(1) : '?';
  box.hidden = false;
  box.innerHTML = `<div><b class="t-r">${esc(sel.r)}</b> in <a href="#${esc(fn || '')}">${esc(fn || 'this block')}</a></div>` +
    `<div class="muted">${sel.list.length} use${sel.list.length > 1 ? 's' : ''} · at line ${ln(sel.list[sel.i])} (${sel.i + 1}/${sel.list.length})</div>` +
    `<div class="btns"><button type="button" data-a="-1" aria-label="Previous use">◀</button><button type="button" data-a="1" aria-label="Next use">▶</button><button type="button" data-a="x">clear</button></div>`;
}
function jump(d) {
  if (!sel) return;
  sel.list[sel.i].classList.remove('cur');
  sel.i = (sel.i + d + sel.list.length) % sel.list.length;
  const el = sel.list[sel.i];
  el.classList.add('cur');
  el.scrollIntoView({ block: 'center', inline: 'nearest' });
  showSel();
}
document.addEventListener('click', e => {
  const b = e.target.closest('#regbox button');
  if (b) { if (b.dataset.a === 'x') clearSel(); else jump(+b.dataset.a); return; }
  const el = e.target.closest('.r');
  if (!el) return;
  const r = el.dataset.r, card = scopeOf(el);
  if (sel && sel.r === r && sel.card === card) { clearSel(); return; }
  clearSel();
  const list = usesOf(el, r);
  sel = { r, card, list, i: Math.max(0, list.indexOf(el)) };
  list.forEach(x => x.classList.add('sel'));
  el.classList.add('cur');
  showSel();
});
document.addEventListener('keydown', e => {
  if (e.target.closest('input, textarea')) return;
  if (e.key === 'Escape') clearSel();
  if (sel && (e.key === 'n' || e.key === 'N')) jump(e.key === 'n' ? 1 : -1);
});

// ---------- a link into a collapsed <details> opens it ----------
function reveal() {
  const id = decodeURIComponent(location.hash.slice(1));
  if (!id) return;
  const el = document.getElementById(id);
  if (!el) return;
  let d = el.closest('details'), opened = false;
  while (d) { if (!d.open) { d.open = true; opened = true; } d = d.parentElement.closest('details'); }
  if (opened) el.scrollIntoView({ block: el.classList.contains('ln') || el.classList.contains('t-lbd') ? 'center' : 'start' });
}
window.addEventListener('hashchange', reveal);
reveal();

// ---------- find: kernels, functions, labels, parameters, arrays ----------
const find = $('find'), hits = $('hits'), toc = $('toc');
if (find) {
  const items = [];
  document.querySelectorAll('.card.fn').forEach(c => items.push({ name: c.dataset.fn, kind: c.querySelector('.badge').textContent, id: c.id }));
  document.querySelectorAll('[id*=":"]').forEach(el => {
    const [fn, name] = el.id.split(':');
    const kind = el.classList.contains('t-lbd') ? 'label' : el.classList.contains('t-pa') ? 'param' : 'array';
    items.push({ name, kind: `${kind} · ${fn}`, id: el.id });
  });
  document.querySelectorAll('[id^="g-"]').forEach(el => items.push({ name: el.id.slice(2), kind: 'data', id: el.id }));
  const run = () => {
    const q = find.value.trim().toLowerCase();
    toc.hidden = !!q; hits.hidden = !q;
    if (!q) return;
    const m = items.filter(it => it.name.toLowerCase().includes(q))
      .sort((a, b) => (a.name.toLowerCase().indexOf(q) - b.name.toLowerCase().indexOf(q)) || a.name.length - b.name.length).slice(0, 80);
    hits.innerHTML = m.length ? m.map(it => `<a href="#${encodeURIComponent(it.id)}"><span>${esc(it.name)}</span><i>${esc(it.kind)}</i></a>`).join('')
      : '<div class="none">no match</div>';
  };
  find.addEventListener('input', run);
  find.addEventListener('keydown', e => {
    if (e.key === 'Enter') { const a = hits.querySelector('a'); if (a) { location.hash = a.getAttribute('href'); } }
    if (e.key === 'Escape') { find.value = ''; run(); }
  });
  document.addEventListener('keydown', e => {
    if (e.key === '/' && !e.target.closest('input, textarea')) { e.preventDefault(); find.focus(); find.select(); }
  });
}

// ---------- the side index follows the card in view ----------
if (toc && 'IntersectionObserver' in window) {
  const links = new Map([...toc.querySelectorAll('a')].map(a => [a.getAttribute('href').slice(1), a]));
  const seen = new Set();
  const io = new IntersectionObserver(es => {
    es.forEach(e => e.isIntersecting ? seen.add(e.target.id) : seen.delete(e.target.id));
    let first = null;
    for (const id of links.keys()) if (seen.has(id)) { first = id; break; }
    links.forEach((a, id) => a.classList.toggle('on', id === first));
  }, { rootMargin: '0px 0px -60% 0px' });
  document.querySelectorAll('.card.fn').forEach(c => links.has(c.id) && io.observe(c));
}
})();
