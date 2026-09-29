'use strict';

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtBytes = b => {
  if (!b) return '0';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(i >= 3 ? 1 : 0) + u[i];
};
const baseName = p => String(p || '').split(/[\\/]/).pop();
const lsGet = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { } };

async function api(path, opt = {}) {
  const init = { method: opt.method || 'GET', headers: {} };
  if (opt.body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opt.body);
  }
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function toast(msg, err) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast show' + (err ? ' err' : '');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.className = 'toast', err ? 5000 : 2500);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    // 非安全上下文（局域网 http）下 clipboard API 不可用
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    ($('#dlg').open ? $('#dlg') : document.body).appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
  toast('已复制到剪贴板');
}

const state = { config: null, defaults: {}, server: {}, groups: [], instances: [], selected: null, es: null, openGroups: new Set() };
const ALIVE = ['loading', 'running', 'stopping'];

// ---------- 参数字段定义 ----------
const CACHE_TYPES = ['f16', 'bf16', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0', 'iq4_nl', 'f32'];
const FIELDS = [
  { k: 'mode', label: '运行方式', type: 'select', opts: [['gpu', 'GPU'], ['cpu', 'CPU']], gpu: true },
  { k: 'nGpuLayers', label: 'GPU 层数 -ngl', type: 'text', hint: '数字、auto 或 all；999 表示全部放到 GPU' },
  { k: 'device', label: 'GPU 设备 --device', type: 'text', hint: '留空自动，如 CUDA0 / Vulkan0' },
  { k: 'ctxSize', label: '上下文长度 -c', type: 'int', hint: '0 表示使用模型训练值（可能很占内存）' },
  { k: 'threads', label: 'CPU 线程数 -t', type: 'int', hint: '留空自动' },
  { k: 'parallel', label: '并发槽数 -np', type: 'int' },
  { k: 'batchSize', label: '批大小 -b', type: 'int' },
  { k: 'ubatchSize', label: '物理批大小 -ub', type: 'int' },
  { k: 'flashAttn', label: 'Flash Attention -fa', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off'], ['none', '不传递（旧版）']] },
  { k: 'cacheTypeK', label: 'K 缓存类型 -ctk', type: 'select', opts: CACHE_TYPES.map(v => [v, v]) },
  { k: 'cacheTypeV', label: 'V 缓存类型 -ctv', type: 'select', opts: CACHE_TYPES.map(v => [v, v]), hint: '量化 V 缓存需开启 Flash Attention' },
  { k: 'jinja', label: 'Jinja 模板 --jinja', type: 'bool' },
  { k: 'mlock', label: '锁定内存 --mlock', type: 'bool' },
  { k: 'noMmap', label: '禁用 mmap --no-mmap', type: 'bool' },
  { k: 'temp', label: '温度 --temp', type: 'float' },
  { k: 'topK', label: 'Top-K --top-k', type: 'int' },
  { k: 'topP', label: 'Top-P --top-p', type: 'float' },
  { k: 'minP', label: 'Min-P --min-p', type: 'float' },
  { k: 'repeatPenalty', label: '重复惩罚 --repeat-penalty', type: 'float' },
  { k: 'presencePenalty', label: '存在惩罚 --presence-penalty', type: 'float' },
  { k: 'host', label: '监听地址 --host', type: 'text', hint: '0.0.0.0 允许局域网访问' },
  { k: 'port', label: '端口 --port', type: 'int', hint: '0 表示自动分配' },
  { k: 'apiKey', label: 'API Key --api-key', type: 'text' },
  { k: 'alias', label: '模型别名 -a', type: 'text', modelOnly: true, hint: '留空使用文件名' },
  { k: 'extraArgs', label: '附加参数（原样追加到命令行）', type: 'textarea', wide: true, hint: '例如：--n-cpu-moe 10 --reasoning-budget 0' },
];
const FIELD_LABEL = Object.fromEntries(FIELDS.map(f => [f.k, f.label]));

function showVal(f, v) {
  if (v === undefined || v === null) return '';
  if (f.type === 'bool') return v ? '开启' : '关闭';
  return String(v);
}

// renderFields 生成参数表单。inherit 为回退值（用于占位提示），isModel 表示模型参数模式。
function renderFields(container, values, inherit, isModel) {
  const gpuOK = state.server.gpuOk !== false;
  container.innerHTML = '';
  for (const f of FIELDS) {
    if (f.modelOnly && !isModel) continue;
    const v = values[f.k];
    const inh = showVal(f, inherit[f.k]);
    const emptyLabel = isModel ? `继承（${inh || '未设置'}）` : (inh ? `默认（${inh}）` : '未设置');
    const lab = document.createElement('label');
    if (f.wide) lab.className = 'wide';
    let input;
    if (f.type === 'select' || f.type === 'bool') {
      const opts = f.type === 'bool' ? [['true', '开启'], ['false', '关闭']] : f.opts;
      input = `<select name="${f.k}"><option value="">${esc(emptyLabel)}</option>` +
        opts.map(([ov, ol]) => {
          const dis = f.gpu && ov === 'gpu' && !gpuOK ? ' disabled' : '';
          return `<option value="${esc(ov)}"${String(v) === ov ? ' selected' : ''}${dis}>${esc(ol)}${dis ? '（无可用 GPU）' : ''}</option>`;
        }).join('') + '</select>';
    } else if (f.type === 'textarea') {
      input = `<textarea name="${f.k}" rows="2" placeholder="${esc(isModel && inh ? '继承：' + inh : '')}">${esc(v ?? '')}</textarea>`;
    } else {
      const t = f.type === 'text' ? 'text' : 'number';
      const step = f.type === 'float' ? ' step="any"' : '';
      input = `<input name="${f.k}" type="${t}"${step} value="${esc(v ?? '')}" placeholder="${esc(inh ? (isModel ? '继承：' : '默认：') + inh : '')}">`;
    }
    lab.innerHTML = `${esc(f.label)}${input}${f.hint ? `<div class="field-hint">${esc(f.hint)}</div>` : ''}`;
    container.appendChild(lab);
  }
}

function collectFields(container) {
  const out = {};
  for (const f of FIELDS) {
    const el = container.querySelector(`[name="${f.k}"]`);
    if (!el) continue;
    const raw = el.value.trim();
    if (raw === '') continue;
    if (f.type === 'int') out[f.k] = parseInt(raw, 10);
    else if (f.type === 'float') out[f.k] = parseFloat(raw);
    else if (f.type === 'bool') out[f.k] = raw === 'true';
    else out[f.k] = f.type === 'textarea' ? el.value : raw;
    if (typeof out[f.k] === 'number' && isNaN(out[f.k])) throw new Error(`「${f.label}」不是有效数字`);
  }
  return out;
}

// ---------- 对话框 ----------
function openDialog(title, bodyHTML, buttons = []) {
  const d = $('#dlg');
  $('#dlgTitle').textContent = title;
  $('#dlgBody').innerHTML = bodyHTML;
  const foot = $('#dlgFoot');
  foot.innerHTML = '';
  for (const b of buttons) {
    const el = document.createElement('button');
    el.type = 'button';
    el.textContent = b.text;
    if (b.cls) el.className = b.cls;
    el.onclick = async () => {
      el.disabled = true;
      try { if (await b.onClick() !== false) d.close(); }
      catch (e) { toast(e.message, true); }
      finally { el.disabled = false; }
    };
    foot.appendChild(el);
  }
  if (!d.open) d.showModal();
  return $('#dlgBody');
}

// ---------- 目录/文件选择器（由服务端列目录，可得到绝对路径） ----------
// mode: 'dir' 选择目录；'server' 选择 llama-server 程序文件或其所在目录。
function pickPath(title, start, mode) {
  return new Promise(resolve => {
    let cur = '';
    let done = false;
    const finish = v => { if (!done) { done = true; resolve(v); } };
    const body = openDialog(title, `
      <div class="picker-path"><button type="button" id="pkUp">上级</button><input id="pkPath"><button type="button" id="pkGo">转到</button></div>
      <div id="pkRoots" class="roots"></div>
      <div id="pkList" class="fs-list"></div>
      <p class="muted small">${mode === 'server' ? '点击 llama-server 程序文件选中，或进入其所在目录后点「选择当前目录」。' : '进入目标目录后点「选择当前目录」。'}</p>`, [
      { text: '取消', onClick: () => { finish(null); } },
      { text: '选择当前目录', cls: 'primary', onClick: () => { finish(cur); } },
    ]);
    $('#dlg').addEventListener('close', () => finish(null), { once: true });
    const load = async p => {
      try {
        const r = await api(`/api/fs?files=${mode === 'server' ? 1 : 0}&path=${encodeURIComponent(p || '')}`);
        cur = r.path;
        $('#pkPath', body).value = r.path;
        $('#pkUp', body).disabled = !r.parent;
        $('#pkUp', body).onclick = () => load(r.parent);
        $('#pkRoots', body).innerHTML = (r.roots || []).map(x => `<button type="button" data-p="${esc(x)}">${esc(x)}</button>`).join('');
        const join = n => r.path.endsWith(r.sep) ? r.path + n : r.path + r.sep + n;
        const list = $('#pkList', body);
        list.innerHTML = r.dirs.map(n => `<div data-dir="${esc(join(n))}">📁 ${esc(n)}</div>`).join('') +
          r.files.map(n => `<div class="file${/^llama-server(\.exe)?$/i.test(n) ? ' hl' : ''}" data-file="${esc(join(n))}">📄 ${esc(n)}</div>`).join('') ||
          '<div class="muted">（空目录）</div>';
      } catch (e) { toast(e.message, true); }
    };
    $('#pkRoots', body).onclick = e => { const b = e.target.closest('[data-p]'); if (b) load(b.dataset.p); };
    $('#pkList', body).onclick = e => {
      const d = e.target.closest('[data-dir]');
      if (d) return load(d.dataset.dir);
      const f = e.target.closest('[data-file]');
      if (f) { finish(f.dataset.file); $('#dlg').close(); }
    };
    $('#pkGo', body).onclick = () => load($('#pkPath', body).value);
    $('#pkPath', body).onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); load(e.target.value); } };
    load(start);
  });
}

// ---------- 标签页 ----------
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => switchTab(b.dataset.tab));
function switchTab(name) {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.id === 'tab-' + name));
  lsSet('tab', name);
}

// ---------- 系统状态 ----------
function barHTML(pct) {
  const p = Math.max(0, Math.min(100, pct || 0));
  const cls = p >= 90 ? 'high' : p >= 75 ? 'mid' : '';
  return `<span class="bar"><i class="${cls}" style="width:${p}%"></i></span>`;
}
function tempHTML(t) {
  if (!t) return '';
  return ` <span class="${t >= 85 ? 'hot' : ''}">${t.toFixed(0)}℃</span>`;
}
async function refreshStats() {
  try {
    const s = await api('/api/stats');
    state.stats = s;
    let h = `<span class="stat">CPU ${barHTML(s.cpuPercent)} ${s.cpuPercent.toFixed(0)}%${tempHTML(s.cpuTemp)}</span>`;
    h += `<span class="stat">内存 ${barHTML(s.memPercent)} ${fmtBytes(s.memUsed)}/${fmtBytes(s.memTotal)}</span>`;
    if (s.swapTotal) {
      const sp = s.swapUsed / s.swapTotal * 100;
      h += `<span class="stat">交换 ${barHTML(sp)} ${fmtBytes(s.swapUsed)}/${fmtBytes(s.swapTotal)}</span>`;
    }
    (s.gpus || []).forEach((g, i) => {
      const mp = g.memTotal ? g.memUsed / g.memTotal * 100 : 0;
      h += `<span class="stat" title="${esc(g.name)}">GPU${i} ${barHTML(g.util)} ${g.util.toFixed(0)}%${tempHTML(g.temp)}</span>`;
      if (g.memTotal) h += `<span class="stat" title="${esc(g.name)}">显存${i} ${barHTML(mp)} ${fmtBytes(g.memUsed)}/${fmtBytes(g.memTotal)}</span>`;
    });
    $('#stats').innerHTML = h;
    updateInstanceMem();
  } catch (e) { /* 忽略临时错误 */ }
}

// ---------- 模型列表（按组折叠） ----------
async function loadModels() {
  try { state.groups = await api('/api/groups'); } catch (e) { toast(e.message, true); }
  renderModels();
}
function groupInst(g) {
  const i = state.instances.find(x => x.id === g.id);
  return i && ALIVE.includes(i.status) ? i : null;
}
const ST_TEXT = { loading: '加载中', running: '运行中', stopping: '停止中' };
function renderModels() {
  const q = $('#modelFilter').value.trim().toLowerCase();
  const dirs = state.config?.modelDirs?.length || 0;
  const list = state.groups.filter(g => !q || g.name.toLowerCase().includes(q) || g.variants.some(v => v.file.toLowerCase().includes(q)));
  const total = state.groups.reduce((n, g) => n + g.variants.length, 0);
  $('#modelHint').textContent = dirs ? `共 ${state.groups.length} 组 / ${total} 个模型文件` : '请先在「全局设置」中添加模型目录';
  $('#modelList').innerHTML = list.map(g => {
    const inst = groupInst(g);
    const tags = (inst ? `<span class="tag ${inst.status}">${ST_TEXT[inst.status]}</span>` : '') +
      (g.hasParams ? '<span class="tag custom">独立参数</span>' : '') +
      (g.mmprojs.length ? '<span class="tag vis">视觉</span>' : '');
    const ops = `<div class="ops">
        ${inst ? `<button data-act="stop" class="danger">停止</button>` : `<button data-act="start" class="primary">启动</button>`}
        <button data-act="cli">CLI 命令</button>
        <button data-act="params">参数</button>
        <button data-act="info">介绍</button>
      </div>`;
    if (g.variants.length === 1) {
      const v = g.variants[0];
      return `<div class="item" data-gid="${g.id}">
        <div class="main"><div class="name">${esc(g.name)}${tags}</div>
        <div class="muted small">${esc(v.root)} · ${esc(v.file)} · ${fmtBytes(v.size)}${v.parts > 1 ? ` · ${v.parts} 个分片` : ''}</div></div>${ops}</div>`;
    }
    const open = state.openGroups.has(g.id) || (q && list.length <= 5);
    const variants = g.variants.map(v => {
      const running = inst && inst.modelPath === v.path;
      return `<div class="variant" data-gid="${g.id}" data-variant="${esc(v.path)}">
        <div class="main"><strong>${esc(v.quant || v.name)}</strong>${running ? '<span class="tag running">运行中</span>' : ''}${v.path === g.lastVariant ? '<span class="tag last">上次运行</span>' : ''}
          <div class="muted small">${esc(v.file)} · ${fmtBytes(v.size)}${v.parts > 1 ? ` · ${v.parts} 个分片` : ''}</div></div>
        <div class="ops">${inst ? '' : '<button data-act="start">启动此版本</button>'}<button data-act="cli">CLI 命令</button></div>
      </div>`;
    }).join('');
    return `<details class="group" data-gid="${g.id}"${open ? ' open' : ''}>
      <summary class="item" data-gid="${g.id}">
        <div class="main"><div class="name"><span class="arrow">▶</span>${esc(g.name)}<span class="tag">${g.variants.length} 个版本</span>${tags}</div>
        <div class="muted small">${esc(g.root)}${g.dir ? ' · ' + esc(g.dir) + '/' : ''}</div></div>${ops}
      </summary>
      <div class="variants">${variants}</div>
    </details>`;
  }).join('') || '<div class="muted">没有找到模型（支持 目录/模型.gguf 与 目录/子目录/模型.gguf）</div>';
}
$('#modelFilter').oninput = renderModels;
$('#btnRefreshModels').onclick = loadModels;
$('#modelList').addEventListener('toggle', e => {
  const d = e.target;
  if (d.matches?.('details.group')) d.open ? state.openGroups.add(d.dataset.gid) : state.openGroups.delete(d.dataset.gid);
}, true);
$('#modelList').onclick = e => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  e.preventDefault(); // 避免点击 summary 内按钮时折叠
  const host = btn.closest('[data-gid]');
  const g = state.groups.find(x => x.id === host.dataset.gid);
  const variant = host.dataset.variant || '';
  const act = btn.dataset.act;
  if (act === 'start') runDialog(g, 'start', variant);
  else if (act === 'cli') runDialog(g, 'cli', variant);
  else if (act === 'stop') stopInstance(g.id);
  else if (act === 'params') paramsDialog(g);
  else if (act === 'info') infoDialog(g);
};

function cmdBox(id, label) {
  return `<p class="muted small">${label}</p><div class="cmd-box"><pre class="cmd" id="${id}">…</pre><button type="button" data-copy="${id}">复制</button></div>`;
}

// runDialog 启动模型或生成命令：可选择版本、是否启用视觉、本次附加参数。
async function runDialog(g, mode, variant) {
  let def;
  try { def = await api(`/api/groups/${g.id}/run`); } catch (e) { return toast(e.message, true); }
  const alive = state.instances.filter(i => ['loading', 'running'].includes(i.status) && i.id !== g.id);
  const warn = mode === 'start' && !state.config.allowMulti && alive.length
    ? `<p class="hot">当前为单开模式，启动后将停止：${alive.map(i => esc(i.modelName)).join('、')}</p>` : '';
  const sel = variant || def.variant;
  const variantHTML = g.variants.length > 1 ? `<label>模型版本<select id="rVariant">${g.variants.map(v =>
    `<option value="${esc(v.path)}"${v.path === sel ? ' selected' : ''}>${esc(v.quant || v.name)} · ${fmtBytes(v.size)}${v.path === g.lastVariant ? '（上次运行）' : ''} — ${esc(v.file)}</option>`).join('')}</select></label>` : '';
  const visionHTML = g.mmprojs.length ? `<label class="inline" style="margin-top:8px"><input type="checkbox" id="rVision"${def.vision ? ' checked' : ''}> 启用视觉功能（--mmproj）</label>` +
    (g.mmprojs.length > 1 ? `<label>视觉投影文件<select id="rMmproj">${g.mmprojs.map(p => `<option value="${esc(p)}"${p === def.mmproj ? ' selected' : ''}>${esc(baseName(p))}</option>`).join('')}</select></label>` : `<div class="muted small">投影文件：${esc(baseName(g.mmprojs[0]))}</div>`) : '';
  const serverBox = cmdBox('cmdServer', 'llama-server 命令：');
  const cliBox = cmdBox('cmdCli', 'llama-cli 命令（在终端中直接对话）：');
  const body = openDialog((mode === 'start' ? '启动模型：' : '运行命令：') + g.name, `${warn}${variantHTML}${visionHTML}
    <label style="margin-top:8px">本次附加参数<textarea id="rExtra" rows="2" placeholder="例如：--seed 42">${esc(def.extra || '')}</textarea></label>
    <div id="rWarn" class="hot small"></div>
    ${state.os === 'windows' ? '<div class="muted small">命令为 PowerShell 格式；在 cmd 中使用时去掉开头的「& 」。</div>' : ''}
    ${mode === 'cli' ? cliBox + serverBox : serverBox + cliBox}`,
    mode === 'start' ? [
      { text: '取消', onClick: () => true },
      {
        text: '启动', cls: 'primary', onClick: async () => {
          await api(`/api/groups/${g.id}/start`, { method: 'POST', body: req() });
          toast('已启动：' + g.name);
          state.selected = g.id;
          await loadInstances();
          loadModels();
          switchTab('instances');
        }
      },
    ] : [{ text: '关闭', onClick: () => true }]);
  const req = () => ({
    variant: $('#rVariant', body)?.value || sel,
    vision: $('#rVision', body)?.checked ?? false,
    mmproj: $('#rMmproj', body)?.value || g.mmprojs[0] || '',
    extra: $('#rExtra', body).value,
  });
  const preview = async () => {
    try {
      const r = await api(`/api/groups/${g.id}/command`, { method: 'POST', body: req() });
      $('#cmdServer', body).textContent = r.server;
      $('#cmdCli', body).textContent = r.cli;
      $('#rWarn', body).textContent = (r.warn ? r.warn + '。' : '') + (r.autoPort && mode === 'start' ? '端口为自动分配，实际端口以启动后为准。' : '');
    } catch (e) { $('#rWarn', body).textContent = '错误：' + e.message; }
  };
  let t;
  body.oninput = body.onchange = () => { clearTimeout(t); t = setTimeout(preview, 250); };
  body.onclick = e => { const b = e.target.closest('[data-copy]'); if (b) copyText($('#' + b.dataset.copy, body).textContent); };
  preview();
}

async function paramsDialog(g) {
  let d;
  try { d = await api(`/api/groups/${g.id}/params`); } catch (e) { return toast(e.message, true); }
  const body = openDialog('模型参数：' + g.name,
    `<p class="muted small">参数对该组所有版本生效。留空的参数继承全局配置，优先级：模型参数 → 全局参数 → 内置默认。</p><div class="grid" id="mParams"></div>`, [
    {
      text: '清空独立参数', cls: 'danger', onClick: async () => {
        if (!confirm('确定清空该模型的独立参数？')) return false;
        await api(`/api/groups/${g.id}/params`, { method: 'PUT', body: {} });
        toast('已清空'); loadModels();
      }
    },
    { text: '取消', onClick: () => true },
    {
      text: '保存', cls: 'primary', onClick: async () => {
        await api(`/api/groups/${g.id}/params`, { method: 'PUT', body: collectFields($('#mParams')) });
        toast('已保存'); loadModels();
      }
    },
  ]);
  renderFields($('#mParams', body), d.params || {}, d.inherit || {}, true);
}

// ---------- 模型介绍 ----------
let mdLibs;
function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = rej;
    document.head.appendChild(s);
  });
}
function ensureMarkdown() {
  mdLibs ??= Promise.all([
    loadScript('https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js'),
    loadScript('https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js'),
  ]).then(() => true, () => false);
  return mdLibs;
}
async function renderReadme(el, md) {
  md = (md || '').replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, ''); // 去掉 YAML 头
  if (await ensureMarkdown() && window.marked && window.DOMPurify) {
    el.innerHTML = DOMPurify.sanitize(marked.parse(md));
    el.querySelectorAll('a').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
  } else {
    el.innerHTML = `<pre style="white-space:pre-wrap">${esc(md)}</pre>`;
  }
}

function recsHTML(recs) {
  if (!recs?.length) return '<div class="recs-box muted small">未在说明中找到推荐运行参数。</div>';
  return `<div class="recs-box"><strong>推荐运行参数</strong> <span class="muted small">（自动提取，请核对出处；说明中常区分思考/非思考模式）</span>
    <table class="recs"><tr><th></th><th>参数</th><th>值</th><th>出处</th></tr>${recs.map((r, i) => `
      <tr><td><input type="checkbox" data-rec="${i}" checked></td><td>${esc(FIELD_LABEL[r.key] || r.key)}</td><td><strong>${r.value}</strong></td>
      <td class="small"><div class="muted">${esc(r.source)}</div>${esc(r.snippet)}</td></tr>`).join('')}</table>
    <button type="button" id="applyRecs" class="primary">应用选中项到该模型参数</button></div>`;
}

async function infoDialog(g, source, repo, refresh) {
  source ||= lsGet('infoSrc') || 'ms';
  const body = openDialog('模型介绍：' + g.name, `
    <div class="toolbar">
      <span class="seg"><button type="button" data-src="ms" class="${source === 'ms' ? 'active' : ''}">ModelScope</button><button type="button" data-src="hf" class="${source === 'hf' ? 'active' : ''}">HuggingFace</button></span>
      <select id="infoRepo" style="width:auto;max-width:100%"></select>
      <button type="button" id="infoRefresh">重新获取</button>
    </div>
    <div id="infoMeta" class="muted small"></div>
    <div id="infoRecs"></div>
    <div id="infoBody" class="readme">正在获取…</div>`);
  body.querySelectorAll('[data-src]').forEach(b => b.onclick = () => { lsSet('infoSrc', b.dataset.src); infoDialog(g, b.dataset.src); });
  $('#infoRefresh', body).onclick = () => infoDialog(g, source, $('#infoRepo').value, true);
  try {
    const q = new URLSearchParams({ source });
    if (repo) q.set('repo', repo);
    if (refresh) q.set('refresh', '1');
    const mi = await api(`/api/groups/${g.id}/info?${q}`);
    const sel = $('#infoRepo', body);
    const cands = mi.candidates || [];
    if (mi.repoId && !cands.some(c => c.repoId === mi.repoId)) cands.unshift({ repoId: mi.repoId });
    sel.innerHTML = cands.map(c => `<option value="${esc(c.repoId)}"${c.repoId === mi.repoId ? ' selected' : ''}>${esc(c.repoId)}${c.downloads ? ` (↓${c.downloads})` : ''}</option>`).join('') || '<option value="">无匹配结果</option>';
    sel.onchange = () => infoDialog(g, source, sel.value);
    $('#infoMeta', body).innerHTML = `搜索关键字：${esc(mi.keyword)}${mi.baseModel ? ' · 基础模型：' + esc(mi.baseModel) : ''} · ` +
      (mi.url ? `<a href="${esc(mi.url)}" target="_blank" rel="noopener">打开模型页</a> · ` : '') +
      `<a href="${esc(mi.searchUrl)}" target="_blank" rel="noopener">在网站中搜索</a> · 获取于 ${new Date(mi.fetchedAt).toLocaleString()}`;
    if (!mi.repoId) { $('#infoBody', body).textContent = '未找到匹配的模型，可点击「在网站中搜索」手动查找。'; return; }
    $('#infoRecs', body).innerHTML = recsHTML(mi.recommends);
    const btn = $('#applyRecs', body);
    if (btn) btn.onclick = async () => {
      try {
        const cur = (await api(`/api/groups/${g.id}/params`)).params || {};
        body.querySelectorAll('[data-rec]:checked').forEach(c => { const r = mi.recommends[c.dataset.rec]; cur[r.key] = r.value; });
        await api(`/api/groups/${g.id}/params`, { method: 'PUT', body: cur });
        toast('已应用到模型参数');
        loadModels();
      } catch (e) { toast(e.message, true); }
    };
    await renderReadme($('#infoBody', body), mi.readme);
  } catch (e) {
    $('#infoBody', body).textContent = '获取失败：' + e.message;
  }
}

// ---------- 运行实例 ----------
const STATUS_TEXT = { loading: '加载中', running: '运行中', stopping: '停止中', stopped: '已停止', error: '异常退出' };
async function loadInstances() {
  try { state.instances = await api('/api/instances'); } catch (e) { return; }
  const alive = state.instances.filter(i => ['loading', 'running'].includes(i.status)).length;
  $('#runCount').textContent = alive || '';
  if (!state.selected || !state.instances.some(i => i.id === state.selected)) {
    state.selected = state.instances.find(i => i.status !== 'stopped')?.id || state.instances[0]?.id || null;
  }
  renderInstances();
  renderModels();
  connectLogs();
}
function instLink(i) {
  const h = ['0.0.0.0', '::', ''].includes(i.host) ? location.hostname : i.host;
  return `http://${h.includes(':') ? `[${h}]` : h}:${i.port}`;
}
function renderInstances() {
  $('#instList').innerHTML = state.instances.map(i => {
    const alive = ALIVE.includes(i.status);
    return `<div class="inst${i.id === state.selected ? ' sel' : ''}" data-id="${i.id}">
      <div class="name"><strong>${esc(i.groupName)}</strong><span class="tag ${i.status}">${STATUS_TEXT[i.status] || i.status}</span><span class="tag">${(i.mode || '').toUpperCase()}</span>${i.vision ? '<span class="tag vis">视觉</span>' : ''}</div>
      ${i.groupName !== i.modelName ? `<div class="small">${esc(i.modelName)}</div>` : ''}
      <div class="muted small">端口 ${i.port} · PID ${i.pid} · 启动于 ${new Date(i.startedAt).toLocaleTimeString()}</div>
      <div class="muted small" data-mem="${i.id}"></div>
      ${i.status === 'running' ? `<div class="small"><a href="${instLink(i)}" target="_blank" rel="noopener">${instLink(i)}</a>（OpenAI 兼容接口：${instLink(i)}/v1）</div>` : ''}
      ${i.exitMsg && !alive ? `<div class="small muted">${esc(i.exitMsg)}</div>` : ''}
      <div class="ops">
        ${alive ? `<button data-act="stop" class="danger">停止</button>` : `<button data-act="restart" class="primary">重新启动</button><button data-act="remove">移除</button>`}
      </div>
    </div>`;
  }).join('') || '<div class="muted">暂无运行实例，请在「模型」页启动模型。</div>';
  updateInstanceMem();
}
function updateInstanceMem() {
  const s = state.stats;
  if (!s) return;
  document.querySelectorAll('[data-mem]').forEach(el => {
    const id = el.dataset.mem;
    el.textContent = s.procMem?.[id] ? `进程内存 ${fmtBytes(s.procMem[id])} · CPU ${(s.procCpu?.[id] || 0).toFixed(0)}%` : '';
  });
}
$('#instList').onclick = async e => {
  const card = e.target.closest('.inst');
  if (!card) return;
  const id = card.dataset.id;
  const btn = e.target.closest('button[data-act]');
  if (!btn) {
    if (state.selected !== id) { state.selected = id; renderInstances(); connectLogs(true); }
    return;
  }
  btn.disabled = true;
  try {
    if (btn.dataset.act === 'stop') await stopInstance(id);
    if (btn.dataset.act === 'remove') { await api(`/api/instances/${id}`, { method: 'DELETE' }); await loadInstances(); }
    if (btn.dataset.act === 'restart') {
      const inst = state.instances.find(i => i.id === id);
      await api(`/api/groups/${id}/start`, { method: 'POST', body: { variant: inst.modelPath } });
      state.selected = id;
      await loadInstances();
      connectLogs(true);
    }
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
};
async function stopInstance(id) {
  try {
    toast('正在停止…');
    await api(`/api/instances/${id}/stop`, { method: 'POST' });
    toast('已停止');
  } catch (e) { toast(e.message, true); }
  loadInstances();
}

// ---------- 日志 ----------
// 日志通过 SSE 批量接收，DOM 中只保留最近 maxLogLines 行以保证页面流畅。
let logKey = '';
let lineCount = 0;
function connectLogs(force) {
  const inst = state.instances.find(i => i.id === state.selected);
  const key = inst ? inst.id + '@' + inst.startedAt : '';
  if (!force && key === logKey) return;
  logKey = key;
  if (state.es) { state.es.close(); state.es = null; }
  const view = $('#logView');
  view.textContent = '';
  lineCount = 0;
  $('#logTitle').textContent = inst ? '日志：' + inst.modelName : '日志';
  if (!inst) return;
  const es = new EventSource(`/api/instances/${inst.id}/logs`);
  es.onmessage = ev => appendLogs(JSON.parse(ev.data));
  es.onopen = () => { view.textContent = ''; lineCount = 0; }; // 重连时服务端会重发历史
  state.es = es;
}
function lineClass(t) {
  if (t.startsWith('[面板]') || t.startsWith('$ ') || t.startsWith('& ')) return 'p';
  if (/error|failed|fatal|out of memory/i.test(t)) return 'e';
  if (/warn/i.test(t)) return 'w';
  return '';
}
function appendLogs(lines) {
  if (!lines || !lines.length) return;
  const view = $('#logView');
  const max = state.config?.maxLogLines || 1000;
  if (lines.length > max) lines = lines.slice(-max);
  const frag = document.createDocumentFragment();
  for (const l of lines) {
    const div = document.createElement('div');
    const c = lineClass(l.t);
    if (c) div.className = c;
    div.textContent = l.t;
    frag.appendChild(div);
  }
  view.appendChild(frag);
  lineCount += lines.length;
  while (lineCount > max) { view.firstChild.remove(); lineCount--; }
  $('#logInfo').textContent = `显示最近 ${lineCount} 行（上限 ${max}）`;
  if ($('#autoScroll').checked) view.scrollTop = view.scrollHeight;
}
$('#btnClearLog').onclick = () => { $('#logView').textContent = ''; lineCount = 0; };
$('#wrapLog').onchange = e => $('#logView').classList.toggle('wrap', e.target.checked);

// ---------- 设置 ----------
function renderDirs(dirs) {
  $('#dirList').innerHTML = dirs.map(d => `<div class="dir-row"><input value="${esc(d)}" data-dir><button type="button" data-pickdir>选择…</button><button type="button" class="danger" data-deldir>删除</button></div>`).join('') ||
    '<div class="muted small">尚未添加模型目录</div>';
}
function currentDirs() {
  return [...document.querySelectorAll('#dirList [data-dir]')].map(i => i.value.trim()).filter(Boolean);
}
$('#dirList').onclick = async e => {
  const row = e.target.closest('.dir-row');
  if (!row) return;
  if (e.target.matches('[data-deldir]')) { row.remove(); if (!currentDirs().length) renderDirs([]); }
  if (e.target.matches('[data-pickdir]')) {
    const p = await pickPath('选择模型目录', row.querySelector('input').value, 'dir');
    if (p) row.querySelector('input').value = p;
  }
};
$('#btnAddDir').onclick = async () => {
  const dirs = currentDirs();
  const p = await pickPath('添加模型目录', dirs[dirs.length - 1] || '', 'dir');
  if (!p) return;
  if (!dirs.includes(p)) dirs.push(p);
  renderDirs(dirs);
  try { await saveSettings(true); toast('已添加目录并保存'); loadModels(); } catch (err) { toast(err.message, true); }
};
document.querySelectorAll('[data-pick]').forEach(b => b.onclick = async () => {
  const input = $('#settingsForm')[b.dataset.pick];
  const p = await pickPath('选择 llama-server（' + (b.dataset.pick === 'serverPathGpu' ? 'GPU 版' : 'CPU 版') + '）', input.value, 'server');
  if (p) input.value = p;
});

async function loadConfig() {
  const d = await api('/api/config');
  state.config = d.config;
  state.defaults = d.defaults;
  state.os = d.os;
  const f = $('#settingsForm');
  f.serverPathGpu.value = d.config.serverPathGpu || '';
  f.serverPathCpu.value = d.config.serverPathCpu || '';
  renderDirs(d.config.modelDirs || []);
  f.allowMulti.checked = !!d.config.allowMulti;
  f.maxLogLines.value = d.config.maxLogLines;
  f.basePort.value = d.config.basePort;
  f.hfEndpoint.value = d.config.hfEndpoint || '';
  f.proxy.value = d.config.proxy || '';
  f.proxyAll.checked = !!d.config.proxyAll;
  renderFields($('#globalParams'), d.config.global || {}, d.defaults, false);
}
function probeHTML(s) {
  if (!s) return '';
  if (s.error && !s.path) return `<span class="hot">${esc(s.error)}</span>`;
  return `路径：${esc(s.path)} · 版本：${esc(s.version || '未知')}` +
    (s.devices?.length ? `<br>设备：${s.devices.map(esc).join('；')}` : '<br>设备：无 GPU 设备（仅 CPU）') +
    (s.error ? `<br><span class="hot">${esc(s.error)}</span>` : '');
}
async function probeServer(refresh) {
  $('#infoGpu').textContent = '检测中…';
  try {
    const s = await api('/api/server' + (refresh ? '?refresh=1' : ''));
    state.server = s;
    $('#infoGpu').innerHTML = probeHTML(s.gpu) + (s.gpuOk ? '' : '<br><span class="hot">GPU 不可用，模型将以 CPU 方式运行</span>');
    $('#infoCpu').innerHTML = s.cpu ? probeHTML(s.cpu) : '未配置';
  } catch (e) { $('#infoGpu').textContent = e.message; }
}
async function loadGPUInfo() {
  try {
    const g = await api('/api/gpu');
    $('#gpuBox').hidden = !g.tool;
    $('#gpuTool').textContent = g.tool ? `（${g.tool}）` : '';
    $('#gpuOut').textContent = g.output || '';
  } catch (e) { }
}
$('#btnGpu').onclick = loadGPUInfo;
$('#btnProbe').onclick = async () => {
  try {
    await saveSettings(true);
    await probeServer(true);
    renderFields($('#globalParams'), state.config.global || {}, state.defaults, false);
  } catch (e) { toast(e.message, true); }
};
async function saveSettings(silent) {
  const f = $('#settingsForm');
  const cfg = {
    serverPathGpu: f.serverPathGpu.value.trim(),
    serverPathCpu: f.serverPathCpu.value.trim(),
    modelDirs: currentDirs(),
    allowMulti: f.allowMulti.checked,
    maxLogLines: parseInt(f.maxLogLines.value, 10) || 1000,
    basePort: parseInt(f.basePort.value, 10) || 8080,
    hfEndpoint: f.hfEndpoint.value.trim(),
    proxy: f.proxy.value.trim(),
    proxyAll: f.proxyAll.checked,
    global: collectFields($('#globalParams')),
  };
  await api('/api/config', { method: 'PUT', body: cfg });
  await loadConfig();
  if (!silent) toast('设置已保存');
}
$('#settingsForm').onsubmit = async e => {
  e.preventDefault();
  try { await saveSettings(); await probeServer(true); loadModels(); } catch (err) { toast(err.message, true); }
};

// ---------- 初始化 ----------
(async function init() {
  const t = lsGet('tab'); if (t) switchTab(t);
  try { await loadConfig(); } catch (e) { toast('加载配置失败：' + e.message, true); }
  await probeServer();
  renderFields($('#globalParams'), state.config?.global || {}, state.defaults, false);
  loadGPUInfo();
  await loadInstances();
  await loadModels();
  refreshStats();
  setInterval(refreshStats, 2000);
  setInterval(loadInstances, 3000);
})();
