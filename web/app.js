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

const state = { config: null, defaults: {}, server: {}, models: [], instances: [], selected: null, es: null };

// ---------- 参数字段定义 ----------
const CACHE_TYPES = ['f16', 'bf16', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0', 'iq4_nl', 'f32'];
const FIELDS = [
  { k: 'mode', label: '运行方式', type: 'select', opts: [['gpu', 'GPU'], ['cpu', 'CPU']], gpu: true },
  { k: 'nGpuLayers', label: 'GPU 层数 -ngl', type: 'int', hint: '999 表示全部放到 GPU' },
  { k: 'device', label: 'GPU 设备 --device', type: 'text', hint: '留空自动，如 CUDA0 / Vulkan0' },
  { k: 'ctxSize', label: '上下文长度 -c', type: 'int', hint: '0 表示使用模型训练值' },
  { k: 'threads', label: 'CPU 线程数 -t', type: 'int', hint: '留空自动' },
  { k: 'parallel', label: '并发槽数 -np', type: 'int' },
  { k: 'batchSize', label: '批大小 -b', type: 'int' },
  { k: 'ubatchSize', label: '物理批大小 -ub', type: 'int' },
  { k: 'flashAttn', label: 'Flash Attention -fa', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off'], ['none', '不传递（旧版）']] },
  { k: 'cacheTypeK', label: 'K 缓存类型 -ctk', type: 'select', opts: CACHE_TYPES.map(v => [v, v]) },
  { k: 'cacheTypeV', label: 'V 缓存类型 -ctv', type: 'select', opts: CACHE_TYPES.map(v => [v, v]), hint: '量化 V 缓存需开启 Flash Attention' },
  { k: 'jinja', label: '使用 Jinja 模板 --jinja', type: 'bool' },
  { k: 'mlock', label: '锁定内存 --mlock', type: 'bool' },
  { k: 'noMmap', label: '禁用 mmap --no-mmap', type: 'bool' },
  { k: 'temp', label: '温度 --temp', type: 'float' },
  { k: 'topK', label: 'Top-K --top-k', type: 'int' },
  { k: 'topP', label: 'Top-P --top-p', type: 'float' },
  { k: 'minP', label: 'Min-P --min-p', type: 'float' },
  { k: 'repeatPenalty', label: '重复惩罚 --repeat-penalty', type: 'float' },
  { k: 'host', label: '监听地址 --host', type: 'text', hint: '0.0.0.0 允许局域网访问' },
  { k: 'port', label: '端口 --port', type: 'int', hint: '0 表示自动分配' },
  { k: 'apiKey', label: 'API Key --api-key', type: 'text' },
  { k: 'alias', label: '模型别名 -a', type: 'text', modelOnly: true, hint: '留空使用文件名' },
  { k: 'mmproj', label: '多模态投影 --mmproj', type: 'mmproj', modelOnly: true },
  { k: 'extraArgs', label: '附加参数（原样追加到命令行）', type: 'textarea', wide: true, hint: '例如：--rope-scaling yarn --n-cpu-moe 10' },
];

function showVal(f, v) {
  if (v === undefined || v === null) return '';
  if (f.type === 'bool') return v ? '开启' : '关闭';
  return String(v);
}

// renderFields 生成参数表单。inherit 为回退值（用于占位提示），model 为模型对象（模型参数模式）。
function renderFields(container, values, inherit, model) {
  const gpuOK = state.server.gpu !== false;
  container.innerHTML = '';
  for (const f of FIELDS) {
    if (f.modelOnly && !model) continue;
    const v = values[f.k];
    const inh = showVal(f, inherit[f.k]);
    const emptyLabel = model ? `继承（${inh || '未设置'}）` : (inh ? `默认（${inh}）` : '未设置');
    const lab = document.createElement('label');
    if (f.wide) lab.className = 'wide';
    let input;
    if (f.type === 'select' || f.type === 'bool' || f.type === 'mmproj') {
      let opts = f.opts;
      if (f.type === 'bool') opts = [['true', '开启'], ['false', '关闭']];
      if (f.type === 'mmproj') opts = (model.mmprojs || []).map(p => [p, p.split(/[\\/]/).pop()]);
      input = `<select name="${f.k}"><option value="">${esc(emptyLabel)}</option>` +
        opts.map(([ov, ol]) => {
          const dis = f.gpu && ov === 'gpu' && !gpuOK ? ' disabled' : '';
          return `<option value="${esc(ov)}"${String(v) === ov ? ' selected' : ''}${dis}>${esc(ol)}${dis ? '（无可用 GPU）' : ''}</option>`;
        }).join('') + '</select>';
    } else if (f.type === 'textarea') {
      input = `<textarea name="${f.k}" rows="2" placeholder="${esc(model && inh ? '继承：' + inh : '')}">${esc(v ?? '')}</textarea>`;
    } else {
      const t = f.type === 'text' ? 'text' : 'number';
      const step = f.type === 'float' ? ' step="any"' : '';
      input = `<input name="${f.k}" type="${t}"${step} value="${esc(v ?? '')}" placeholder="${esc(inh ? (model ? '继承：' : '默认：') + inh : '')}">`;
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

// ---------- 标签页 ----------
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => switchTab(b.dataset.tab));
function switchTab(name) {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.id === 'tab-' + name));
  try { localStorage.setItem('tab', name); } catch (e) { }
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
    h += `<span class="stat" title="已用 ${fmtBytes(s.memUsed)} / ${fmtBytes(s.memTotal)}">内存 ${barHTML(s.memPercent)} ${fmtBytes(s.memUsed)}/${fmtBytes(s.memTotal)}</span>`;
    if (s.swapTotal) {
      const sp = s.swapUsed / s.swapTotal * 100;
      h += `<span class="stat">交换 ${barHTML(sp)} ${fmtBytes(s.swapUsed)}/${fmtBytes(s.swapTotal)}</span>`;
    }
    (s.gpus || []).forEach((g, i) => {
      const mp = g.memTotal ? g.memUsed / g.memTotal * 100 : 0;
      h += `<span class="stat" title="${esc(g.name)}">GPU${i} ${barHTML(g.util)} ${g.util.toFixed(0)}%${tempHTML(g.temp)}</span>`;
      if (g.memTotal) h += `<span class="stat">显存${i} ${barHTML(mp)} ${fmtBytes(g.memUsed)}/${fmtBytes(g.memTotal)}</span>`;
    });
    $('#stats').innerHTML = h;
    updateInstanceMem();
  } catch (e) { /* 忽略临时错误 */ }
}

// ---------- 模型列表 ----------
async function loadModels() {
  try {
    state.models = await api('/api/models');
  } catch (e) { toast(e.message, true); }
  renderModels();
}
function renderModels() {
  const q = $('#modelFilter').value.trim().toLowerCase();
  const running = new Map(state.instances.map(i => [i.id, i]));
  const list = state.models.filter(m => !q || m.file.toLowerCase().includes(q));
  const dirs = state.config?.modelDirs?.length || 0;
  $('#modelHint').textContent = dirs ? `共 ${state.models.length} 个模型` : '请先在「全局设置」中添加模型目录';
  $('#modelList').innerHTML = list.map(m => {
    const inst = running.get(m.id);
    const st = inst && ['loading', 'running', 'stopping'].includes(inst.status) ? inst.status : '';
    const stText = { loading: '加载中', running: '运行中', stopping: '停止中' }[st];
    return `<div class="item" data-id="${m.id}">
      <div class="main">
        <div class="name">${esc(m.name)}${st ? `<span class="tag ${st}">${stText}</span>` : ''}${m.hasParams ? '<span class="tag custom">独立参数</span>' : ''}${m.mmprojs.length ? '<span class="tag">多模态</span>' : ''}</div>
        <div class="muted small">${esc(m.root)} · ${esc(m.file)} · ${fmtBytes(m.size)}${m.parts > 1 ? ` · ${m.parts} 个分片` : ''}</div>
      </div>
      <div class="ops">
        ${st ? `<button data-act="stop" class="danger">停止</button>` : `<button data-act="start" class="primary">启动</button>`}
        <button data-act="params">参数</button>
        <button data-act="info">介绍</button>
      </div>
    </div>`;
  }).join('') || '<div class="muted">没有找到模型（支持 目录/模型.gguf 与 目录/子目录/模型.gguf）</div>';
}
$('#modelFilter').oninput = renderModels;
$('#btnRefreshModels').onclick = loadModels;
$('#modelList').onclick = e => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const m = state.models.find(x => x.id === btn.closest('.item').dataset.id);
  ({ start: startDialog, stop: m => stopInstance(m.id), params: paramsDialog, info: infoDialog })[btn.dataset.act](m);
};

async function startDialog(m) {
  const alive = state.instances.filter(i => ['loading', 'running'].includes(i.status));
  const warn = !state.config.allowMulti && alive.length
    ? `<p class="hot">当前为单开模式，启动后将停止：${alive.map(i => esc(i.modelName)).join('、')}</p>` : '';
  const body = openDialog('启动模型：' + m.name, `${warn}
    <label>本次附加参数（仅本次启动生效）<textarea id="startExtra" rows="2" placeholder="例如：--seed 42 --no-webui"></textarea></label>
    <p class="muted small">命令预览：</p><pre class="cmd" id="cmdPreview">…</pre>`, [
    { text: '取消', onClick: () => true },
    {
      text: '启动', cls: 'primary', onClick: async () => {
        await api(`/api/models/${m.id}/start`, { method: 'POST', body: { extra: $('#startExtra').value } });
        toast('已启动：' + m.name);
        state.selected = m.id;
        await loadInstances();
        switchTab('instances');
      }
    },
  ]);
  const preview = async () => {
    try {
      const r = await api(`/api/models/${m.id}/command?extra=${encodeURIComponent($('#startExtra').value)}`);
      $('#cmdPreview').textContent = r.command + (r.autoPort ? '\n\n（端口为自动分配，实际端口可能不同）' : '');
    } catch (e) { $('#cmdPreview').textContent = '错误：' + e.message; }
  };
  let t;
  $('#startExtra', body).oninput = () => { clearTimeout(t); t = setTimeout(preview, 300); };
  preview();
}

async function paramsDialog(m) {
  let d;
  try { d = await api(`/api/models/${m.id}/params`); } catch (e) { return toast(e.message, true); }
  const body = openDialog('模型参数：' + m.name,
    `<p class="muted small">留空的参数继承全局配置。优先级：模型参数 → 全局参数 → 内置默认。</p><div class="grid" id="mParams"></div>`, [
    {
      text: '清空独立参数', cls: 'danger', onClick: async () => {
        if (!confirm('确定清空该模型的独立参数？')) return false;
        await api(`/api/models/${m.id}/params`, { method: 'PUT', body: {} });
        toast('已清空'); loadModels();
      }
    },
    { text: '取消', onClick: () => true },
    {
      text: '保存', cls: 'primary', onClick: async () => {
        await api(`/api/models/${m.id}/params`, { method: 'PUT', body: collectFields($('#mParams')) });
        toast('已保存'); loadModels();
      }
    },
  ]);
  renderFields($('#mParams', body), d.params || {}, d.inherit || {}, d.model);
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

async function infoDialog(m, source, repo, refresh) {
  source ||= (() => { try { return localStorage.getItem('infoSrc'); } catch (e) { } })() || 'ms';
  const body = openDialog('模型介绍：' + m.name, `
    <div class="toolbar">
      <span class="seg"><button type="button" data-src="ms" class="${source === 'ms' ? 'active' : ''}">ModelScope</button><button type="button" data-src="hf" class="${source === 'hf' ? 'active' : ''}">HuggingFace</button></span>
      <select id="infoRepo" style="width:auto;max-width:100%"></select>
      <button type="button" id="infoRefresh">重新获取</button>
    </div>
    <div id="infoMeta" class="muted small"></div>
    <div id="infoBody" class="readme">正在获取…</div>`);
  body.querySelectorAll('[data-src]').forEach(b => b.onclick = () => {
    try { localStorage.setItem('infoSrc', b.dataset.src); } catch (e) { }
    infoDialog(m, b.dataset.src);
  });
  $('#infoRefresh', body).onclick = () => infoDialog(m, source, $('#infoRepo').value, true);
  try {
    const q = new URLSearchParams({ source });
    if (repo) q.set('repo', repo);
    if (refresh) q.set('refresh', '1');
    const mi = await api(`/api/models/${m.id}/info?${q}`);
    const sel = $('#infoRepo', body);
    const cands = mi.candidates || [];
    if (mi.repoId && !cands.some(c => c.repoId === mi.repoId)) cands.unshift({ repoId: mi.repoId });
    sel.innerHTML = cands.map(c => `<option value="${esc(c.repoId)}"${c.repoId === mi.repoId ? ' selected' : ''}>${esc(c.repoId)}${c.downloads ? ` (↓${c.downloads})` : ''}</option>`).join('') || '<option value="">无匹配结果</option>';
    sel.onchange = () => infoDialog(m, source, sel.value);
    $('#infoMeta', body).innerHTML = `搜索关键字：${esc(mi.keyword)} · ` +
      (mi.url ? `<a href="${esc(mi.url)}" target="_blank" rel="noopener">打开模型页</a> · ` : '') +
      `<a href="${esc(mi.searchUrl)}" target="_blank" rel="noopener">在网站中搜索</a> · 获取于 ${new Date(mi.fetchedAt).toLocaleString()}`;
    if (!mi.repoId) $('#infoBody', body).textContent = '未找到匹配的模型，可点击「在网站中搜索」手动查找。';
    else await renderReadme($('#infoBody', body), mi.readme);
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
    const alive = ['loading', 'running', 'stopping'].includes(i.status);
    return `<div class="inst${i.id === state.selected ? ' sel' : ''}" data-id="${i.id}">
      <div class="name"><strong>${esc(i.modelName)}</strong><span class="tag ${i.status}">${STATUS_TEXT[i.status] || i.status}</span></div>
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
      await api(`/api/models/${id}/start`, { method: 'POST', body: {} });
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
  es.onerror = () => { /* 浏览器会自动重连；重连时重新加载 */ };
  es.onopen = () => { view.textContent = ''; lineCount = 0; };
  state.es = es;
}
let lineCount = 0;
function lineClass(t) {
  if (t.startsWith('[面板]') || t.startsWith('$ ')) return 'p';
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
async function loadConfig() {
  const d = await api('/api/config');
  state.config = d.config;
  state.defaults = d.defaults;
  const f = $('#settingsForm');
  f.llamaServerPath.value = d.config.llamaServerPath || '';
  f.modelDirs.value = (d.config.modelDirs || []).join('\n');
  f.allowMulti.checked = !!d.config.allowMulti;
  f.maxLogLines.value = d.config.maxLogLines;
  f.basePort.value = d.config.basePort;
  f.hfEndpoint.value = d.config.hfEndpoint || '';
  renderFields($('#globalParams'), d.config.global || {}, d.defaults, null);
}
async function probeServer(refresh) {
  $('#serverInfo').textContent = '检测中…';
  try {
    const s = await api('/api/server' + (refresh ? '?refresh=1' : ''));
    state.server = s;
    $('#serverInfo').innerHTML = s.error && !s.path ? `<span class="hot">${esc(s.error)}</span>` :
      `路径：${esc(s.path)}<br>版本：${esc(s.version || '未知')}<br>` +
      `GPU：${s.gpu ? '可用' : '<span class="hot">不可用（将仅使用 CPU 运行）</span>'}` +
      (s.devices?.length ? `<br>设备：${s.devices.map(esc).join('；')}` : '') +
      (s.error ? `<br><span class="hot">${esc(s.error)}</span>` : '');
  } catch (e) { $('#serverInfo').textContent = e.message; }
}
$('#btnProbe').onclick = async () => {
  // 先保存路径再检测
  await saveSettings(true);
  await probeServer(true);
  renderFields($('#globalParams'), state.config.global || {}, state.defaults, null);
};
async function saveSettings(silent) {
  const f = $('#settingsForm');
  const cfg = {
    llamaServerPath: f.llamaServerPath.value.trim(),
    modelDirs: f.modelDirs.value.split('\n').map(s => s.trim()).filter(Boolean),
    allowMulti: f.allowMulti.checked,
    maxLogLines: parseInt(f.maxLogLines.value, 10) || 1000,
    basePort: parseInt(f.basePort.value, 10) || 8080,
    hfEndpoint: f.hfEndpoint.value.trim(),
    global: collectFields($('#globalParams')),
  };
  await api('/api/config', { method: 'PUT', body: cfg });
  await loadConfig();
  if (!silent) toast('设置已保存');
}
$('#settingsForm').onsubmit = async e => {
  e.preventDefault();
  try { await saveSettings(); await probeServer(); loadModels(); } catch (err) { toast(err.message, true); }
};

// ---------- 初始化 ----------
(async function init() {
  try { const t = localStorage.getItem('tab'); if (t) switchTab(t); } catch (e) { }
  try { await loadConfig(); } catch (e) { toast('加载配置失败：' + e.message, true); }
  await probeServer();
  renderFields($('#globalParams'), state.config?.global || {}, state.defaults, null);
  await loadInstances();
  await loadModels();
  refreshStats();
  setInterval(refreshStats, 2000);
  setInterval(loadInstances, 3000);
})();
