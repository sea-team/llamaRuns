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
const ONOFF = [['on', 'on'], ['off', 'off']];
const FIELDS = [
  { sec: 'GPU 与显存', note: '仅 GPU 运行方式生效' },
  { k: 'mode', label: '运行方式', type: 'select', opts: [['gpu', 'GPU'], ['cpu', 'CPU']], gpu: true },
  { k: 'nGpuLayers', label: 'GPU 层数 -ngl', type: 'text', hint: '显存充裕（如 24GB+ 跑 7B/8B）填 999：全层卸载最快、行为确定可复现；显存紧张或多模型混跑用 auto 并开启 -fit，自动适配显存' },
  { k: 'fit', label: '自动适配显存 -fit', type: 'select', opts: ONOFF, hint: '根据显存自动调整未设置的参数（新版默认 on）' },
  { k: 'fitTarget', label: '每设备预留显存 -fitt (MiB)', type: 'text', hint: '默认 1024；多卡可用逗号分隔，如 1024,2048' },
  { k: 'fitCtx', label: '-fit 最小上下文 -fitc', type: 'int', hint: '默认 4096' },
  { k: 'device', label: 'GPU 设备 --device', type: 'text', hint: '留空自动，如 CUDA0 / Vulkan0，多个用逗号分隔' },
  { k: 'splitMode', label: '多卡切分 -sm', type: 'select', opts: [['layer', 'layer（按层）'], ['row', 'row（按行）'], ['tensor', 'tensor'], ['none', 'none（单卡）']] },
  { k: 'tensorSplit', label: '多卡比例 -ts', type: 'text', hint: '如 3,1' },
  { k: 'mainGpu', label: '主 GPU -mg', type: 'int' },
  { k: 'cpuMoe', label: 'MoE 专家全放 CPU -cmoe', type: 'bool' },
  { k: 'nCpuMoe', label: '前 N 层 MoE 放 CPU -ncmoe', type: 'int', hint: '显存不足跑 MoE 模型时使用' },
  { k: 'kvOffload', label: 'KV 缓存放 GPU', type: 'bool', hint: '关闭时传 -nkvo，节省显存但变慢' },
  { sec: '上下文与性能' },
  { k: 'ctxSize', label: '上下文长度 -c', type: 'int', hint: '0 表示使用模型训练值（可能很占内存）' },
  { k: 'threads', label: 'CPU 线程数 -t', type: 'int', hint: '留空自动' },
  { k: 'threadsBatch', label: '批处理线程数 -tb', type: 'int', hint: '默认同 -t' },
  { k: 'parallel', label: '并发槽数 -np', type: 'int', hint: '默认自动' },
  { k: 'batchSize', label: '批大小 -b', type: 'int' },
  { k: 'ubatchSize', label: '物理批大小 -ub', type: 'int' },
  { k: 'flashAttn', label: 'Flash Attention -fa', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off'], ['none', '不传递（旧版）']] },
  { k: 'cacheTypeK', label: 'K 缓存类型 -ctk', type: 'select', opts: CACHE_TYPES.map(v => [v, v]) },
  { k: 'cacheTypeV', label: 'V 缓存类型 -ctv', type: 'select', opts: CACHE_TYPES.map(v => [v, v]), hint: 'q8_0 可省一半 KV 显存' },
  { k: 'cacheRam', label: '提示缓存上限 -cram (MiB)', type: 'int', hint: '默认 8192，-1 不限，0 关闭' },
  { k: 'mlock', label: '锁定内存 --mlock', type: 'bool' },
  { k: 'noMmap', label: '禁用 mmap --no-mmap', type: 'bool' },
  { k: 'contextShift', label: '上下文平移 --context-shift', type: 'bool', hint: '超长生成时丢弃早期内容' },
  { sec: '采样' },
  { k: 'temp', label: '温度 --temp', type: 'float', hint: '默认 0.8' },
  { k: 'topK', label: 'Top-K --top-k', type: 'int', hint: '默认 40' },
  { k: 'topP', label: 'Top-P --top-p', type: 'float', hint: '默认 0.95' },
  { k: 'minP', label: 'Min-P --min-p', type: 'float', hint: '默认 0.05' },
  { k: 'repeatPenalty', label: '重复惩罚 --repeat-penalty', type: 'float', hint: '默认 1.0（关闭）' },
  { k: 'presencePenalty', label: '存在惩罚 --presence-penalty', type: 'float', hint: '默认 0' },
  { k: 'seed', label: '随机种子 -s', type: 'int', hint: '-1 随机' },
  { k: 'nPredict', label: '最大生成长度 -n', type: 'int', hint: '-1 不限' },
  { sec: '模板与推理' },
  { k: 'jinja', label: 'Jinja 模板 --jinja', type: 'bool', hint: '新版默认开启' },
  { k: 'reasoning', label: '思考模式 -rea', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off']] },
  { k: 'reasoningBudget', label: '思考预算 --reasoning-budget', type: 'int', hint: '-1 不限，0 立即结束思考' },
  { k: 'chatTemplateKwargs', label: '模板参数 --chat-template-kwargs', type: 'text', hint: `JSON，如 {"enable_thinking":false}` },
  { sec: '服务', note: '仅 llama-server 生效' },
  { k: 'host', label: '监听地址 --host', type: 'text', hint: '0.0.0.0 允许局域网访问' },
  { k: 'port', label: '端口 --port', type: 'int', hint: '0 表示自动分配' },
  { k: 'apiKey', label: 'API Key --api-key', type: 'text' },
  { k: 'alias', label: '模型别名 -a', type: 'text', modelOnly: true, hint: '留空使用文件名' },
  { k: 'webui', label: '内置 Web UI', type: 'bool', hint: '关闭时传 --no-webui' },
  { k: 'metrics', label: '监控指标 --metrics', type: 'bool' },
  { sec: '其它' },
  { k: 'extraArgs', label: '附加参数（原样追加到命令行）', type: 'textarea', wide: true, hint: '其它参数见 llama-server --help，例如：--rope-scaling yarn --swa-full' },
];
const FIELD_LABEL = Object.fromEntries(FIELDS.filter(f => f.k).map(f => [f.k, f.label]));

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
    if (f.sec) {
      const h = document.createElement('div');
      h.className = 'fsec';
      h.innerHTML = esc(f.sec) + (f.note ? ` <small>${esc(f.note)}</small>` : '');
      container.appendChild(h);
      continue;
    }
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
  for (const f of FIELDS.filter(f => f.k)) {
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
const RECENT_KEY = 'recentDirs';
function pickPath(title, start, mode) {
  return new Promise(resolve => {
    let cur = '', chosen = '', filter = '', data = null;
    let done = false;
    const finish = v => {
      if (done) return;
      done = true;
      if (v) {
        const d = mode === 'dir' ? v : cur;
        const rec = [d, ...JSON.parse(lsGet(RECENT_KEY) || '[]').filter(x => x !== d)].slice(0, 6);
        lsSet(RECENT_KEY, JSON.stringify(rec));
      }
      resolve(v);
    };
    const isServer = n => /^llama-server(\.exe)?$/i.test(n);
    const body = openDialog(title, `
      <div class="picker">
        <div class="picker-top">
          <button type="button" id="pkUp" title="上级目录">⬆</button>
          <input id="pkPath" spellcheck="false" placeholder="输入路径后回车">
          <input id="pkFilter" placeholder="筛选…" style="flex:0 0 120px">
        </div>
        <div id="pkCrumbs" class="crumbs"></div>
        <div class="picker-main">
          <div id="pkSide" class="picker-side"></div>
          <div id="pkList" class="picker-list"></div>
        </div>
        <div id="pkSel" class="picker-sel"></div>
      </div>`, [
      { text: '取消', onClick: () => { finish(null); } },
      { text: '确定', cls: 'primary', onClick: () => { finish(chosen || cur); } },
    ]);
    $('#dlg').addEventListener('close', () => finish(null), { once: true });
    const join = n => data.path.endsWith(data.sep) ? data.path + n : data.path + data.sep + n;
    const showSel = () => {
      $('#pkSel', body).innerHTML = `将选择：<b>${esc(chosen || cur)}</b>` +
        (mode === 'server' && !chosen ? '（目录，程序会在其中查找 llama-server）' : '');
    };
    const render = () => {
      const f = filter.toLowerCase();
      const dirs = data.dirs.filter(n => !f || n.toLowerCase().includes(f));
      const files = data.files.filter(n => !f || n.toLowerCase().includes(f));
      // 服务端程序排在文件最前
      files.sort((a, b) => isServer(b) - isServer(a));
      $('#pkList', body).innerHTML = dirs.map(n => `<div class="fs-item" data-dir="${esc(join(n))}"><span class="ic">📁</span><span class="nm">${esc(n)}</span></div>`).join('') +
        files.map(n => `<div class="fs-item file${isServer(n) ? ' hl' : ''}${join(n) === chosen ? ' sel' : ''}" data-file="${esc(join(n))}"><span class="ic">${isServer(n) ? '⚙️' : '📄'}</span><span class="nm">${esc(n)}</span>${isServer(n) ? '<span class="hint">推荐</span>' : ''}</div>`).join('') ||
        `<div class="fs-item muted">${f ? '没有匹配项' : '（空目录）'}</div>`;
    };
    const load = async p => {
      try {
        data = await api(`/api/fs?files=${mode === 'server' ? 1 : 0}&path=${encodeURIComponent(p || '')}`);
        cur = data.path; chosen = ''; filter = '';
        $('#pkFilter', body).value = '';
        $('#pkPath', body).value = cur;
        $('#pkUp', body).disabled = !data.parent;
        $('#pkUp', body).onclick = () => load(data.parent);
        // 面包屑
        const parts = [];
        let acc = '';
        const segs = cur.split(data.sep).filter((x, i) => x || i === 0);
        segs.forEach((seg, i) => {
          acc = i === 0 ? (seg === '' ? data.sep : seg + data.sep) : (acc.endsWith(data.sep) ? acc : acc + data.sep) + seg;
          parts.push(`<button type="button" data-p="${esc(acc)}">${esc(seg || data.sep)}</button>`);
        });
        $('#pkCrumbs', body).innerHTML = parts.join('<span class="sep">›</span>');
        // 侧栏：位置与最近使用
        const recent = JSON.parse(lsGet(RECENT_KEY) || '[]');
        $('#pkSide', body).innerHTML = '<div class="lbl">位置</div>' +
          (data.roots || []).map(x => `<button type="button" data-p="${esc(x)}" title="${esc(x)}" class="${cur === x ? 'on' : ''}">${x.length <= 3 ? '💽' : '🏠'} ${esc(x)}</button>`).join('') +
          (recent.length ? '<div class="lbl">最近使用</div>' + recent.map(x => `<button type="button" data-p="${esc(x)}" title="${esc(x)}">🕘 ${esc(baseName(x) || x)}</button>`).join('') : '');
        render();
        // 当前目录包含 llama-server 时自动选中
        if (mode === 'server') {
          const hit = data.files.find(isServer);
          if (hit) { chosen = join(hit); render(); }
        }
        showSel();
      } catch (e) { toast(e.message, true); }
    };
    body.addEventListener('click', e => {
      const b = e.target.closest('[data-p]');
      if (b) load(b.dataset.p);
    });
    const list = $('#pkList', body);
    list.onclick = e => {
      const d = e.target.closest('[data-dir]');
      if (d) return load(d.dataset.dir);
      const f = e.target.closest('[data-file]');
      if (f) { chosen = f.dataset.file; render(); showSel(); }
    };
    list.ondblclick = e => {
      const f = e.target.closest('[data-file]');
      if (f) { finish(f.dataset.file); $('#dlg').close(); }
    };
    $('#pkFilter', body).oninput = e => { filter = e.target.value; render(); };
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
const HIST_LEN = 90; // 保留约 3 分钟历史
const hist = {};
function pushHist(key, v) {
  (hist[key] ||= []).push(v);
  if (hist[key].length > HIST_LEN) hist[key].shift();
}
function sparkSVG(key) {
  const d = hist[key] || [];
  if (d.length < 2) return '<svg class="spark"></svg>';
  const w = 300, h = 36, step = w / (HIST_LEN - 1), x0 = w - (d.length - 1) * step;
  const pts = d.map((v, i) => `${(x0 + i * step).toFixed(1)},${(h - Math.max(0, Math.min(100, v)) / 100 * (h - 2) - 1).toFixed(1)}`);
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path class="area" d="M${pts[0].split(',')[0]},${h}L${pts.join('L')}L${w},${h}Z"/><path d="M${pts.join('L')}"/></svg>`;
}
function levelCls(p) { return p >= 90 ? 'high' : p >= 75 ? 'mid' : ''; }
function metricHTML(label, pct, text, key) {
  const p = Math.max(0, Math.min(100, pct || 0));
  return `<div class="metric"><div class="row"><span>${label}</span><span>${text}</span></div>
    <div class="big"><i class="${levelCls(p)}" style="width:${p}%"></i></div>${key ? sparkSVG(key) : ''}</div>`;
}
function tempText(t) { return t ? `<span class="${t >= 85 ? 'hot' : ''}">${t.toFixed(0)}℃</span>` : '—'; }

async function loadHost() {
  try {
    const h = await api('/api/host');
    const up = h.bootTime ? Math.floor((Date.now() / 1000 - h.bootTime) / 3600) : 0;
    $('#sysHost').textContent = [h.hostname, h.platform || h.os, h.arch, up ? `已运行 ${up} 小时` : ''].filter(Boolean).join(' · ');
    state.host = h;
  } catch (e) { }
}

async function refreshStats() {
  let s;
  try { s = await api('/api/stats'); } catch (e) { return; }
  state.stats = s;
  const swapPct = s.swapTotal ? s.swapUsed / s.swapTotal * 100 : 0;
  pushHist('cpu', s.cpuPercent);
  pushHist('mem', s.memPercent);
  (s.gpus || []).forEach((g, i) => {
    pushHist('gpu' + i, g.util);
    pushHist('vram' + i, g.memTotal ? g.memUsed / g.memTotal * 100 : 0);
  });
  // 顶部精简状态：只显示最关键的占用率，便于在其它页面留意是否爆内存
  const vram = (s.gpus || []).filter(g => g.memTotal).map(g => g.memUsed / g.memTotal * 100);
  const pctB = p => `<b class="${p >= 90 ? 'hot' : ''}">${p.toFixed(0)}%</b>`;
  $('#miniStat').innerHTML = `CPU ${pctB(s.cpuPercent)} · 内存 ${pctB(s.memPercent)}` + (vram.length ? ` · 显存 ${vram.map(pctB).join('/')}` : '');
  updateInstanceMem();
  if (!$('#tab-system').classList.contains('active')) return;

  const h = state.host || {};
  let cards = `<div class="card"><h4>CPU <small title="${esc(h.cpuModel)}">${esc(h.cpuModel || '')}</small></h4>
    ${metricHTML('占用率', s.cpuPercent, s.cpuPercent.toFixed(0) + '%', 'cpu')}
    <div class="kv"><span>温度</span><span>${tempText(s.cpuTemp)}</span><span>核心</span><span>${h.physicalCores || '?'} 物理 / ${h.logicalCores || s.cpuCores} 逻辑</span></div></div>`;
  cards += `<div class="card"><h4>内存 <small>${fmtBytes(s.memTotal)}</small></h4>
    ${metricHTML('已用', s.memPercent, `${fmtBytes(s.memUsed)} / ${fmtBytes(s.memTotal)}（${s.memPercent.toFixed(0)}%）`, 'mem')}
    ${s.swapTotal ? metricHTML('交换 / 虚拟内存', swapPct, `${fmtBytes(s.swapUsed)} / ${fmtBytes(s.swapTotal)}`) : ''}
    <div class="kv"><span>可用</span><span>${fmtBytes(s.memTotal - s.memUsed)}</span></div></div>`;
  (s.gpus || []).forEach((g, i) => {
    const mp = g.memTotal ? g.memUsed / g.memTotal * 100 : 0;
    const kv = [['温度', tempText(g.temp)]];
    if (g.power) kv.push(['功耗', `${g.power.toFixed(0)} W${g.powerLimit ? ' / ' + g.powerLimit.toFixed(0) + ' W' : ''}`]);
    if (g.fan) kv.push(['风扇', g.fan.toFixed(0) + '%']);
    if (g.pstate) kv.push(['性能状态', esc(g.pstate)]);
    if (g.driver) kv.push(['驱动', esc(g.driver)]);
    if (g.cuda) kv.push(['CUDA', esc(g.cuda)]);
    cards += `<div class="card"><h4>GPU ${i} <small title="${esc(g.name)}">${esc(g.name)}</small></h4>
      ${metricHTML('占用率', g.util, g.util.toFixed(0) + '%', 'gpu' + i)}
      ${g.memTotal ? metricHTML('显存', mp, `${fmtBytes(g.memUsed)} / ${fmtBytes(g.memTotal)}（${mp.toFixed(0)}%）`, 'vram' + i) : ''}
      <div class="kv">${kv.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('')}</div></div>`;
  });
  if (!(s.gpus || []).length) cards += `<div class="card"><h4>GPU</h4><div class="muted small">未检测到 NVIDIA / AMD 显卡监控工具（nvidia-smi / rocm-smi），不显示显卡信息。</div></div>`;
  $('#sysCards').innerHTML = cards;

  const alive = state.instances.filter(i => ALIVE.includes(i.status));
  $('#sysProcs').innerHTML = alive.length ? `<table class="procs"><tr><th>模型</th><th>方式</th><th>端口</th><th>进程内存</th><th>CPU</th></tr>${alive.map(i =>
    `<tr><td>${esc(i.modelName)}</td><td>${(i.mode || '').toUpperCase()}</td><td>${i.port}</td><td>${fmtBytes(s.procMem?.[i.id] || 0)}</td><td>${(s.procCpu?.[i.id] || 0).toFixed(0)}%</td></tr>`).join('')}</table>`
    : '<div class="muted small">暂无运行中的模型</div>';
}
$('#miniStat').onclick = () => { switchTab('system'); refreshStats(); };

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
  const modeHTML = def.gpuOk
    ? `<label style="margin-top:8px">运行方式<span class="seg" id="rMode" style="display:flex;width:max-content;margin-top:4px">${[['gpu', 'GPU'], ['cpu', 'CPU']].map(([v, l]) =>
      `<button type="button" data-mode="${v}" class="${def.mode === v ? 'active' : ''}">${l}</button>`).join('')}</span></label>`
    : '<div class="muted small" style="margin-top:8px">未检测到可用 GPU，将以 CPU 方式运行</div>';
  const visionHTML = g.mmprojs.length ? `<label class="inline" style="margin-top:8px"><input type="checkbox" id="rVision"${def.vision ? ' checked' : ''}> 启用视觉功能（--mmproj）</label>` +
    (g.mmprojs.length > 1 ? `<label>视觉投影文件<select id="rMmproj">${g.mmprojs.map(p => `<option value="${esc(p)}"${p === def.mmproj ? ' selected' : ''}>${esc(baseName(p))}</option>`).join('')}</select></label>` : `<div class="muted small">投影文件：${esc(baseName(g.mmprojs[0]))}</div>`) : '';
  const serverBox = cmdBox('cmdServer', 'llama-server 命令：');
  const cliBox = cmdBox('cmdCli', 'llama-cli 命令（在终端中直接对话）：');
  const body = openDialog((mode === 'start' ? '启动模型：' : '运行命令：') + g.name, `${warn}${variantHTML}${modeHTML}${visionHTML}
    <label style="margin-top:8px">本次附加参数<textarea id="rExtra" rows="2" placeholder="例如：--seed 42">${esc(def.extra || '')}</textarea></label>
    <div id="rWarn" class="hot small"></div>
    ${state.os === 'windows' ? '<div class="muted small">命令为 PowerShell 格式；在 cmd 中使用时去掉开头的「& 」。</div>' : ''}
    ${mode === 'cli' ? cliBox + serverBox : serverBox + cliBox}`,
    mode === 'start' ? [
      { text: '取消', onClick: () => true },
      { text: '在终端运行 llama-cli', onClick: () => runInTerminal() },
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
    ] : [{ text: '关闭', onClick: () => true }, { text: '在终端运行 llama-cli', cls: 'primary', onClick: () => runInTerminal() }]);
  let runMode = def.gpuOk ? def.mode : '';
  const runInTerminal = async () => {
    const r = await api(`/api/groups/${g.id}/terminal`, { method: 'POST', body: req() });
    toast(`已在 ${r.terminal} 中打开 llama-cli`);
  };
  const req = () => ({
    variant: $('#rVariant', body)?.value || sel,
    mode: runMode,
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
  body.onclick = e => {
    const b = e.target.closest('[data-copy]');
    if (b) copyText($('#' + b.dataset.copy, body).textContent);
    const m = e.target.closest('[data-mode]');
    if (m) {
      runMode = m.dataset.mode;
      body.querySelectorAll('[data-mode]').forEach(x => x.classList.toggle('active', x === m));
      preview();
    }
  };
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

// ---------- 引导 ----------
const OS_NAME = { windows: 'Windows', linux: 'Linux', darwin: 'macOS', android: 'Android' };
async function saveConfigPatch(patch) {
  await api('/api/config', { method: 'PUT', body: { ...state.config, ...patch } });
  await loadConfig();
}
async function loadGuide(refresh) {
  const el = $('#guideBody');
  if (refresh || !el.dataset.loaded) el.innerHTML = '<div class="muted">正在检测系统并获取 llama.cpp 最新发布信息…</div>';
  let gi;
  try { gi = await api('/api/guide' + (refresh ? '?refresh=1' : '')); } catch (e) { el.textContent = e.message; return; }
  el.dataset.loaded = '1';
  state.guide = gi;
  const gpuRecs = gi.recs.filter(r => r.kind === 'gpu');
  const cpuRecs = gi.recs.filter(r => r.kind === 'cpu');
  const recHTML = r => `<div class="rec${r.alt ? ' alt' : ''}"><strong>${esc(r.title)}</strong>${r.alt ? ' <span class="tag">备选</span>' : ' <span class="tag running">推荐</span>'}
    ${r.note ? `<div class="muted small">${esc(r.note)}</div>` : ''}
    <div class="dl">${r.assets.map((a, i) => `<a class="btn${i ? ' sec' : ''}" href="${esc(a.url)}" target="_blank" rel="noopener">⬇ ${esc(a.name)} <span style="opacity:.75">${fmtBytes(a.size)}</span></a>`).join('')}</div></div>`;
  const sv = gi.servers;
  const svLine = (label, si, conf) => !conf && !si?.path ? `<div>${label}：<span class="muted">未配置</span></div>` :
    si?.path ? `<div>${label}：<span class="ok-text">✔</span> ${esc(si.path)} <span class="muted">（${esc(si.version || '版本未知')}${si.devices?.length ? '，' + si.devices.map(esc).join('；') : '，仅 CPU'}）</span></div>`
      : `<div>${label}：<span class="hot">✘ ${esc(si?.error || '未找到')}</span></div>`;
  const step3Done = !!(sv.gpu?.path || sv.cpu?.path);
  const step4Done = gi.modelCount > 0;
  const mac = gi.os === 'darwin';
  el.innerHTML = `
  <div class="step done"><div class="num">1</div><div class="body"><h3>系统检测</h3>
    <div>系统：<b>${esc(OS_NAME[gi.os] || gi.os)}</b> · 架构：<b>${esc(gi.arch)}</b></div>
    <div style="margin-top:4px">显卡：${gi.gpus.length ? gi.gpus.map(g => `<span class="chip">${esc(g.vendor)} · ${esc(g.name)}</span>`).join('') : '<span class="muted">未检测到独立显卡</span>'}
      ${gi.cuda ? `<span class="chip">驱动支持 CUDA ${esc(gi.cuda)}</span>` : ''}</div>
    <div style="margin-top:6px">${mac ? 'macOS 版同一程序同时支持 Metal GPU 与 CPU 运行。' : gpuRecs.length
      ? '可同时使用 <b>GPU</b> 与 <b>CPU</b> 方式运行：建议分别下载 GPU 版与 CPU 版，GPU 版用于日常加速，CPU 版用于显存不足或对比测试。'
      : '未发现可用于加速的显卡，建议仅下载 <b>CPU 版</b>。'}</div>
  </div></div>

  <div class="step"><div class="num">2</div><div class="body"><h3>下载 llama.cpp ${gi.release ? `<a href="${esc(gi.release.url)}" target="_blank" rel="noopener">${esc(gi.release.tag)}</a> <span class="muted small">发布于 ${new Date(gi.release.published).toLocaleString()}</span>` : ''}</h3>
    ${gi.error ? `<div class="hot small">${esc(gi.error)}</div><div class="small">也可直接打开 <a href="https://github.com/ggml-org/llama.cpp/releases" target="_blank" rel="noopener">GitHub Releases</a> 手动下载。</div>` : `
      ${gpuRecs.length ? `<div class="small muted" style="margin-top:4px">GPU 版</div>${gpuRecs.map(recHTML).join('')}` : ''}
      ${cpuRecs.length ? `<div class="small muted" style="margin-top:8px">CPU 版</div>${cpuRecs.map(recHTML).join('')}` : ''}
      ${!gi.recs.length ? '<div class="muted">未找到适合当前系统的预编译包，请参考 llama.cpp 文档自行编译。</div>' : ''}`}
    <div class="muted small" style="margin-top:6px">下载后分别解压到独立目录，例如 <code>${gi.os === 'windows' ? 'D:\\llama.cpp\\gpu' : '~/llama.cpp/gpu'}</code> 与 <code>${gi.os === 'windows' ? 'D:\\llama.cpp\\cpu' : '~/llama.cpp/cpu'}</code>。${gi.os === 'android' ? '在 Termux 中解压并 <code>chmod +x llama-*</code>。' : ''}
      <button type="button" id="gRefresh" style="margin-left:6px">重新获取</button></div>
  </div></div>

  <div class="step${step3Done ? ' done' : ''}"><div class="num">3</div><div class="body"><h3>配置程序路径</h3>
    ${svLine(mac ? '程序' : 'GPU 版', sv.gpu, state.config.serverPathGpu)}
    ${mac ? '' : svLine('CPU 版', sv.cpu, state.config.serverPathCpu)}
    <div class="dl" style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
      <button type="button" class="primary" data-gpick="serverPathGpu">选择${mac ? '' : ' GPU 版'}目录…</button>
      ${mac ? '' : '<button type="button" data-gpick="serverPathCpu">选择 CPU 版目录…</button>'}
    </div>
  </div></div>

  <div class="step${step4Done ? ' done' : ''}"><div class="num">4</div><div class="body"><h3>添加模型目录</h3>
    <div>已添加 ${gi.modelDirs} 个目录，共找到 <b>${gi.modelCount}</b> 个模型文件。</div>
    <div class="muted small">可在 <a href="https://www.modelscope.cn/models?name=GGUF" target="_blank" rel="noopener">ModelScope</a> 或 <a href="https://huggingface.co/models?library=gguf" target="_blank" rel="noopener">HuggingFace</a> 下载 GGUF 格式模型，放到 <code>目录/模型.gguf</code> 或 <code>目录/子目录/模型.gguf</code>。</div>
    <button type="button" id="gAddDir" style="margin-top:8px">添加模型目录…</button>
  </div></div>

  <div class="step${step3Done && step4Done ? ' done' : ''}"><div class="num">5</div><div class="body"><h3>开始使用</h3>
    <button type="button" class="primary" id="gGo" ${step3Done && step4Done ? '' : 'disabled'}>去启动模型</button>
  </div></div>`;
  $('#gRefresh').onclick = () => loadGuide(true);
  $('#gGo').onclick = () => switchTab('models');
  el.querySelectorAll('[data-gpick]').forEach(b => b.onclick = async () => {
    const k = b.dataset.gpick;
    const p = await pickPath('选择 llama-server 所在目录', state.config[k] || '', 'server');
    if (!p) return;
    try {
      await saveConfigPatch({ [k]: p });
      await probeServer(true);
      loadGuide();
    } catch (e) { toast(e.message, true); }
  });
  $('#gAddDir').onclick = async () => {
    const p = await pickPath('添加模型目录', '', 'dir');
    if (!p) return;
    try {
      const dirs = state.config.modelDirs || [];
      if (!dirs.includes(p)) await saveConfigPatch({ modelDirs: [...dirs, p] });
      await loadModels();
      loadGuide();
    } catch (e) { toast(e.message, true); }
  };
}
document.querySelector('.tabs [data-tab=guide]').addEventListener('click', () => loadGuide());
document.querySelector('.tabs [data-tab=system]').addEventListener('click', () => refreshStats());

// ---------- 初始化 ----------
(async function init() {
  const t = lsGet('tab'); if (t) switchTab(t);
  try { await loadConfig(); } catch (e) { toast('加载配置失败：' + e.message, true); }
  await probeServer();
  renderFields($('#globalParams'), state.config?.global || {}, state.defaults, false);
  loadHost();
  // 首次使用（找不到 llama-server 或未添加模型目录）时打开引导页
  const noServer = !state.server.gpu?.path && !state.server.cpu?.path;
  if (noServer || !(state.config?.modelDirs || []).length) switchTab('guide');
  if ($('#tab-guide').classList.contains('active')) loadGuide();
  await loadInstances();
  await loadModels();
  refreshStats();
  setInterval(refreshStats, 2000);
  setInterval(loadInstances, 3000);
})();
