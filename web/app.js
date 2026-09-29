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
  { sec: 'GPU 与显存', note: '仅 GPU 运行方式生效；显存不够或启动报 out of memory 时，优先调整这一组' },
  { k: 'mode', label: '运行方式', type: 'select', opts: [['gpu', 'GPU'], ['cpu', 'CPU']], gpu: true,
    hint: 'GPU：用显卡加速（使用 GPU 版 llama-server），速度快；CPU：只用处理器和内存，不占显存但慢很多。未检测到可用 GPU 时自动使用 CPU。启动时也可临时切换。' },
  { k: 'nGpuLayers', label: 'GPU 层数 -ngl', type: 'text',
    hint: '放进显存的模型层数，越多越快、越占显存，放不下的层由 CPU 计算。auto：配合 -fit 按显存自动计算，显存紧张或多模型混跑时推荐；999 / all：全部放入显存，最快且行为确定可复现，显存充裕（如 24GB 跑 7B/8B）时推荐；也可填具体数字。' },
  { k: 'fit', label: '自动适配显存 -fit', type: 'select', opts: ONOFF,
    hint: '开启后按显卡剩余显存自动调整未手动设置的参数（如 GPU 层数、上下文长度），尽量避免显存溢出。新版默认 on；想完全手动控制时设为 off。' },
  { k: 'fitTarget', label: '每卡预留显存 -fitt (MiB)', type: 'text',
    hint: '-fit 计算时给每块显卡留出的空闲显存，默认 1024。同时还要跑游戏、其它模型或桌面特效时调大；多卡可分别设置，如 1024,2048。' },
  { k: 'fitCtx', label: '-fit 最小上下文 -fitc', type: 'int',
    hint: '-fit 自动缩小上下文时不会低于这个值（默认 4096）；若仍放不下，会改为减少 GPU 层数。' },
  { k: 'device', label: '计算设备 --device', type: 'text',
    hint: '指定用哪些设备计算，名称见设置页“检测”结果，如 CUDA0、Vulkan0，多个用逗号分隔。留空则自动使用全部可用设备。' },
  { k: 'splitMode', label: '多卡切分方式 -sm', type: 'select', opts: [['layer', 'layer（按层）'], ['row', 'row（按行）'], ['tensor', 'tensor'], ['none', 'none（只用主卡）']],
    hint: '多块显卡时如何分配模型：layer 按层分到各卡（默认，兼容性最好）；row 把权重按行切分，部分场景更快但依赖卡间带宽；none 只用主 GPU。单卡无需设置。' },
  { k: 'tensorSplit', label: '多卡分配比例 -ts', type: 'text',
    hint: '多卡时每张卡承担的比例，如 3,1 表示第一张卡放 3/4、第二张放 1/4。通常按各卡显存大小的比例填写。' },
  { k: 'mainGpu', label: '主 GPU 序号 -mg', type: 'int',
    hint: '切分方式为 none 时使用的显卡，或 row 模式下存放中间结果与 KV 缓存的显卡，从 0 开始，默认 0。' },
  { k: 'cpuMoe', label: 'MoE 专家全部放 CPU -cmoe', type: 'bool',
    hint: '仅对 MoE（混合专家）模型有效，如 Qwen3-30B-A3B、gpt-oss：把所有专家权重放在内存由 CPU 计算，注意力等其余部分仍在 GPU。适合显存小、内存大的机器跑大 MoE 模型。' },
  { k: 'nCpuMoe', label: '前 N 层 MoE 放 CPU -ncmoe', type: 'int',
    hint: '只把前 N 层的专家权重放到 CPU，比 -cmoe 更精细：从小往大逐步增加 N，直到显存刚好放得下，速度优于全部放 CPU。' },
  { k: 'kvOffload', label: 'KV 缓存放显存', type: 'bool',
    hint: 'KV 缓存是模型对上下文的“记忆”，上下文越长越大。默认放显存；关闭（传 -nkvo）改放内存，可省下不少显存，但生成速度明显下降。' },

  { sec: '上下文与性能', note: '影响能处理多长的对话、内存/显存占用和处理速度' },
  { k: 'ctxSize', label: '上下文长度 -c', type: 'int',
    hint: '模型一次能“记住”的最大 token 数（提示词 + 历史对话 + 回复）。越大越占内存/显存。0 表示用模型训练时的最大值（可能 128K 以上，容易爆内存）；日常对话 4096~16384，长文档/代码 32768 左右。' },
  { k: 'threads', label: 'CPU 线程数 -t', type: 'int',
    hint: '生成回复时使用的 CPU 线程数，留空自动。纯 CPU 运行时设为物理核心数通常最快，超过物理核心数反而可能变慢。' },
  { k: 'threadsBatch', label: '提示处理线程数 -tb', type: 'int',
    hint: '处理输入提示词（预填充）时的线程数，默认与 -t 相同。可设为逻辑核心数以加快长提示词的处理。' },
  { k: 'parallel', label: '并发槽数 -np', type: 'int',
    hint: '服务端能同时处理的请求数。多个请求会分摊上下文，如 -c 8192 -np 4 时每个请求约 2048。多人或多个客户端同时使用时调大；默认自动。' },
  { k: 'batchSize', label: '逻辑批大小 -b', type: 'int',
    hint: '一次最多提交处理的提示词 token 数，默认 2048。调大可加快长提示词处理，但占用更多显存/内存。' },
  { k: 'ubatchSize', label: '物理批大小 -ub', type: 'int',
    hint: '实际每次送入计算的 token 数，默认 512，不能大于 -b。显存不足时可减小到 256 或 128；显存充裕时调大可加快提示处理。' },
  { k: 'flashAttn', label: 'Flash Attention -fa', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off'], ['none', '不传递（旧版）']],
    hint: '一种更高效的注意力计算方式：降低显存占用、加快长上下文处理，也是量化 V 缓存的前提。auto 自动判断（推荐）。旧版 llama.cpp 不认识 on/off/auto 时选“不传递”。' },
  { k: 'cacheTypeK', label: 'K 缓存精度 -ctk', type: 'select', opts: CACHE_TYPES.map(v => [v, v]),
    hint: 'KV 缓存中 K 部分的数据类型。默认 f16；q8_0 约省一半 KV 占用且几乎无损；q4_0 更省但回答质量会下降。' },
  { k: 'cacheTypeV', label: 'V 缓存精度 -ctv', type: 'select', opts: CACHE_TYPES.map(v => [v, v]),
    hint: 'KV 缓存中 V 部分的数据类型，量化（如 q8_0）需要开启 Flash Attention。长上下文时与 K 一起设为 q8_0 可显著节省显存。' },
  { k: 'cacheRam', label: '提示缓存上限 -cram (MiB)', type: 'int',
    hint: '在内存中缓存处理过的提示词，多轮对话或重复前缀时可跳过重复计算、加快响应。默认 8192，-1 不限制，0 关闭。' },
  { k: 'mlock', label: '锁定内存 --mlock', type: 'bool',
    hint: '把模型锁在物理内存中，防止被系统换到硬盘导致忽快忽慢。需要内存足够大；Linux 下可能需要额外权限。' },
  { k: 'noMmap', label: '禁用内存映射 --no-mmap', type: 'bool',
    hint: '默认模型按需从文件映射到内存，加载快且可与系统共享缓存。开启后改为一次性完整读入内存：加载更慢、占用更多，但在网络盘等环境下更稳定。一般保持关闭。' },
  { k: 'contextShift', label: '上下文平移 --context-shift', type: 'bool',
    hint: '上下文写满时自动丢弃最早的一部分内容继续生成，而不是报错停止。开启后模型会“忘记”对话开头的内容。' },

  { sec: '采样', note: '控制回复的随机性与风格；建议优先使用“模型介绍”中提取的推荐值' },
  { k: 'temp', label: '温度 --temp', type: 'float',
    hint: '越高回复越随机、越有创意，越低越稳定、越确定。代码/数学建议 0.2~0.6，聊天/写作 0.7~1.0。默认 0.8。' },
  { k: 'topK', label: 'Top-K --top-k', type: 'int',
    hint: '每一步只从概率最高的 K 个候选词中挑选，越小越保守。0 表示不限制，默认 40。' },
  { k: 'topP', label: 'Top-P --top-p', type: 'float',
    hint: '只在累计概率达到 P 的候选词里挑选（核采样），越小越保守。1.0 表示关闭，默认 0.95。' },
  { k: 'minP', label: 'Min-P --min-p', type: 'float',
    hint: '丢弃概率低于“最高概率 × P”的候选词，比 Top-P 更能适应不同情形。0 表示关闭，默认 0.05。' },
  { k: 'repeatPenalty', label: '重复惩罚 --repeat-penalty', type: 'float',
    hint: '降低最近出现过的词再次出现的概率，大于 1 才生效（常用 1.05~1.15）。过大会让表达变得别扭。默认 1.0（关闭）。' },
  { k: 'presencePenalty', label: '存在惩罚 --presence-penalty', type: 'float',
    hint: '对出现过的词统一施加惩罚，鼓励谈论新内容，可缓解思考类模型陷入循环重复（Qwen3 推荐 0~1.5）。默认 0（关闭）。' },
  { k: 'seed', label: '随机种子 -s', type: 'int',
    hint: '固定种子后，相同输入和参数会得到相同输出，便于对比调参；-1 表示每次随机。' },
  { k: 'nPredict', label: '最大回复长度 -n', type: 'int',
    hint: '单次回复最多生成的 token 数，-1 表示不限制（直到模型自行结束或上下文用满）。客户端请求中的 max_tokens 会覆盖此值。' },

  { sec: '模板与推理', note: '聊天格式、工具调用和思考模式' },
  { k: 'jinja', label: 'Jinja 聊天模板 --jinja', type: 'bool',
    hint: '使用模型文件内置的 Jinja 聊天模板，工具调用（function calling）和思考模式都依赖它。新版默认开启，一般保持开启。' },
  { k: 'reasoning', label: '思考模式 -rea', type: 'select', opts: [['auto', 'auto'], ['on', 'on'], ['off', 'off']],
    hint: '针对会“先思考再回答”的模型（如 Qwen3、DeepSeek-R1）：auto 按模板自动识别；on 强制开启；off 关闭思考，回复更快但复杂问题效果可能变差。' },
  { k: 'reasoningBudget', label: '思考预算 --reasoning-budget', type: 'int',
    hint: '思考阶段最多使用的 token 数：-1 不限制；0 跳过思考直接回答；正数表示达到后强制结束思考并给出答案。' },
  { k: 'chatTemplateKwargs', label: '模板参数 --chat-template-kwargs', type: 'text',
    hint: '传给聊天模板的额外参数（JSON 格式），例如 {"enable_thinking":false} 可关闭 Qwen3 的思考。' },

  { sec: '服务', note: '仅 llama-server 生效，llama-cli 会忽略' },
  { k: 'host', label: '监听地址 --host', type: 'text',
    hint: '127.0.0.1 仅本机可访问；0.0.0.0 允许局域网内其它设备访问（建议同时设置 API Key）。' },
  { k: 'port', label: '端口 --port', type: 'int',
    hint: '服务端口。0 表示从设置中的“自动分配端口起始值”开始寻找空闲端口；同时运行多个模型时建议保持自动。' },
  { k: 'apiKey', label: 'API Key --api-key', type: 'text',
    hint: '设置后，客户端需在请求头携带 Authorization: Bearer <密钥> 才能调用。服务对局域网或外网开放时务必设置。' },
  { k: 'alias', label: '模型别名 -a', type: 'text', modelOnly: true,
    hint: '接口 /v1/models 返回的模型名称，客户端按这个名字调用。留空使用文件名。' },
  { k: 'webui', label: '内置网页聊天 Web UI', type: 'bool',
    hint: 'llama-server 自带的网页聊天界面，浏览器打开 http://地址:端口 即可使用。只作 API 服务时可关闭（传 --no-webui）。' },
  { k: 'metrics', label: '监控指标 --metrics', type: 'bool',
    hint: '开启 /metrics 接口，输出 Prometheus 格式的吞吐、排队等指标，用于接入监控系统。' },

  { sec: '其它' },
  { k: 'extraArgs', label: '附加参数（原样追加到命令行）', type: 'textarea', wide: true,
    hint: '上面没有列出的参数写在这里，会原样追加到命令末尾，例如 --rope-scaling yarn、--swa-full、--lora 路径。完整列表可运行 llama-server --help 查看。' },
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
// 曲线数据来自服务端采样历史（约 3 分钟），打开页面即可看到完整曲线。
let hist = [];
function histSeries(key) {
  const [k, i] = key.split(':');
  return hist.map(h => ({ t: h.t, v: i === undefined ? h[k] : (h[k] || [])[+i] ?? 0 }));
}
function agoText(ms) {
  const sec = Math.round(ms / 1000);
  return sec < 60 ? `${sec} 秒前` : `${Math.round(sec / 60)} 分钟前`;
}
function sparkSVG(key) {
  const d = histSeries(key);
  const W = 300, H = 48;
  const grid = [0.25, 0.5, 0.75].map(y => `<line x1="0" x2="${W}" y1="${H * y}" y2="${H * y}"/>`).join('');
  if (d.length < 2) {
    return `<div class="spark-wrap"><svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><rect class="bg" width="${W}" height="${H}"/><g class="grid">${grid}</g></svg>
      <div class="spark-cap"><span>采集中…</span><span>现在</span></div></div>`;
  }
  const t0 = d[0].t, t1 = d[d.length - 1].t, span = Math.max(t1 - t0, 1);
  const pts = d.map(p => [((p.t - t0) / span * W).toFixed(1), (H - Math.max(0, Math.min(100, p.v)) / 100 * (H - 2) - 1).toFixed(1)]);
  const line = 'M' + pts.map(p => p.join(',')).join('L');
  const peak = Math.max(...d.map(p => p.v)), avg = d.reduce((a, p) => a + p.v, 0) / d.length;
  return `<div class="spark-wrap"><svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <rect class="bg" width="${W}" height="${H}"/><g class="grid">${grid}</g>
      <path class="area" d="${line}L${W},${H}L0,${H}Z"/><path class="line" d="${line}" vector-effect="non-scaling-stroke"/></svg>
    <div class="spark-cap"><span>${agoText(t1 - t0)}</span><span>峰值 ${peak.toFixed(0)}% · 平均 ${avg.toFixed(0)}%</span><span>现在</span></div></div>`;
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
  // 顶部精简状态：只显示最关键的占用率，便于在其它页面留意是否爆内存
  const vram = (s.gpus || []).filter(g => g.memTotal).map(g => g.memUsed / g.memTotal * 100);
  const pctB = p => `<b class="${p >= 90 ? 'hot' : ''}">${p.toFixed(0)}%</b>`;
  $('#miniStat').innerHTML = `CPU ${pctB(s.cpuPercent)} · 内存 ${pctB(s.memPercent)}` + (vram.length ? ` · 显存 ${vram.map(pctB).join('/')}` : '');
  updateInstanceMem();
  if (!$('#tab-system').classList.contains('active')) return;
  try { hist = await api('/api/stats/history'); } catch (e) { }

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
      ${metricHTML('占用率', g.util, g.util.toFixed(0) + '%', 'gpu:' + i)}
      ${g.memTotal ? metricHTML('显存', mp, `${fmtBytes(g.memUsed)} / ${fmtBytes(g.memTotal)}（${mp.toFixed(0)}%）`, 'vram:' + i) : ''}
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
    `<p class="muted small">参数对该组所有版本生效。留空的参数继承全局配置，优先级：模型参数 → 全局参数 → 内置默认。</p><div class="grid params-grid" id="mParams"></div>`, [
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
