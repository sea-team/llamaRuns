# llamaRuns 开发归档

llama.cpp（llama-server / llama-cli）的 Web 管理面板。Go 单文件程序，前端内嵌，无 cgo，支持 Windows / Linux / macOS / Android(Termux)。本文档记录架构、约定与需求演进，修改功能前先读这里；用户使用说明见 README.md，原始需求见 des.txt。

## 构建与测试

```bash
go vet ./... && go test ./...          # 单元测试在 main_test.go
node --check web/app.js                # 前端语法检查（无构建步骤）
go build -o llamaRuns .                # 本机构建
./build.sh                             # 交叉编译全部平台到 dist/llamaRuns-<os>-<arch>[.exe]
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o dist/llamaRuns-windows-amd64.exe .
```

- 必须保持 `CGO_ENABLED=0` 可编译（6 个目标：windows/amd64、linux/amd64、linux/arm64、darwin/amd64、darwin/arm64、android/arm64）。改动平台相关代码后用 `GOOS=windows go vet ./...`、`GOOS=darwin go vet ./...` 检查。
- 界面验证：用假的 llama-server 脚本（响应 `--version`、`--list-devices`，并在 `--port` 上提供 `/health`）配合 agent-browser 截图。

### 测试安全约定（重要）

- 用户在 Windows 上直接运行 `dist/` 下的 exe，监听默认 `127.0.0.1:8686`，配置为 `dist/config.json`。WSL 与 Windows 共享 localhost 端口。
- 测试一律用独立端口（如 8699 / 18686）和临时目录的 `-config`，先确认 8686 是否被占用；**绝不能向用户实例发 PUT 请求**（曾经覆盖过用户配置）。
- 用户正在运行的 exe 会被锁定，不能覆盖；新版输出为另一个文件名并告知用户替换。
- 结束测试进程只按 PID 或用 `timeout`；不要用 `taskkill /IM`（会误杀用户进程），也不要用 `pkill -f <含命令文本的模式>`（会匹配执行它的 shell 自身）。

## 文件职责

| 文件 | 职责 |
|---|---|
| `main.go` | 入口：`-addr`（默认 127.0.0.1:8686）、`-config`（默认 exe 同目录 config.json）；先监听再打印启动信息；收到信号时停止所有实例 |
| `config.go` | `Params`（全部启动参数，指针=未设置）、`Config`、`LastRun`、`FlexStr`；`merge` 用 JSON 往返实现覆盖；`defaultParams()` 内置默认；`ConfigStore` 原子写文件 |
| `models.go` | 扫描模型目录（两级）、分片合并、mmproj 识别；`BaseName`/`Quant` 量化后缀识别；`GroupModels` 分组 |
| `process.go` | 日志环形缓冲 + 订阅；`argSpecs` 表驱动生成命令行；`BuildArgs`/`RunOpt`；`ServerBinary`/`CLIBinary`；实例管理（启动、就绪探测 `/health`、停止、端口分配）；`ProbeServer`（`--version`、`--list-devices`）；跨平台引号 `quoteCmd` |
| `proc_windows.go` / `proc_other.go` | `prepareCmd`：Windows 隐藏子进程控制台窗口 |
| `terminal_windows.go` / `terminal_other.go` | `OpenTerminal`：Windows 优先 `wt.exe new-tab`，否则 `CREATE_NEW_CONSOLE` 的 PowerShell；macOS osascript 调用 Terminal；Linux 依次尝试常见终端；WSL 借助 wt.exe；Android 不支持 |
| `sysinfo.go` | `SysMonitor` 后台约每 2 秒采样：CPU/内存/交换/温度、GPU（nvidia-smi → rocm-smi → amdgpu sysfs）、实例进程 RSS/CPU；保留 90 个历史点；`hostInfo` 静态主机信息；`nvidiaCUDA` 解析驱动支持的 CUDA 版本 |
| `modelinfo.go` | 模型介绍：按关键字在 HF / ModelScope 搜索（优先 GGUF 过滤），评分选仓库，拉 README；推荐参数提取（README 正则 → base_model README → generation_config.json）；HTTP 客户端支持代理；结果缓存在 `cache/info/<组ID>_<源>.json` |
| `guide.go` | 引导：`detectGPUs` 识别显卡厂商；拉取 GitHub 最新含二进制包的 release（缓存 30 分钟，走 HF 代理设置）；`recommendAssets` 按系统/架构/厂商/CUDA 版本推荐下载包 |
| `server.go` | 路由与处理；`plan` 汇总一次启动所需的一切（版本、运行方式、视觉、程序路径、参数、上次选择）；目录浏览 `listDir`；日志 SSE |
| `web/index.html` `web/app.js` `web/style.css` | 原生 JS 单页，`//go:embed web` 内嵌；Markdown 渲染按需从 jsdelivr 加载 marked + DOMPurify（加载失败退回纯文本） |
| `main_test.go` | 量化识别、分组、推荐参数提取、参数切分、参数优先级、命令生成、下载包推荐、CUDA 版本选择 |

## 核心概念

### 模型组
- 扫描规则：`目录/*.gguf` 与 `目录/子目录/*.gguf`；文件名含 `mmproj` 的是视觉投影文件，不作为模型，挂到同目录模型上；`-00001-of-0000N.gguf` 分片合并为一项（大小累加）。
- 分组：子目录下的所有模型为一组（组名=子目录名）；根目录下按 `BaseName`（去量化/分片后缀）同名归组。单个模型的组显示完整文件名。
- `Group.Key`：子目录绝对路径，或 `root|basename小写`；`Group.ID = sha1(Key)[:12]`。**参数、上次启动记录、实例 ID、介绍缓存都以组为单位**。

### 参数
- 优先级：模型组参数 `Config.Models[groupKey]` → 全局参数 `Config.Global` → `defaultParams()`（mode=gpu、ngl=auto、ctx=4096、host=127.0.0.1、port=0 自动、fa=auto、jinja=true）。
- **新增一个参数的步骤**：① `config.go` 的 `Params` 加字段（指针 + json 名）；② `process.go` 的 `argSpecs` 加映射（`argVal`/`argTrue`/`argFalse`，`gpu:true` 表示 CPU 方式跳过，`server:true` 表示 llama-cli 跳过）；③ `web/app.js` 的 `FIELDS` 加表单项（放到对应 `sec` 分区，写清 `hint` 作用说明）；④ 如需在 `main_test.go` 的 `TestBuildArgs` 中覆盖。
- `BuildArgs` 通过 JSON（`UseNumber`，避免大整数变科学计数法）遍历 `argSpecs`；CPU 方式且用的是 GPU 版程序时追加 `-ngl 0 --device none`（仅当程序支持 `--device`）；视觉时追加 `--mmproj`；服务端追加 `-a/--host/--port`；最后追加参数中的附加参数与本次附加参数（`SplitArgs` 支持引号）。
- `-ngl` 为 `FlexStr`（兼容旧配置中的数字）；`-fa` 选 `none` 表示不传递（兼容旧版 llama.cpp）。

### 程序选择（CPU/GPU 版）
- `serverPathGpu`（留空则用 PATH 中的 llama-server）与 `serverPathCpu`，可填文件或目录（`ResolveServerPath` 会在目录中找 llama-server[.exe]）。旧字段 `llamaServerPath` 读取时迁移到 GPU 路径。
- 实际运行方式 = 本次选择 / 上次选择 / 参数 mode；GPU 版程序 `--list-devices` 无设备时强制 CPU（`gpuOk=false`）。
- CPU 方式优先用 CPU 版；未配置 CPU 版时用 GPU 版并禁用 GPU。llama-cli 取与所用 llama-server 同目录的文件。

### 启动流程
`POST /api/groups/{id}/start` → `plan()` → `Manager.Start()`：非多开先停其它实例 → 分配端口（`basePort` 起找空闲）→ 启动进程，stdout/stderr 按 `\r`/`\n` 切行写入环形缓冲 → 每秒探测 `/health`，200 即 running → 保存 `LastRun`（版本、运行方式、视觉、mmproj、附加参数）。停止：Windows 直接 Kill，其它系统先 SIGINT，8 秒后 Kill。

### 日志
服务端每实例只保留 `maxLogLines` 行；SSE 先发历史，再每 200ms 批量推送；前端 DOM 同样只保留 N 行。

### 模型介绍与推荐参数
- 关键字：组内第一个版本的 `BaseName` → 子目录名（去 -GGUF）→ 去掉最后一段。先带 GGUF 过滤搜索（HF `filter=gguf`，ModelScope `Criterion libraries contains gguf`），全部无结果再放宽。
- 评分：名称等于关键字(+GGUF) 100、包含 50、组织名是关键字前缀 20、含 gguf 10、下载量对数。
- 推荐参数键与 `Params` 的 JSON 名一致（temp/topP/topK/minP/presencePenalty/repeatPenalty/ctxSize），前端“应用”即合并进组参数。

### 引导推荐
- 厂商检测：nvidia-smi；Windows 用 `Win32_VideoController`；Linux/Android 用 `/sys/class/drm/card*/device/vendor`（0x10de/0x1002/0x8086）与 `/sys/class/kgsl`（Adreno）；Apple Silicon 视为 Metal。
- CUDA：选不超过驱动支持版本的最高包；都超过时选同主版本最低的并提示兼容性；Windows/Linux CUDA 包同时推荐同版本 `cudart-*`。
- 资产名形如 `llama-b<N>-bin-win-cuda-12.4-x64.zip`、`llama-b<N>-bin-ubuntu-x64.tar.gz`、`cudart-llama-bin-win-cuda-12.4-x64.zip`；上游改名时需同步 `recommendAssets` 的正则与 `TestRecommendAssets`。

## HTTP 接口

| 方法 路径 | 说明 |
|---|---|
| `GET/PUT /api/config` | 读取（含 defaults、os）/ 保存设置（保留 models、lastRun） |
| `GET /api/server?refresh=1` | GPU/CPU 版程序探测结果与 `gpuOk` |
| `GET /api/fs?path=&files=1` | 目录浏览（dirs、files、roots：Windows 盘符 / `/` / 用户目录） |
| `GET /api/groups` | 模型组列表（含 hasParams、lastVariant） |
| `GET/PUT /api/groups/{id}/params` | 组参数（GET 同时返回继承值 inherit） |
| `GET /api/groups/{id}/run` | 启动框默认值（上次版本、运行方式、视觉、附加参数、gpuOk） |
| `POST /api/groups/{id}/command` | 生成 llama-server 与 llama-cli 命令（body：variant/mode/vision/mmproj/extra） |
| `POST /api/groups/{id}/start` | 启动 |
| `POST /api/groups/{id}/terminal` | 在新终端运行 llama-cli |
| `GET /api/groups/{id}/info?source=hf\|ms&repo=&refresh=1` | 模型介绍与推荐参数 |
| `GET /api/instances`、`POST /api/instances/{id}/stop`、`DELETE /api/instances/{id}` | 实例列表 / 停止 / 移除 |
| `GET /api/instances/{id}/logs` | 日志 SSE |
| `GET /api/stats`、`GET /api/stats/history`、`GET /api/host` | 实时状态 / 采样历史 / 主机信息 |
| `GET /api/guide?refresh=1` | 引导信息（系统、显卡、CUDA、release、推荐包、配置状态） |

## 前端结构（web/app.js）

按注释分节：工具函数与 `api()` → `FIELDS` 参数定义（`sec` 分区 + 字段 `hint` 说明）与 `renderFields/collectFields` → 通用对话框 `openDialog` → 目录/文件选择器 `pickPath`（面包屑、位置/最近使用、筛选、双击确定）→ 标签页 → 系统状态（曲线数据来自服务端历史，按实际时间铺满）→ 模型组列表（`<details>` 折叠，展开状态存于 `state.openGroups`）→ 启动/命令对话框 `runDialog` → 参数对话框 → 模型介绍 → 运行实例与日志 SSE → 设置 → 引导 `loadGuide` → 初始化（未找到程序或无模型目录时自动打开引导页）。

轮询：状态 2 秒、实例 3 秒；本地存储仅用于记住标签页、介绍来源、最近目录。

## 配置文件（config.json）

`serverPathGpu`、`serverPathCpu`、`modelDirs[]`、`allowMulti`、`maxLogLines`（默认 1000）、`basePort`（默认 8080）、`hfEndpoint`、`proxy`（http/https/socks5）、`proxyAll`（false=仅 HF/GitHub 走代理）、`global`（Params）、`models`（组 Key → Params）、`lastRun`（组 Key → LastRun）。

## 约定

- 界面文字、注释、错误信息使用中文；代码风格与现有文件一致，注释密度适中。
- 不引入前端构建工具与框架；不引入 cgo 依赖。
- 面板可启动任意进程，默认只监听 127.0.0.1；渲染外部 README 必须经过 DOMPurify。
- 引导/介绍依赖外网（GitHub、HF、ModelScope），失败时要给出可操作的提示（配置代理 / 手动链接）。

## 需求归档（按轮次）

1. **初版**：Web 面板启动 llama-server（路径可配，默认 PATH）；常用参数表单 + 启动时附加参数；多模型目录自动扫描；从 ModelScope/HF 获取模型介绍；输出同步显示且限制行数；默认单开、可多开；参数优先级 模型→全局；CPU/GPU 运行，无 GPU 时屏蔽；显示 CPU/GPU/内存占用与温度；多平台。
2. **路径与分组**：目录选择方式配置路径；区分 CPU/GPU 版 llama-server 路径；生成 llama-cli 完整命令；从模型页面提取推荐参数；显示 nvidia-smi / AMD 显卡信息；视觉模型（mmproj）启动时可选；同一模型多版本可选且默认上次；HF 代理；子目录多规格模型折叠展示。
3. **引导与体验**：模型介绍只找 GGUF 版本；有 GPU 时启动可选运行方式；llama-cli 可直接在终端运行（Windows 优先 Windows Terminal）；美化文件选择器；显卡信息只提取主要部分，与顶部状态合并为独立的系统状态页；新增引导页按系统/显卡推荐下载 llama.cpp release；`-ngl` 参考配置（显存充裕 999，紧张用 auto + `-fit on` + `--fit-target`）；参照 server README 完善参数。
4. **细节修正**：系统状态曲线显示异常（改为服务端历史 + 网格 + 按实际时间铺满）；为每个参数补充作用说明。
5. **归档与改名**：项目更名为 llamaRuns，新增本归档文档。

## 已知限制

- 温度：Windows 通常需管理员权限；macOS 纯 Go 构建取不到 CPU 温度与 GPU 占用；Windows 的“交换”显示的是提交内存。
- AMD 显卡监控仅 Linux（rocm-smi / sysfs）；Windows AMD 无监控数据。
- 在终端运行 llama-cli 只在面板所在机器上生效；Android 不支持。
- Windows 生成的命令为 PowerShell 格式（`& "路径" ...`），cmd 中需去掉开头的 `& `。
