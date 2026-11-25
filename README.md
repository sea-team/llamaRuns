# llamaRuns

llama-server 的 Web 管理面板：扫描本地 GGUF 模型、配置参数、启动/切换模型、实时查看输出，并监控 CPU/GPU/内存。单文件运行，支持 Windows / Linux / macOS / Android(Termux)。

## 运行

```bash
go build -o llamaRuns .      # 或 ./build.sh 交叉编译全部平台到 dist/
./llamaRuns                  # 默认 http://127.0.0.1:8686
./llamaRuns -addr 0.0.0.0:8686 -config /path/config.json
```

开发说明（架构、接口、约定、需求归档）见 [CLAUDE.md](CLAUDE.md)。

- `-addr`：面板监听地址。面板可以启动进程，**不要在不可信网络中暴露**。
- `-config`：配置文件路径，默认为程序所在目录下的 `config.json`；模型介绍缓存在其旁边的 `cache/info/`，更新下载的安装包暂存在 `cache/update/`（更新成功后删除）。

## 功能

- **引导页**：检测系统、架构与显卡（NVIDIA/AMD/Intel/Adreno/Apple），读取驱动支持的 CUDA 版本，从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 最新版本中推荐对应的 GPU 版与 CPU 版下载包（CUDA 版附带 cudart 运行库），并引导完成程序路径与模型目录配置。首次使用自动打开。
- **llama.cpp 更新**：打开页面时检测一次 GitHub 最新发布（服务端缓存 30 分钟），设置页可手动检测。有新版本时顶部提示，可一键更新：按程序目录中的文件识别包类型（CPU / CUDA / Vulkan / ROCm / SYCL 等，CUDA 保持原主版本，缺少运行库时一并下载 cudart），下载并按 sha256 校验后替换程序目录中的文件；替换期间停止使用该程序的实例，完成后在原端口重新启动；新版本无法运行时自动还原。也可手动选择其它发布包（如从 CPU 版换成 Vulkan 版）。可开启“检测到新版本时自动更新”。程序来自系统 PATH 时不自动更新。
- **llama-server 路径**：CPU 版与 GPU 版分别配置（文件或所在目录，页面内浏览选择）。GPU 方式用 GPU 版；CPU 方式用 CPU 版，未配置时用 GPU 版加 `-ngl 0 --device none`。GPU 版留空则使用 PATH。无可用 GPU 时强制 CPU 运行。
- **模型目录**：可添加多个，识别 `目录/模型.gguf` 与 `目录/子目录/模型.gguf`；同一子目录或同名不同量化的模型归为一组折叠展示，分片模型合并为一项。支持共享目录：Windows 直接填 `\\主机\共享\目录` 或映射的网络驱动器（选择器输入 `\\主机` 可列出共享）；Linux/macOS 先挂载 SMB/NFS，可填挂载点，或填 `smb://主机/共享`、`nfs://主机/导出路径`、`主机:/导出路径` 自动匹配已挂载位置。网络目录读取限时 10 秒，不可用时在设置页与模型页提示。
- **启动**：可选择版本（默认上次运行的）、运行方式（有 GPU 时可选 GPU/CPU）、是否启用视觉（`mmproj*.gguf`），选择会被记住。
- **llama-cli**：生成完整命令一键复制，或直接在新终端中运行（Windows 优先 Windows Terminal，否则新 PowerShell 窗口；macOS 用“终端”；Linux 自动识别常见终端）。
- **参数**：按 GPU 与显存 / 上下文与性能 / 采样 / 模板与推理 / 服务 分区，覆盖 `-ngl`、`-fit`、`-fitt`、`-sm`、`-ts`、`-ncmoe`、`-nkvo`、`-ctk/-ctv`、`-cram`、`-rea`、`--reasoning-budget` 等常用参数；优先级 模型参数 → 全局参数 → 内置默认（`-ngl auto`，新版默认 `-fit on`）。显存充裕时可设 `-ngl 999` 以获得确定、可复现的全层卸载。每个参数都标出命令行写法，并分“作用 / 怎么填”说明，可按关键字搜索或只看常用参数。
- **推荐配置**（全局设置）：填入内存、显存并选择模型，读取 GGUF 文件头（层数、KV 结构、MoE 专家权重等）估算权重 / KV 缓存 / 计算缓冲占用，按“全部放 GPU → MoE 专家部分放 CPU（`-ncmoe`/`-cmoe`）→ 部分层放 GPU → 纯 CPU”选出最快且放得下的方案，并按内存规格（选择 DDR4-3200、DDR5-5600、LPDDR5X、Apple 芯片等具体规格，与通道一起换算带宽）估算生成速度，生成完整 llama-server 命令、占用示意与逐项原因说明，可一键写入该模型参数。模型列表中每组也有“推荐配置”入口。
- **单开/多开**、**实时日志**（只保留最近 N 行）。
- **模型介绍与推荐参数**：优先在 ModelScope / HuggingFace 搜索 GGUF 仓库，展示 README；从 README 与 `generation_config.json`（含 base_model）提取推荐采样参数，可一键应用。支持代理与 HF 镜像；GitHub（更新检测与下载）可单独配置代理与下载镜像（镜像仅用于带 sha256 的发布包，下载后校验），并可“测试连接”；代理都留空时自动跟随系统代理（环境变量、Windows/macOS 系统代理设置，如 Clash、v2rayN 的系统代理模式）。
- **系统状态页**：主机信息、CPU（型号/占用/温度）、内存与交换、每块显卡的占用/显存/温度/功耗/风扇/驱动/CUDA，带近 3 分钟曲线，以及每个实例的内存与 CPU 占用；顶部保留精简的 CPU/内存/显存占用率。

## 平台说明

- 温度读取依赖系统传感器：Windows 通常需管理员权限；macOS 的 CPU 温度与 GPU 占用暂不支持（纯 Go 构建）。
- Android：在 Termux 中运行 `android-arm64` 或 `linux-arm64` 版本，配合 Termux 编译/安装的 llama-server。
- `-fa on/off/auto` 为新版 llama.cpp 语法；旧版本若报错，可将 Flash Attention 设为“不传递”，需要时用附加参数 `-fa`。
