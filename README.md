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
- `-config`：配置文件路径，默认为程序所在目录下的 `config.json`；模型介绍缓存在其旁边的 `cache/info/`。

## 功能

- **引导页**：检测系统、架构与显卡（NVIDIA/AMD/Intel/Adreno/Apple），读取驱动支持的 CUDA 版本，从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 最新版本中推荐对应的 GPU 版与 CPU 版下载包（CUDA 版附带 cudart 运行库），并引导完成程序路径与模型目录配置。首次使用自动打开。
- **llama-server 路径**：CPU 版与 GPU 版分别配置（文件或所在目录，页面内浏览选择）。GPU 方式用 GPU 版；CPU 方式用 CPU 版，未配置时用 GPU 版加 `-ngl 0 --device none`。GPU 版留空则使用 PATH。无可用 GPU 时强制 CPU 运行。
- **模型目录**：可添加多个，识别 `目录/模型.gguf` 与 `目录/子目录/模型.gguf`；同一子目录或同名不同量化的模型归为一组折叠展示，分片模型合并为一项。支持共享目录：Windows 直接填 `\\主机\共享\目录` 或映射的网络驱动器（选择器输入 `\\主机` 可列出共享）；Linux/macOS 先挂载 SMB/NFS，可填挂载点，或填 `smb://主机/共享`、`nfs://主机/导出路径`、`主机:/导出路径` 自动匹配已挂载位置。网络目录读取限时 10 秒，不可用时在设置页与模型页提示。
- **启动**：可选择版本（默认上次运行的）、运行方式（有 GPU 时可选 GPU/CPU）、是否启用视觉（`mmproj*.gguf`），选择会被记住。
- **llama-cli**：生成完整命令一键复制，或直接在新终端中运行（Windows 优先 Windows Terminal，否则新 PowerShell 窗口；macOS 用“终端”；Linux 自动识别常见终端）。
- **参数**：按 GPU 与显存 / 上下文与性能 / 采样 / 模板与推理 / 服务 分区，覆盖 `-ngl`、`-fit`、`-fitt`、`-sm`、`-ts`、`-ncmoe`、`-nkvo`、`-ctk/-ctv`、`-cram`、`-rea`、`--reasoning-budget` 等常用参数；优先级 模型参数 → 全局参数 → 内置默认（`-ngl auto`，新版默认 `-fit on`）。显存充裕时可设 `-ngl 999` 以获得确定、可复现的全层卸载。
- **单开/多开**、**实时日志**（只保留最近 N 行）。
- **模型介绍与推荐参数**：优先在 ModelScope / HuggingFace 搜索 GGUF 仓库，展示 README；从 README 与 `generation_config.json`（含 base_model）提取推荐采样参数，可一键应用。支持代理与 HF 镜像。
- **系统状态页**：主机信息、CPU（型号/占用/温度）、内存与交换、每块显卡的占用/显存/温度/功耗/风扇/驱动/CUDA，带近 3 分钟曲线，以及每个实例的内存与 CPU 占用；顶部保留精简的 CPU/内存/显存占用率。

## 平台说明

- 温度读取依赖系统传感器：Windows 通常需管理员权限；macOS 的 CPU 温度与 GPU 占用暂不支持（纯 Go 构建）。
- Android：在 Termux 中运行 `android-arm64` 或 `linux-arm64` 版本，配合 Termux 编译/安装的 llama-server。
- `-fa on/off/auto` 为新版 llama.cpp 语法；旧版本若报错，可将 Flash Attention 设为“不传递”，需要时用附加参数 `-fa`。
