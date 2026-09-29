# llamaRunModel

llama-server 的 Web 管理面板：扫描本地 GGUF 模型、配置参数、启动/切换模型、实时查看输出，并监控 CPU/GPU/内存。单文件运行，支持 Windows / Linux / macOS / Android(Termux)。

## 运行

```bash
go build -o llamaRunModel .      # 或 ./build.sh 交叉编译全部平台
./llamaRunModel                  # 默认 http://127.0.0.1:8686
./llamaRunModel -addr 0.0.0.0:8686 -config /path/config.json
```

- `-addr`：面板监听地址。面板可以启动进程，**不要在不可信网络中暴露**。
- `-config`：配置文件路径，默认为程序所在目录下的 `config.json`；模型介绍缓存在其旁边的 `cache/info/`。

## 功能

- **llama-server 路径**：llama.cpp 发布包分 CPU 版与 GPU 版（CUDA/Vulkan/ROCm…），可分别配置（程序文件或所在目录，支持在页面中浏览选择）。GPU 方式使用 GPU 版；CPU 方式使用 CPU 版，未配置 CPU 版时用 GPU 版加 `-ngl 0 --device none`。GPU 版留空则使用 PATH 中的 llama-server。“检测”通过 `--list-devices` 判断 GPU，无 GPU 时强制 CPU 运行并禁用 GPU 选项。
- **模型目录**：可添加多个（页面中浏览选择），识别 `目录/模型.gguf` 与 `目录/子目录/模型.gguf`。同一子目录下的模型、或根目录下去掉量化后缀同名的模型归为一组，折叠展示；分片模型合并为一项。
- **启动**：可选择组内版本（默认上次运行的版本）；有 `mmproj*.gguf` 时可勾选是否启用视觉。每次选择会被记住。
- **命令生成**：同时生成 llama-server 与 llama-cli（与 llama-server 同目录）的完整命令，一键复制。
- **参数优先级**：模型参数 → 全局参数 → 内置默认，模型参数对组内所有版本生效。常用参数有表单项，其它参数写在“附加参数”或启动时临时附加。
- **单开/多开**：默认单开，启动新模型会先停止旧模型；开启多开后可同时运行，端口自动分配。
- **输出日志**：SSE 实时推送，服务端和页面只保留最近 N 行（可配置）。
- **模型介绍与推荐参数**：由文件名推导关键字，在 ModelScope / HuggingFace 搜索并展示 README；从 README 与 `generation_config.json`（含 base_model）提取推荐的 temp/top-p/top-k/min-p/惩罚/上下文，可勾选后一键应用到模型参数。
- **网络代理**：可配置 http/https/socks5 代理，默认仅用于 HuggingFace，可选 ModelScope 也走代理；也可把 HF 地址改为 `https://hf-mirror.com`。
- **系统状态**：CPU 占用/温度、内存、交换、每个实例的进程内存；GPU 占用/显存/温度支持 NVIDIA（nvidia-smi）与 AMD（rocm-smi 或 Linux sysfs）。设置页展示 nvidia-smi / rocm-smi / amd-smi 的原始信息，无显卡工具时不显示。

## 平台说明

- 温度读取依赖系统传感器：Windows 通常需管理员权限；macOS 的 CPU 温度与 GPU 占用暂不支持（纯 Go 构建）。
- Android：在 Termux 中运行 `android-arm64` 或 `linux-arm64` 版本，配合 Termux 编译/安装的 llama-server。
- `-fa on/off/auto` 为新版 llama.cpp 语法；旧版本若报错，可将 Flash Attention 设为“不传递”，需要时用附加参数 `-fa`。
