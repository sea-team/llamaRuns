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

- **llama-server 路径**：可指定文件或所在目录，留空则从系统 PATH 查找。“检测”会读取版本并执行 `--list-devices` 判断 GPU；无 GPU 时强制 CPU 运行（`-ngl 0 --device none`），GPU 选项被禁用。
- **模型目录**：可配置多个，识别 `目录/模型.gguf` 与 `目录/子目录/模型.gguf`；分片模型（`-00001-of-0000N`）合并为一项，`mmproj*.gguf` 作为同目录模型的多模态投影可选项。
- **参数优先级**：模型参数 → 全局参数 → 内置默认。常用参数有表单项；其它参数可写在“附加参数”（全局/模型）中，或在启动时临时附加。
- **单开/多开**：默认单开，启动新模型会先停止旧模型；在设置中开启多开后可同时运行，端口从“起始端口”自动分配（也可按模型固定端口）。
- **输出日志**：通过 SSE 实时推送，服务端和页面都只保留最近 N 行（可配置），避免页面卡顿。
- **模型介绍**：由文件名推导关键字（去掉量化后缀、分片后缀，子目录名兜底），在 ModelScope / HuggingFace 搜索并展示 README，可切换候选仓库；HF 访问受限时可将地址改为 `https://hf-mirror.com`。
- **系统状态**：CPU 占用/温度、内存、交换、每个实例的进程内存；GPU 支持 NVIDIA（nvidia-smi）与 Linux AMD（sysfs）。

## 平台说明

- 温度读取依赖系统传感器：Windows 通常需管理员权限；macOS 的 CPU 温度与 GPU 占用暂不支持（纯 Go 构建）。
- Android：在 Termux 中运行 `android-arm64` 或 `linux-arm64` 版本，配合 Termux 编译/安装的 llama-server。
- `-fa on/off/auto` 为新版 llama.cpp 语法；旧版本若报错，可将 Flash Attention 设为“不传递”，需要时用附加参数 `-fa`。
