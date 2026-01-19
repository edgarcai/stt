```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

## 项目概述

**STT-Dev** 是一个本地化离线运行的语音识别转文字 Web 应用系统，基于多种开源语音识别引擎（Whisper、FunASR），提供统一的 API 接口和 Web 界面。

## 核心特性

- **多后端支持**: faster-whisper (CPU/CUDA)、mlx-whisper (Apple Silicon)、FunASR (阿里达摩院)
- **离线运行**: 模型下载后无需联网，保护数据隐私
- **OpenAI 兼容**: 提供与 OpenAI Whisper API 兼容的 REST 接口
- **多格式输出**: 支持 SRT 字幕、JSON、纯文本三种格式
- **简繁转换**: 内置 OpenCC 支持繁简体相互转换
- **自动设备检测**: 自动识别并使用最佳推理设备 (CUDA > MLX > CPU)

## 技术栈

```
前端层：HTML5 + Layui + Jinja2 模板
Web 框架：Flask + Gevent (异步 WSGI)
业务逻辑：Python 3.9-3.12
推理引擎：
  - faster-whisper (CTranslate2) → CPU/CUDA
  - mlx-whisper (Apple MLX) → Apple Silicon
  - FunASR (PyTorch) → 阿里语音模型
音频处理：FFmpeg (格式转换、采样率统一)
深度学习框架：PyTorch + (可选) CUDA
```

## 目录结构

```
stt-dev/
├── start.py                    # 主入口：Web 服务 + 后台 Worker 线程
├── set.ini                     # 用户配置文件
├── version.json                # 版本信息
├── requirements.txt            # Python 依赖
│
├── stslib/                     # 核心库目录
│   ├── __init__.py             # 包初始化，定义 VERSION
│   ├── cfg.py                  # 全局配置：路径常量、任务队列、语言映射表
│   ├── tool.py                 # 工具函数：FFmpeg 调用、时间格式转换
│   ├── whisper_adapter.py      # Whisper 模型适配器 (统一封装)
│   ├── funasr_adapter.py       # FunASR 模型适配器
│   ├── funasr_model.py         # FunASR 自定义模型定义 (remote_code)
│   └── ctc.py                  # CTC 解码支持
│
├── models/                     # 模型存储目录
│   ├── models--Systran--*/     # faster-whisper 模型缓存
│   ├── models--openai--*/      # mlx-whisper 模型缓存
│   └── modelscope/             # FunASR ModelScope 模型缓存
│
├── static/                     # 静态资源
│   ├── layui/                  # Layui UI 框架
│   └── tmp/                    # 临时文件（上传的音视频、转换的 WAV）
│
└── templates/
    └── index.html              # 主页面模板
```

## 快速开始

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如需 CUDA 加速（NVIDIA GPU）
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 启动服务

```bash
python start.py
```

服务将在 `http://127.0.0.1:9977` 启动并自动打开浏览器。

## 核心模块

### 1. 模型适配器层 (Adapter Pattern)

项目采用适配器设计模式，将不同的语音识别引擎封装为统一接口：

- **WhisperModelAdapter**: 封装 faster-whisper 和 mlx-whisper，自动根据设备类型选择后端
- **FunASRModelAdapter**: 封装阿里达摩院的 FunASR 模型

```python
# 工厂函数自动选择适配器
from stslib.whisper_adapter import create_whisper_model
model = create_whisper_model("large-v3", device_type="auto")
segments, info = model.transcribe("audio.wav")
```

### 2. 设备自动检测

```python
from stslib.whisper_adapter import get_optimal_device
device = get_optimal_device()  # 返回 'mlx' (Apple Silicon) | 'cuda' (NVIDIA) | 'cpu'
```

### 3. 任务队列与异步处理

- 任务队列: `cfg.TASK_QUEUE` (列表)
- 模型缓存: `cfg.MODEL_DICT` (字典)
- 进度跟踪: `cfg.progressbar` 和 `cfg.progressresult` (字典)

### 4. Web 服务 API

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 主页 |
| `/upload` | POST | 文件上传 |
| `/process` | POST | 提交识别任务 |
| `/progressbar` | POST | 查询进度 |
| `/api` | POST | 原生 API |
| `/v1/audio/transcriptions` | POST | OpenAI 兼容 API |

#### OpenAI 兼容 API 示例

```python
from openai import OpenAI
client = OpenAI(api_key='any', base_url='http://127.0.0.1:9977/v1')
audio_file = open("speech.wav", "rb")
transcription = client.audio.transcriptions.create(
    model="small",
    file=audio_file,
    response_format="text"
)
print(transcription.text)
```

#### 原生 API 示例

```python
import requests
url = "http://127.0.0.1:9977/api"
files = {"file": open("audio.wav", "rb")}
data = {"language": "zh", "model": "base", "response_format": "json"}
response = requests.post(url, data=data, files=files, timeout=600)
print(response.json())
```

## 配置文件 (set.ini)

```ini
[server]
web_address = 127.0.0.1:9977
lang = zh

[model]
devtype = auto
cuda_com_type = float32
beam_size = 5
best_of = 5
vad = true
temperature = 0
condition_on_previous_text = false
opencc = t2s
initial_prompt_zh = 转录为中文简体。
model_list = tiny,base,small,medium,large-v3,fun-asr-nano,sensevoice-small,paraformer-zh
```

## 模型管理

### 支持的模型

#### Whisper 模型

- tiny, base, small, medium, large-v3 (faster-whisper 或 mlx-whisper)
- distil-whisper 系列

#### FunASR 模型

- fun-asr-nano: FunAudioLLM/Fun-ASR-Nano-2512 (31种语言，中文优化)
- sensevoice-small: iic/SenseVoiceSmall (高精度多语言识别)
- paraformer-zh: iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch (FunClip同款)

### 模型下载与缓存

- **faster-whisper**: 自动从 Hugging Face 下载到 `models/` 目录
- **mlx-whisper**: 自动从 Hugging Face 下载到 `models/` 目录（Apple Silicon）
- **FunASR**: 自动从 ModelScope 下载到 `models/modelscope/` 目录

## 音频处理

所有音视频文件会被统一转换为 16kHz 单声道 WAV 格式：

```python
params = [
    "-i", input_file,
    "-ar", "16000",     # 采样率 16kHz
    "-ac", "1",         # 单声道
    output_file
]
```

## 输出格式

| 格式 | 说明 |
|------|------|
| **srt** | SRT 字幕格式 |
| **json** | 结构化格式 ([{"line":1,"start_time":"00:00:00,000","end_time":"00:00:05,000","text":"你好"}]) |
| **text** | 纯文本格式 |

## 性能优化

### 推理加速策略

| 策略 | 适用场景 | 配置 |
|------|----------|------|
| **MLX 加速** | Apple Silicon | `devtype=mlx` |
| **CUDA 加速** | NVIDIA GPU | `devtype=cuda` |
| **VAD 过滤** | 长音频 | `vad=true` |
| **降低 beam_size** | 速度优先 | `beam_size=1` |

### 内存管理

- 队列空闲时自动释放模型
- 禁用前文依赖 (`condition_on_previous_text=false`)
- 贪婪解码 (`temperature=0`)

## 开发命令

### 运行测试

```bash
python test.py          # 基础测试
python testcuda.py      # CUDA 可用性测试
```

### 检查更新

程序启动时会自动检查更新，或手动访问 `/checkupdate` 路由。

## 扩展开发

### 添加新后端

1. 创建适配器类实现 `transcribe()` 和 `unload()` 方法
2. 更新工厂函数 `create_whisper_model()`
3. 更新 `set.ini` 的 `model_list` 配置

### 添加新输出格式

在 `_api_process()` 函数中添加格式处理分支。

## 常见问题

### CUDA 加速问题

1. 确保已安装 CUDA Toolkit 和 cuDNN
2. 验证 CUDA 安装：`nvcc --version` 和 `nvidia-smi`
3. 修改 `set.ini` 中的 `devtype=cuda`

### Apple Silicon 加速

- 确保使用 Python 3.9+
- 自动检测到 Apple Silicon 会使用 mlx-whisper
- 可手动设置 `devtype=mlx`

### 模型下载失败

- 检查网络连接
- 确保 Hugging Face/HF Mirror 可访问（中国用户建议使用 HF Mirror）
- 手动下载模型到 `models/` 目录
