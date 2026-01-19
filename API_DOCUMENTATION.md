# STT-Dev API 接口文档

> 版本：v0.0.94  
> 基础地址：`http://127.0.0.1:9977`  
> 更新日期：2026-01-19

---

## 目录

1. [接口概览](#一接口概览)
2. [OpenAI 兼容接口](#二openai-兼容接口)
3. [原生 API 接口](#三原生-api-接口)
4. [Web 前端接口](#四web-前端接口)
5. [错误码说明](#五错误码说明)
6. [代码示例](#六代码示例)

---

## 一、接口概览

### 1.1 接口列表

| 接口路径 | 方法 | 类型 | 描述 |
|----------|------|------|------|
| `/v1/audio/transcriptions` | POST | OpenAI 兼容 | 语音转文字（兼容 OpenAI SDK） |
| `/api` | POST | 原生 API | 语音转文字（自定义格式） |
| `/upload` | POST | Web 前端 | 文件上传 |
| `/process` | POST | Web 前端 | 提交识别任务 |
| `/progressbar` | POST | Web 前端 | 查询识别进度 |
| `/checkupdate` | GET | 工具 | 检查版本更新 |

### 1.2 支持的模型

| 模型名称 | 类型 | 描述 |
|----------|------|------|
| `large-v1` | Whisper | 大型模型 v1，高精度语音识别 |

### 1.3 支持的语言

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| 中文 | `zh` | 英语 | `en` |
| 法语 | `fr` | 德语 | `de` |
| 日语 | `ja` | 韩语 | `ko` |
| 俄语 | `ru` | 西班牙语 | `es` |
| 泰语 | `th` | 意大利语 | `it` |
| 葡萄牙语 | `pt` | 越南语 | `vi` |
| 阿拉伯语 | `ar` | 土耳其语 | `tr` |
| 自动检测 | `auto` | | |

### 1.4 输出格式

| 格式 | 描述 | 示例 |
|------|------|------|
| `text` | 纯文本 | `你好世界` |
| `srt` | SRT 字幕格式 | 带时间戳的字幕 |
| `json` | JSON 结构化数据 | 包含行号、时间、文本 |

---

## 二、OpenAI 兼容接口

### 2.1 语音转文字

**兼容 OpenAI Whisper API，可直接使用 OpenAI Python SDK 调用。**

#### 请求信息

- **URL**: `/v1/audio/transcriptions`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 音频/视频文件（支持 mp3, wav, mp4, flac, m4a, aac 等） |
| `model` | String | ✅ | - | 模型名称（如 `large-v1`, `small`） |
| `language` | String | ❌ | auto | 语言代码（如 `zh`, `en`） |
| `response_format` | String | ❌ | text | 返回格式：`text`, `srt`, `json` |
| `prompt` | String | ❌ | - | 初始提示词，用于指导识别 |

#### 响应格式

**response_format=text 时：**

```json
{
  "text": "识别的文本内容"
}
```

**response_format=srt 时：**

返回纯文本 SRT 字幕：

```
1
00:00:00,000 --> 00:00:03,500
你好世界

2
00:00:03,500 --> 00:00:07,200
这是第二句话
```

**response_format=json 时：**

```json
[
  {
    "line": 1,
    "start_time": "00:00:00,000",
    "end_time": "00:00:03,500",
    "text": "你好世界"
  },
  {
    "line": 2,
    "start_time": "00:00:03,500",
    "end_time": "00:00:07,200",
    "text": "这是第二句话"
  }
]
```

#### 错误响应

```json
{
  "error": "错误描述信息"
}
```

#### cURL 示例

```bash
curl -X POST "http://127.0.0.1:9977/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "model=large-v1" \
  -F "language=zh" \
  -F "response_format=srt"
```

---

## 三、原生 API 接口

### 3.1 语音转文字

#### 请求信息

- **URL**: `/api`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 音频/视频文件 |
| `model` | String | ✅ | - | 模型名称 |
| `language` | String | ❌ | auto | 语言代码 |
| `response_format` | String | ❌ | srt | 返回格式：`text`, `srt`, `json` |

#### 成功响应

```json
{
  "code": 0,
  "msg": "ok",
  "data": "识别结果（根据 response_format 格式化）"
}
```

**data 字段示例：**

当 `response_format=srt` 时：

```json
{
  "code": 0,
  "msg": "ok",
  "data": "1\n00:00:00,000 --> 00:00:03,500\n你好世界\n\n2\n00:00:03,500 --> 00:00:07,200\n这是第二句话\n"
}
```

当 `response_format=json` 时：

```json
{
  "code": 0,
  "msg": "ok",
  "data": [
    {"line": 1, "start_time": "00:00:00,000", "end_time": "00:00:03,500", "text": "你好世界"},
    {"line": 2, "start_time": "00:00:03,500", "end_time": "00:00:07,200", "text": "这是第二句话"}
  ]
}
```

当 `response_format=text` 时：

```json
{
  "code": 0,
  "msg": "ok",
  "data": "你好世界\n这是第二句话"
}
```

#### 错误响应

```json
{
  "code": 1,
  "msg": "错误描述信息"
}
```

或

```json
{
  "code": 2,
  "msg": "错误描述信息"
}
```

#### cURL 示例

```bash
curl -X POST "http://127.0.0.1:9977/api" \
  -F "file=@/path/to/audio.wav" \
  -F "model=large-v1" \
  -F "language=zh" \
  -F "response_format=json"
```

---

## 四、Web 前端接口

### 4.1 文件上传

用于上传音视频文件，服务端会自动转换为 16kHz 单声道 WAV 格式。

#### 请求信息

- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `audio` | File | ✅ | 音频/视频文件 |

#### 成功响应

```json
{
  "code": 0,
  "msg": "上传成功,已转为wav格式",
  "data": "filename.wav"
}
```

#### 错误响应

```json
{
  "code": 1,
  "msg": "具体错误信息"
}
```

---

### 4.2 提交识别任务

将上传的文件加入识别队列。

#### 请求信息

- **URL**: `/process`
- **Method**: `POST`
- **Content-Type**: `application/x-www-form-urlencoded`

#### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `wav_name` | String | ✅ | 上传返回的 WAV 文件名 |
| `model` | String | ✅ | 模型名称 |
| `language` | String | ✅ | 语言代码 |
| `data_type` | String | ✅ | 输出格式：`text`, `srt`, `json` |

#### 成功响应

```json
{
  "code": 0,
  "msg": "ing"
}
```

#### 错误响应

```json
{
  "code": 1,
  "msg": "文件不存在"
}
```

---

### 4.3 查询识别进度

轮询查询识别任务的进度和结果。

#### 请求信息

- **URL**: `/progressbar`
- **Method**: `POST`
- **Content-Type**: `application/x-www-form-urlencoded`

#### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `wav_name` | String | ✅ | WAV 文件名 |
| `model` | String | ✅ | 模型名称 |
| `language` | String | ✅ | 语言代码 |
| `data_type` | String | ✅ | 输出格式 |

#### 进行中响应

```json
{
  "code": 0,
  "data": 0.45,
  "msg": "ok"
}
```

> `data` 字段：0.0 ~ 1.0 的浮点数，表示进度百分比

#### 完成响应

```json
{
  "code": 0,
  "data": 1,
  "msg": "ok",
  "result": "识别结果（根据 data_type 格式化）"
}
```

#### 错误响应

```json
{
  "code": 1,
  "msg": "错误描述信息"
}
```

---

## 五、错误码说明

| code | 含义 | 处理建议 |
|------|------|----------|
| 0 | 成功 | - |
| 1 | 业务错误 | 查看 msg 字段获取详情 |
| 2 | 系统错误 | 查看服务端日志 |

### 常见错误信息

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `FFmpeg 未安装或未在系统 PATH 中` | FFmpeg 未正确安装 | 安装 FFmpeg 并添加到 PATH |
| `ffprobe 未安装或未在系统 PATH 中` | ffprobe 未安装 | 同上 |
| `从 huggingface.co 下载模型失败` | 网络问题或模型不存在 | 检查网络或模型名称 |
| `模型文件不存在` | 本地模型未下载 | 下载模型到 models 目录 |
| `文件不存在` | 上传的文件已被清理 | 重新上传文件 |

---

## 六、代码示例

### 6.1 Python - 使用 OpenAI SDK

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="任意字符串",  # 本地服务无需真实 API Key
    base_url="http://127.0.0.1:9977/v1"
)

# 打开音频文件
audio_file = open("/path/to/audio.mp3", "rb")

# 调用转录接口
transcription = client.audio.transcriptions.create(
    model="large-v1",           # 模型名称
    file=audio_file,            # 音频文件
    language="zh",              # 可选：指定语言
    response_format="text"      # 可选：text, srt, json
)

# 获取结果
print(transcription.text)
```

### 6.2 Python - 使用 requests 库

```python
import requests

# 请求地址
url = "http://127.0.0.1:9977/api"

# 准备文件和参数
files = {"file": open("/path/to/audio.wav", "rb")}
data = {
    "language": "zh",
    "model": "large-v1",
    "response_format": "json"
}

# 发送请求
response = requests.post(url, files=files, data=data, timeout=600)

# 解析结果
result = response.json()
if result["code"] == 0:
    print("识别成功:")
    print(result["data"])
else:
    print("识别失败:", result["msg"])
```

### 6.3 Python - 异步批量处理

```python
import requests
import time

BASE_URL = "http://127.0.0.1:9977"

def upload_file(file_path):
    """上传文件"""
    with open(file_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        return response.json()

def start_process(wav_name, model, language, data_type):
    """提交识别任务"""
    data = {
        "wav_name": wav_name,
        "model": model,
        "language": language,
        "data_type": data_type
    }
    response = requests.post(f"{BASE_URL}/process", data=data)
    return response.json()

def poll_progress(wav_name, model, language, data_type):
    """轮询进度直到完成"""
    data = {
        "wav_name": wav_name,
        "model": model,
        "language": language,
        "data_type": data_type
    }
    
    while True:
        response = requests.post(f"{BASE_URL}/progressbar", data=data)
        result = response.json()
        
        if result["code"] != 0:
            raise Exception(result["msg"])
        
        progress = result["data"]
        print(f"进度: {progress * 100:.1f}%")
        
        if progress >= 1:
            return result.get("result")
        
        time.sleep(0.5)

# 使用示例
def transcribe_audio(file_path, model="large-v1", language="zh", data_type="srt"):
    """完整的转录流程"""
    # 1. 上传文件
    upload_result = upload_file(file_path)
    if upload_result["code"] != 0:
        raise Exception(f"上传失败: {upload_result['msg']}")
    
    wav_name = upload_result["data"]
    print(f"上传成功: {wav_name}")
    
    # 2. 提交任务
    process_result = start_process(wav_name, model, language, data_type)
    if process_result["code"] != 0:
        raise Exception(f"提交失败: {process_result['msg']}")
    
    print("任务已提交，开始处理...")
    
    # 3. 轮询进度
    result = poll_progress(wav_name, model, language, data_type)
    print("识别完成!")
    
    return result

# 调用
if __name__ == "__main__":
    result = transcribe_audio("/path/to/video.mp4")
    print(result)
```

### 6.4 cURL - 命令行使用

**基础调用：**

```bash
# 使用原生 API
curl -X POST "http://127.0.0.1:9977/api" \
  -F "file=@./audio.mp3" \
  -F "model=large-v1" \
  -F "language=zh" \
  -F "response_format=srt"
```

**使用 OpenAI 兼容接口：**

```bash
curl -X POST "http://127.0.0.1:9977/v1/audio/transcriptions" \
  -H "Authorization: Bearer any-key" \
  -F "file=@./audio.mp3" \
  -F "model=large-v1" \
  -F "response_format=text"
```

### 6.5 JavaScript - Fetch API

```javascript
async function transcribeAudio(file, model = 'large-v1', language = 'zh') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('language', language);
    formData.append('response_format', 'json');

    const response = await fetch('http://127.0.0.1:9977/api', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    
    if (result.code === 0) {
        return result.data;
    } else {
        throw new Error(result.msg);
    }
}

// 使用示例
const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    try {
        const subtitles = await transcribeAudio(file);
        console.log('识别结果:', subtitles);
    } catch (error) {
        console.error('识别失败:', error.message);
    }
});
```

### 6.6 Go - HTTP 调用

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "path/filepath"
)

type APIResponse struct {
    Code int         `json:"code"`
    Msg  string      `json:"msg"`
    Data interface{} `json:"data"`
}

func TranscribeAudio(filePath, model, language, responseFormat string) (*APIResponse, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)

    part, err := writer.CreateFormFile("file", filepath.Base(filePath))
    if err != nil {
        return nil, err
    }
    io.Copy(part, file)

    writer.WriteField("model", model)
    writer.WriteField("language", language)
    writer.WriteField("response_format", responseFormat)
    writer.Close()

    req, err := http.NewRequest("POST", "http://127.0.0.1:9977/api", body)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", writer.FormDataContentType())

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result APIResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}

func main() {
    result, err := TranscribeAudio("./audio.mp3", "large-v1", "zh", "srt")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }

    if result.Code == 0 {
        fmt.Println("识别结果:", result.Data)
    } else {
        fmt.Println("识别失败:", result.Msg)
    }
}
```

---

## 附录

### A. 支持的音视频格式

| 类型 | 格式 |
|------|------|
| 音频 | wav, mp3, flac, aac, m4a, ogg, wma |
| 视频 | mp4, mkv, avi, mov, mpeg, wmv, webm |

> 注：所有格式都会被 FFmpeg 自动转换为 16kHz 单声道 WAV

### B. 性能建议

| 场景 | 建议模型 | 预计速度 |
|------|----------|----------|
| 高精度识别 | large-v1 | 实时 x1-2 |

> 注：large-v1 模型在 Apple Silicon (MLX) 或 NVIDIA GPU (CUDA) 上运行效果最佳

### C. 配置调优

在 `set.ini` 中调整以下参数可优化性能：

```ini
; 降低以减少显存占用
beam_size=1
best_of=1

; 关闭以减少显存
vad=false
condition_on_previous_text=false

; 设备选择
devtype=auto  ; 自动选择最佳设备
```
