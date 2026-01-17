# -*- coding: utf-8 -*-
"""
FunASR 模型适配器模块

该模块提供 FunASR 模型的适配器，支持 Fun-ASR-Nano-2512 和 SenseVoice 等模型。
FunASR 是阿里达摩院/通义实验室开发的语音识别工具包，支持 31 种语言。
"""

import os


# 获取 stslib 目录路径
_STSLIB_DIR = os.path.dirname(os.path.abspath(__file__))

# FunASR 模型配置
FUNASR_MODELS = {
    "fun-asr-nano": {
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",  # 添加标点模型
        "timestamp_model": "fa-zh", # 添加强制对齐模型以获取时间戳
        "trust_remote_code": True,
        "remote_code": os.path.join(_STSLIB_DIR, "funasr_model.py"),
        "description": "FunASR Nano 2512 - 31种语言，针对中文优化"
    },
    "sensevoice-small": {
        "model": "iic/SenseVoiceSmall", 
        "vad_model": "fsmn-vad",
        "description": "SenseVoice Small - 高精度多语言识别"
    },
    "paraformer-zh": {
        "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "description": "Paraformer Large - FunClip同款 (推荐用于字幕生成)"
    },
    "funclip-paraformer": {
        "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "description": "FunClip 官方模型 - Paraformer Large"
    },
}


def is_funasr_model(model_name):
    """
    检查模型名称是否为 FunASR 模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        bool: 如果是 FunASR 模型返回 True
    """
    model_lower = model_name.lower()
    return (
        model_lower in FUNASR_MODELS or
        model_lower.startswith("fun-asr") or
        model_lower.startswith("funasr") or
        model_lower.startswith("sensevoice") or
        model_lower.startswith("paraformer") or
        "FunAudioLLM" in model_name or
        "SenseVoice" in model_name
    )


class FunASRModelAdapter:
    """
    FunASR 模型适配器
    
    提供与 WhisperModelAdapter 兼容的接口来使用 FunASR 模型。
    """
    
    def __init__(self, model_name, device_type="auto", download_root=None):
        """
        初始化 FunASR 模型适配器
        
        Args:
            model_name: 模型名称 (如 'fun-asr-nano', 'sensevoice-small')
            device_type: 设备类型 ('cpu', 'cuda', 'mps', 'auto')
            download_root: 模型下载目录
        """
        self.original_model_name = model_name
        self.download_root = download_root
        
        # 解析设备类型
        if device_type == "auto":
            self.device = self._get_optimal_device()
        elif device_type == "mlx":
            # FunASR 不支持 MLX，使用 MPS
            self.device = "mps" if self._is_apple_silicon() else "cpu"
        else:
            self.device = device_type
        
        self._model = None
        self._backend = "funasr"
        
        # 获取模型配置
        self._model_config = self._get_model_config(model_name)
        
    def _is_apple_silicon(self):
        """检测是否为 Apple Silicon"""
        import platform
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    
    def _get_optimal_device(self):
        """自动检测最佳设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _get_model_config(self, model_name):
        """获取模型配置"""
        model_lower = model_name.lower()
        
        # 检查预定义配置
        if model_lower in FUNASR_MODELS:
            return FUNASR_MODELS[model_lower].copy()
        
        # Fun-ASR-Nano 变体
        if "nano" in model_lower or "fun-asr" in model_lower:
            return FUNASR_MODELS["fun-asr-nano"].copy()
        
        # SenseVoice 变体
        if "sensevoice" in model_lower:
            return FUNASR_MODELS["sensevoice-small"].copy()
        
        # Paraformer 变体
        if "paraformer" in model_lower:
            return FUNASR_MODELS["paraformer-zh"].copy()
        
        # 默认使用 Fun-ASR-Nano
        return {
            "model": model_name if "/" in model_name else f"FunAudioLLM/{model_name}",
            "vad_model": "fsmn-vad",
            "trust_remote_code": True,
            "remote_code": os.path.join(_STSLIB_DIR, "funasr_model.py")
        }
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self._model is not None:
            return
        
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "funasr 未安装。请运行: pip install funasr\n"
                "FunASR 需要额外依赖，可能需要较长时间安装。"
            )
        
        config = self._model_config
        model_kwargs = {
            "model": config["model"],
            "device": self.device,
            "disable_update": True,  # 禁用更新检查以加快加载速度
        }
        
        # Fun-ASR-Nano 需要 trust_remote_code
        if config.get("trust_remote_code"):
            model_kwargs["trust_remote_code"] = True
        if config.get("remote_code"):
            model_kwargs["remote_code"] = config["remote_code"]
        
        # 添加 VAD 模型
        if config.get("vad_model"):
            model_kwargs["vad_model"] = config["vad_model"]
            model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        
        # 添加标点模型
        if config.get("punc_model"):
            model_kwargs["punc_model"] = config["punc_model"]

        # 添加时间戳模型
        if config.get("timestamp_model"):
            model_kwargs["timestamp_model"] = config["timestamp_model"]
        
        print(f"加载 FunASR 模型: {config['model']}")
        print(f"设备: {self.device}")
        print(f"模型参数: {model_kwargs}")
        self._model = AutoModel(**model_kwargs)
        print(f"FunASR 模型加载完成")
    
    def transcribe(self, audio_file, **kwargs):
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径
            **kwargs: 转录参数
                - language: 语言代码 (可选)
                
        Returns:
            tuple: (segments 生成器, info 对象) - 与 faster-whisper 兼容的格式
        """
        self._load_model()
        
        print(f"开始 FunASR 转录，音频文件: {audio_file}")
        
        # 构建生成参数
        # 参考 FunClip 实现: https://github.com/edgarcai/FunClip/blob/main/funclip/videoclipper.py
        generate_kwargs = {
            "input": audio_file,
            "cache": {},
            "batch_size_s": 0,
            "sentence_timestamp": True,
            "return_raw_text": True,
            "is_final": True,
            "disable_pbar": True,  # 禁用进度条，避免 RTF 计算负值和终端输出混乱
        }
        
        # SenseVoice 特定参数
        if "sensevoice" in self._model_config["model"].lower():
            generate_kwargs["language"] = kwargs.get("language", "auto")
            generate_kwargs["use_itn"] = True
            generate_kwargs["merge_vad"] = True
        
        print(f"FunASR generate 参数: {generate_kwargs}")
        result = self._model.generate(**generate_kwargs)
        print(f"FunASR 转录完成")
        
        # 转换为与 faster-whisper 兼容的格式
        return self._convert_result(result, audio_file)
    
    def _convert_result(self, result, audio_file):
        """
        将 FunASR 结果转换为 faster-whisper 兼容格式
        
        Args:
            result: FunASR 的返回结果
            audio_file: 音频文件路径
            
        Returns:
            tuple: (segments 生成器, info 对象)
        """
        # 创建 info 对象
        class TranscriptionInfo:
            def __init__(self, duration, language):
                self.duration = duration
                self.language = language
        
        # 创建 segment 对象
        class Segment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
        
        # 解析 FunASR 结果
        if not result or len(result) == 0:
            return iter([]), TranscriptionInfo(0, "unknown")
        
        first_result = result[0]
        
        # 优先使用 sentence_info (FunClip logic)
        sentence_info = first_result.get("sentence_info", [])
        timestamps = first_result.get("timestamp", [])
        full_text = first_result.get("text", "")
        
        duration = 0
        
        # 尝试估算总时长
        if sentence_info:
             last_sent = sentence_info[-1]
             if 'timestamp' in last_sent and last_sent['timestamp']:
                 duration = last_sent['timestamp'][-1][1] / 1000.0
        elif timestamps:
             if len(timestamps[-1]) > 1:
                duration = timestamps[-1][1] / 1000.0
        
        info = TranscriptionInfo(duration, "zh")
        
        def segment_generator():
            if sentence_info:
                # 使用 sentence_info 生成字幕 (FunClip 方式)
                for sent in sentence_info:
                    text = sent.get('text', '')
                    ts = sent.get('timestamp', [])
                    if ts:
                        # ts 是一系列 token 的时间戳 [[s,e], [s,e], ...]
                        start_ms = ts[0][0]
                        end_ms = ts[-1][1]
                        yield Segment(
                            start=start_ms / 1000.0,
                            end=end_ms / 1000.0,
                            text=text
                        )
            elif timestamps:
                # 回退旧逻辑
                print("FunASR INFO: 未找到 sentence_info，使用 timestamp 回退逻辑")
                for ts in timestamps:
                    if len(ts) >= 3:
                        start_ms, end_ms, text = ts[0], ts[1], ts[2]
                        yield Segment(
                            start=start_ms / 1000.0,
                            end=end_ms / 1000.0,
                            text=text
                        )
            else:
                 # 无时间戳
                 print("FunASR WARNING: 模型未返回时间戳，生成的字幕将没有准确时间！")
                 yield Segment(
                    start=0,
                    end=duration if duration > 0 else 10.0,
                    text=full_text
                )
        
        return segment_generator(), info
    
    def unload(self):
        """卸载模型以释放内存"""
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
