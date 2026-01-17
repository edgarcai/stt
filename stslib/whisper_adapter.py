# -*- coding: utf-8 -*-
"""
Whisper 模型适配器模块

该模块提供统一的接口来封装 faster-whisper 和 mlx-whisper，
根据设备类型自动选择合适的后端实现。

支持的设备类型：
- cpu: 使用 faster-whisper CPU 模式
- cuda: 使用 faster-whisper CUDA 模式  
- mlx: 使用 mlx-whisper (仅 Apple Silicon)
- auto: 自动检测最佳设备类型
"""

import platform
import os
import sys

# 设置 Hugging Face 缓存目录到项目的 models 目录
# 这样 mlx-whisper 下载的模型也会存储在项目内
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_ROOT_DIR, "models")
os.environ["HF_HUB_CACHE"] = _MODELS_DIR


# MLX 模型名称映射表：faster-whisper 模型名 -> mlx-whisper 模型名
# 注意：使用 OpenAI 官方 Whisper 模型仓库，mlx-whisper 会自动下载并转换为 MLX 格式
MLX_MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base",
    "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small",
    "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium",
    "medium.en": "openai/whisper-medium.en",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}


def is_apple_silicon():
    """
    检测当前系统是否为 Apple Silicon (M1/M2/M3/M4 等)
    
    Returns:
        bool: 如果是 Apple Silicon 返回 True，否则返回 False
    """
    if platform.system() != "Darwin":
        return False
    try:
        # 检测处理器架构
        return platform.machine() == "arm64"
    except Exception:
        return False


def is_cuda_available():
    """
    检测 CUDA 是否可用
    
    Returns:
        bool: 如果 CUDA 可用返回 True，否则返回 False
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_optimal_device():
    """
    自动检测最佳设备类型
    
    Returns:
        str: 最佳设备类型 ('mlx', 'cuda', 或 'cpu')
    """
    if is_apple_silicon():
        return "mlx"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_mlx_model_name(model_name):
    """
    将 faster-whisper 模型名称转换为 mlx-whisper 模型名称
    
    Args:
        model_name: faster-whisper 格式的模型名称
        
    Returns:
        str: mlx-whisper 格式的模型名称
    """
    # 如果已经是完整的 HuggingFace 路径，直接返回
    if "/" in model_name:
        return model_name
    
    # 处理 distil-whisper 系列
    if model_name.startswith("distil-"):
        clean_name = model_name.replace("distil-whisper-", "distil-whisper/distil-")
        return clean_name
    
    # 标准模型映射
    return MLX_MODEL_MAP.get(model_name, f"mlx-community/whisper-{model_name}")


# -----------------------------------------------------------------------------
# Monkey Patch: 修复 mlx-whisper 兼容性问题
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Monkey Patch: 修复 mlx-whisper 兼容性问题
# -----------------------------------------------------------------------------
try:
    import mlx_whisper.whisper
    
    # 获取原始类
    OriginalModelDimensions = mlx_whisper.whisper.ModelDimensions
    
    # 增强 Patch：同时支持位置参数和关键字参数
    def patched_init(self, *args, **kwargs):
        # 参数名列表，对应 dataclass 的字段顺序
        keys = [
            "n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", 
            "n_audio_layer", "n_vocab", "n_text_ctx", "n_text_state", 
            "n_text_head", "n_text_layer"
        ]
        
        # 处理位置参数
        for i, val in enumerate(args):
            if i < len(keys):
                setattr(self, keys[i], val)
                
        # 处理关键字参数 (覆盖位置参数，如果有)
        for k in keys:
            if k in kwargs:
                setattr(self, k, kwargs[k])
                
        # 设置默认值 (如果未设置)
        defaults = {
            "n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 1280,
            "n_audio_head": 20, "n_audio_layer": 32, "n_vocab": 51865,
            "n_text_ctx": 448, "n_text_state": 1280, "n_text_head": 20,
            "n_text_layer": 32
        }
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        
        # 忽略其他无关参数
    
    # 应用 Patch
    mlx_whisper.whisper.ModelDimensions.__init__ = patched_init
    print("已应用 mlx-whisper ModelDimensions 兼容性补丁 (v2)")

except ImportError:
    pass
except Exception as e:
    print(f"应用 Patch 失败: {e}")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class WhisperModelAdapter:
    """
    Whisper 模型适配器
    
    提供统一的接口封装 faster-whisper 和 mlx-whisper，
    自动根据设备类型选择合适的后端。
    """
    
    def __init__(self, model_name, device_type="auto", download_root=None):

        """
        初始化 Whisper 模型适配器
        
        Args:
            model_name: 模型名称 (如 'tiny', 'base', 'small', 'large-v3')
            device_type: 设备类型 ('cpu', 'cuda', 'mlx', 'auto')
            download_root: 模型下载目录 (仅对 faster-whisper 有效)
        """
        self.original_model_name = model_name
        self.download_root = download_root
        
        # 解析实际设备类型
        if device_type == "auto":
            self.device_type = get_optimal_device()
        else:
            self.device_type = device_type
        
        # 验证设备类型
        if self.device_type == "mlx" and not is_apple_silicon():
            print("警告: MLX 模式仅支持 Apple Silicon，回退到 CPU 模式")
            self.device_type = "cpu"
        
        self._model = None
        self._backend = None
        
    def _load_model(self):
        """
        加载模型（延迟加载）
        """
        if self._model is not None:
            return
            
        if self.device_type == "mlx":
            self._load_mlx_model()
        else:
            self._load_faster_whisper_model()
    
    def _load_mlx_model(self):
        """
        加载 MLX Whisper 模型
        """
        try:
            import mlx_whisper
            self._backend = "mlx"
            # MLX 模型在 transcribe 时加载，这里只标记
            self._model = "mlx_lazy"
            print(f"已选择 MLX 后端，模型: {get_mlx_model_name(self.original_model_name)}")
        except ImportError as e:
            print(f"警告: 无法导入 mlx_whisper ({e})，回退到 faster-whisper CPU 模式")
            self.device_type = "cpu"
            self._load_faster_whisper_model()
    
    def _load_faster_whisper_model(self):
        """
        加载 faster-whisper 模型
        """
        from faster_whisper import WhisperModel
        
        model_name = self.original_model_name
        # 处理 distil-whisper 模型名称
        if model_name.startswith('distil-'):
            model_name = model_name.replace('-whisper', '')
        
        self._model = WhisperModel(
            model_name,
            device=self.device_type,
            download_root=self.download_root
        )
        self._backend = "faster-whisper"
        print(f"已选择 faster-whisper 后端 ({self.device_type})，模型: {model_name}")
    
    def transcribe(self, audio_file, **kwargs):
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径
            **kwargs: 转录参数
                - beam_size: 束搜索大小
                - best_of: 候选采样数
                - language: 语言代码
                - initial_prompt: 初始提示
                - vad_filter: VAD 过滤
                - condition_on_previous_text: 是否基于前文
                
        Returns:
            tuple: (segments 生成器, info 对象) - 与 faster-whisper 兼容的格式
        """
        self._load_model()
        
        if self._backend == "mlx":
            return self._transcribe_mlx(audio_file, **kwargs)
        else:
            return self._transcribe_faster_whisper(audio_file, **kwargs)
    
    def _transcribe_mlx(self, audio_file, **kwargs):
        """
        使用 MLX Whisper 进行转录
        """
        import mlx_whisper
        
        mlx_model_name = get_mlx_model_name(self.original_model_name)
        
        # 构建 mlx_whisper 参数
        mlx_kwargs = {
            "path_or_hf_repo": mlx_model_name,
        }
        
        # 映射参数
        # 注意：mlx-whisper 目前不支持 beam_size（Beam search decoder is not yet implemented）
        if kwargs.get("language") and kwargs["language"] != "auto":
            mlx_kwargs["language"] = kwargs["language"]
        if kwargs.get("initial_prompt"):
            mlx_kwargs["initial_prompt"] = kwargs["initial_prompt"]
        
        # 总是启用 word_timestamps 以获得更精确的时间戳
        mlx_kwargs["word_timestamps"] = True
        
        print(f"开始 MLX 转录，音频文件: {audio_file}")
        print(f"MLX 参数: {mlx_kwargs}")
        result = mlx_whisper.transcribe(audio_file, **mlx_kwargs)
        print(f"MLX 转录完成")
        
        # 转换为与 faster-whisper 兼容的格式
        return self._convert_mlx_result(result)
    
    def _convert_mlx_result(self, result):
        """
        将 MLX Whisper 结果转换为 faster-whisper 兼容格式
        
        Args:
            result: mlx_whisper.transcribe() 的返回结果
            
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
        
        # 计算总时长
        segments_data = result.get("segments", [])
        if segments_data:
            duration = segments_data[-1].get("end", 0)
        else:
            duration = 0
        
        language = result.get("language", "unknown")
        info = TranscriptionInfo(duration, language)
        
        # 创建 segments 生成器
        def segment_generator():
            for seg in segments_data:
                yield Segment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", "")
                )
        
        return segment_generator(), info
    
    def _transcribe_faster_whisper(self, audio_file, **kwargs):
        """
        使用 faster-whisper 进行转录
        """
        return self._model.transcribe(audio_file, **kwargs)
    
    def unload(self):
        """
        卸载模型以释放内存
        """
        if self._backend == "faster-whisper":
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        elif self._backend == "mlx":
            # MLX 使用延迟加载，无需特别卸载
            self._model = None
        
        self._backend = None


def create_whisper_model(model_name, device_type="auto", download_root=None):
    """
    创建语音识别模型实例的工厂函数
    
    自动检测模型类型并返回对应的适配器：
    - Whisper 模型 -> WhisperModelAdapter
    - FunASR 模型 -> FunASRModelAdapter
    
    Args:
        model_name: 模型名称
        device_type: 设备类型
        download_root: 模型下载目录
        
    Returns:
        适配器实例 (WhisperModelAdapter 或 FunASRModelAdapter)
    """
    # 检查是否为 FunASR 模型
    try:
        from stslib.funasr_adapter import is_funasr_model, FunASRModelAdapter
        if is_funasr_model(model_name):
            return FunASRModelAdapter(model_name, device_type, download_root)
    except ImportError:
        pass
    
    # 默认使用 Whisper 适配器
    return WhisperModelAdapter(model_name, device_type, download_root)

