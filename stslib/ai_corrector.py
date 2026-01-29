# -*- coding: utf-8 -*-
"""
AI 错别字校正模块

该模块使用 AI 大语言模型对语音识别生成的字幕文本进行错别字校正。
支持 OpenAI 兼容的 API 接口。
"""

import re
from typing import List, Dict, Optional
from openai import OpenAI


# 默认配置
DEFAULT_AI_CONFIG = {
    "base_url": "https://integrate.api.nvidia.com/v1",
    "api_key": "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",
    "model": "z-ai/glm4.7",
    "temperature": 0.3,
    "max_tokens": 8192,
    "timeout": 60.0,
}


class AICorrector:
    """
    AI 错别字校正器
    
    使用大语言模型对语音识别结果进行错别字校正，
    保持原有的时间戳和格式不变。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 AI 校正器
        
        Args:
            config: 配置字典，包含 base_url, api_key, model 等
        """
        self.config = {**DEFAULT_AI_CONFIG, **(config or {})}
        self.client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 60.0)
        )
        
    def _create_correction_prompt(self, texts: List[str]) -> str:
        """
        创建校正提示词
        
        Args:
            texts: 待校正的文本列表
            
        Returns:
            str: 完整的提示词
        """
        numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        
        prompt = f"""你是一个专业的字幕校对专家。请对以下语音识别生成的字幕文本进行错别字校正。

要求：
1. 只修正明显的错别字、同音字错误
2. 保持原有的语句结构和标点符号
3. 不要改变原文的意思
4. 不要添加或删除内容
5. 如果文本没有错误，保持原样返回
6. 按照原有的编号格式返回校正后的文本

待校正的文本：
{numbered_texts}

请直接返回校正后的文本，格式为：
1. [校正后的文本1]
2. [校正后的文本2]
...

只返回校正结果，不要有其他说明。"""
        
        return prompt
    
    def _parse_correction_response(self, response: str, original_count: int) -> List[str]:
        """
        解析 AI 返回的校正结果

        Args:
            response: AI 返回的响应文本
            original_count: 原始文本数量

        Returns:
            List[str]: 校正后的文本列表
        """
        if not response:
            return []

        corrected_texts = []
        # 兼容新版 ChatCompletion content 可能为 list 的情况
        if isinstance(response, list):
            joined = []
            for part in response:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    if isinstance(text, str) and text.strip():
                        joined.append(text.strip())
                elif isinstance(part, str):
                    if part.strip():
                        joined.append(part.strip())
            response_text = "\n".join(joined)
        else:
            response_text = str(response)

        lines = response_text.strip().split("\n")

        for line in lines:
            if not line or not line.strip():
                continue

            # 匹配 "1. 文本" 或 "1、文本" 或 "1 文本" 格式
            match = re.match(r'^\d+[\.\、\s]\s*(.+)$', line.strip())
            if match:
                text = match.group(1).strip()
                if text:
                    corrected_texts.append(text)
        
        # 如果解析结果数量不匹配，返回原始行（去掉编号后的内容）
        if len(corrected_texts) != original_count:
            # 尝试更宽松的解析
            corrected_texts = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 去掉可能的编号前缀
                    cleaned = re.sub(r'^\d+[\.\、\s]*', '', line).strip()
                    if cleaned:
                        corrected_texts.append(cleaned)
        
        return corrected_texts
    
    def correct_texts(self, texts: List[str], batch_size: int = 20) -> List[str]:
        """
        校正文本列表
        
        Args:
            texts: 待校正的文本列表
            batch_size: 每批处理的文本数量
            
        Returns:
            List[str]: 校正后的文本列表
        """
        if not texts:
            return texts
        
        corrected_all = []
        
        # 分批处理，避免单次请求过大
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            prompt = self._create_correction_prompt(batch)
            
            try:
                # 调用 AI API
                completion = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                    stream=False
                )
                
                content = completion.choices[0].message.content
                corrected_batch = self._parse_correction_response(content, len(batch))
                
                if len(corrected_batch) > len(batch):
                    corrected_batch = corrected_batch[: len(batch)]
                elif len(corrected_batch) < len(batch):
                    corrected_batch = corrected_batch + batch[len(corrected_batch) :]
                
                if len(corrected_batch) != len(batch):
                    print(f"AI 校正解析失败，使用原文。期望 {len(batch)} 条，得到 {len(corrected_batch)} 条")
                    corrected_batch = batch
                
                corrected_all.extend(corrected_batch)
                
            except Exception as e:
                print(f"AI 校正请求失败: {e}，使用原文")
                corrected_all.extend(batch)
        
        return corrected_all
    
    def correct_srt_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        校正 SRT 字幕段落
        
        Args:
            segments: 字幕段落列表，每个段落包含 line, start_time, end_time, text
            
        Returns:
            List[Dict]: 校正后的字幕段落列表
        """
        if not segments:
            return segments
        
        # 提取文本
        texts = [seg.get("text", "") for seg in segments]
        
        # 校正文本
        corrected_texts = self.correct_texts(texts)
        
        # 更新段落
        corrected_segments = []
        for i, seg in enumerate(segments):
            corrected_seg = seg.copy()
            if i < len(corrected_texts):
                corrected_seg["text"] = corrected_texts[i]
            corrected_segments.append(corrected_seg)
        
        return corrected_segments
    
    def correct_srt_string(self, srt_content: str) -> str:
        """
        校正 SRT 格式的字幕字符串
        
        Args:
            srt_content: SRT 格式的字幕内容
            
        Returns:
            str: 校正后的 SRT 字幕内容
        """
        # 兼容传入 list 的情况
        if isinstance(srt_content, list):
            srt_content = "\n".join(str(x) for x in srt_content)
        if not srt_content or not str(srt_content).strip():
            return srt_content
        
        # 解析 SRT
        segments = self._parse_srt(str(srt_content))
        
        if not segments:
            return srt_content
        
        # 提取并校正文本
        texts = [seg["text"] for seg in segments]
        corrected_texts = self.correct_texts(texts)
        
        # 重建 SRT
        corrected_srt_lines = []
        for i, seg in enumerate(segments):
            corrected_text = corrected_texts[i] if i < len(corrected_texts) else seg["text"]
            corrected_srt_lines.append(f"{seg['line']}\n{seg['time']}\n{corrected_text}\n")
        
        return "\n".join(corrected_srt_lines)
    
    def _parse_srt(self, srt_content: str) -> List[Dict]:
        """
        解析 SRT 字幕内容
        
        Args:
            srt_content: SRT 格式的字幕字符串
            
        Returns:
            List[Dict]: 解析后的字幕段落列表
        """
        segments = []
        blocks = re.split(r'\n\n+', srt_content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    line_num = int(lines[0].strip())
                    time_line = lines[1].strip()
                    text = '\n'.join(lines[2:]).strip()
                    
                    segments.append({
                        "line": line_num,
                        "time": time_line,
                        "text": text
                    })
                except (ValueError, IndexError):
                    continue
        
        return segments


def create_ai_corrector(config: Optional[Dict] = None) -> AICorrector:
    """
    创建 AI 校正器实例
    
    Args:
        config: 配置字典
        
    Returns:
        AICorrector: AI 校正器实例
    """
    return AICorrector(config)


# 便捷函数
def correct_srt(srt_content: str, config: Optional[Dict] = None) -> str:
    """
    便捷函数：校正 SRT 字幕
    
    Args:
        srt_content: SRT 格式的字幕内容
        config: AI 配置（可选）
        
    Returns:
        str: 校正后的 SRT 字幕内容
    """
    corrector = create_ai_corrector(config)
    return corrector.correct_srt_string(srt_content)


def correct_text_list(texts: List[str], config: Optional[Dict] = None) -> List[str]:
    """
    便捷函数：校正文本列表
    
    Args:
        texts: 待校正的文本列表
        config: AI 配置（可选）
        
    Returns:
        List[str]: 校正后的文本列表
    """
    corrector = create_ai_corrector(config)
    return corrector.correct_texts(texts)
