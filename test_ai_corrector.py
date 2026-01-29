#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 错别字校正功能测试脚本

测试 ai_corrector.py 模块的各项功能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stslib.ai_corrector import AICorrector, correct_srt, correct_text_list


def test_text_correction():
    """测试纯文本校正"""
    print("=" * 50)
    print("测试 1: 纯文本校正")
    print("=" * 50)
    
    # 模拟语音识别产生的错别字文本
    texts = [
        "今天天起很好，我们去公园玩吧",  # 天起 -> 天气
        "这个问提很难解决",  # 问提 -> 问题
        "他是一个非长优秀的人",  # 非长 -> 非常
        "请帮我查一下这个子料",  # 子料 -> 资料
        "我已经在哪里等了很久了",  # 在哪里 -> 在那里
    ]
    
    print("原始文本:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    print("\n正在调用 AI 校正...")
    corrector = AICorrector()
    corrected = corrector.correct_texts(texts)
    
    print("\n校正后文本:")
    for i, text in enumerate(corrected):
        print(f"  {i+1}. {text}")
    
    print()


def test_srt_correction():
    """测试 SRT 字幕校正"""
    print("=" * 50)
    print("测试 2: SRT 字幕校正")
    print("=" * 50)
    
    srt_content = """1
00:00:00,000 --> 00:00:03,500
今天的天起真的很不错

2
00:00:03,500 --> 00:00:07,200
我们一起去公元散步吧

3
00:00:07,200 --> 00:00:11,000
这个问提确实很复杂"""
    
    print("原始 SRT:")
    print(srt_content)
    
    print("\n正在调用 AI 校正...")
    corrected_srt = correct_srt(srt_content)
    
    print("\n校正后 SRT:")
    print(corrected_srt)
    print()


def test_json_segments_correction():
    """测试 JSON 格式字幕段落校正"""
    print("=" * 50)
    print("测试 3: JSON 格式字幕校正")
    print("=" * 50)
    
    segments = [
        {"line": 1, "start_time": "00:00:00,000", "end_time": "00:00:03,500", "text": "欢应收看本期节目"},
        {"line": 2, "start_time": "00:00:03,500", "end_time": "00:00:07,200", "text": "今天我们来聊一聊人工只能"},
        {"line": 3, "start_time": "00:00:07,200", "end_time": "00:00:11,000", "text": "这是一个非长有趣的话提"},
    ]
    
    print("原始 JSON 段落:")
    for seg in segments:
        print(f"  Line {seg['line']}: {seg['text']}")
    
    print("\n正在调用 AI 校正...")
    corrector = AICorrector()
    corrected_segments = corrector.correct_srt_segments(segments)
    
    print("\n校正后 JSON 段落:")
    for seg in corrected_segments:
        print(f"  Line {seg['line']}: {seg['text']}")
    print()


def test_api_connection():
    """测试 API 连接"""
    print("=" * 50)
    print("测试 0: API 连接测试")
    print("=" * 50)
    
    try:
        corrector = AICorrector()
        result = corrector.correct_texts(["这是一个测试"])
        print(f"API 连接成功！")
        print(f"测试结果: {result}")
        return True
    except Exception as e:
        print(f"API 连接失败: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AI 错别字校正功能测试")
    print("=" * 50 + "\n")
    
    # 测试 API 连接
    if not test_api_connection():
        print("\nAPI 连接失败，请检查配置后重试。")
        sys.exit(1)
    
    print("\n")
    
    # 运行各项测试
    try:
        test_text_correction()
        test_srt_correction()
        test_json_segments_correction()
        
        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试过程出错: {e}")
        import traceback
        traceback.print_exc()
