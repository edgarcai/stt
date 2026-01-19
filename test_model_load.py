#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stslib.whisper_adapter import create_whisper_model
from stslib.funasr_adapter import FunASRModelAdapter, is_funasr_model


def test_whisper_model_load():
    """测试 Whisper 模型加载"""
    print("=== 测试 Whisper 模型加载 ===")
    try:
        # 测试创建 tiny 模型
        model = create_whisper_model("tiny", device_type="cpu")
        print("✓ 成功创建 Whisper 模型适配器")

        print(f"  - 模型类型: {type(model)}")
        print(f"  - 后端: {model._backend}")

        model.unload()
        print("✓ 成功卸载模型")

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True


def test_funasr_model_load():
    """测试 FunASR 模型加载"""
    print("\n=== 测试 FunASR 模型加载 ===")
    try:
        # 测试创建 FunASR 模型
        if is_funasr_model("fun-asr-nano"):
            model = FunASRModelAdapter("fun-asr-nano", device_type="cpu")
            print("✓ 成功创建 FunASR 模型适配器")

            print(f"  - 模型类型: {type(model)}")

            model.unload()
            print("✓ 成功卸载模型")

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True


def test_model_directory_config():
    """测试模型目录配置"""
    print("\n=== 测试模型目录配置 ===")
    from stslib.cfg import MODEL_DIR

    try:
        # 检查模型目录是否存在
        if os.path.exists(MODEL_DIR):
            print(f"✓ 模型目录存在: {MODEL_DIR}")
        else:
            print(f"⚠ 模型目录不存在，将在首次运行时创建: {MODEL_DIR}")

        # 检查配置的模型目录是否正确
        if "models" in MODEL_DIR.lower():
            print("✓ 模型目录配置正确")
        else:
            print("✗ 模型目录配置不正确")
            return False

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    return True


def main():
    """主测试函数"""
    print("开始测试模型加载功能...\n")

    results = []
    results.append(("Whisper 模型加载", test_whisper_model_load()))
    results.append(("FunASR 模型加载", test_funasr_model_load()))
    results.append(("模型目录配置", test_model_directory_config()))

    print("\n" + "="*30)
    print("测试结果汇总")
    print("="*30)

    all_passed = True
    for test_name, passed in results:
        status = "✓ 成功" if passed else "✗ 失败"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n所有测试通过！模型加载路径配置正确。")
    else:
        print("\n部分测试失败，请检查相关配置。")
        sys.exit(1)


if __name__ == "__main__":
    main()
