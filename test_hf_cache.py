#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Hugging Face 缓存目录是否正确设置
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stslib.whisper_adapter import _MODELS_DIR
from huggingface_hub import snapshot_download

def test_hf_cache_env():
    """测试 HF_HUB_CACHE 环境变量是否正确设置"""
    print("=== 测试 HF_HUB_CACHE 环境变量 ===")

    print(f"当前进程的 HF_HUB_CACHE 值: {os.environ.get('HF_HUB_CACHE', '未设置')}")
    print(f"whisper_adapter 中的 _MODELS_DIR 值: {_MODELS_DIR}")

    assert os.environ.get('HF_HUB_CACHE') == _MODELS_DIR, "HF_HUB_CACHE 未正确设置"
    print("✓ 环境变量设置正确")

def test_snapshot_download_cache_dir():
    """测试 snapshot_download 是否使用了正确的缓存目录"""
    print("\n=== 测试 snapshot_download 缓存目录 ===")

    try:
        # 下载一个小模型来测试
        repo_id = "mlx-community/whisper-tiny"
        download_path = snapshot_download(repo_id)

        print(f"模型下载路径: {download_path}")

        if _MODELS_DIR in download_path:
            print("✓ snapshot_download 正确使用了项目的 models 目录")
        else:
            print(f"✗ snapshot_download 使用了默认目录: {download_path}")
            print(f"期望的目录前缀: {_MODELS_DIR}")

            # 尝试使用 cache_dir 参数
            print("\n尝试使用 cache_dir 参数:")
            download_path_with_cache = snapshot_download(repo_id, cache_dir=_MODELS_DIR)
            print(f"使用 cache_dir 参数的下载路径: {download_path_with_cache}")

            if _MODELS_DIR in download_path_with_cache:
                print("✓ 使用 cache_dir 参数时正确")
            else:
                print("✗ 使用 cache_dir 参数时也不正确")

            # 清理下载的模型
            import shutil
            import re
            match = re.match(r'(.*?)snapshots.*', download_path_with_cache)
            if match:
                cache_dir = match.group(1)
                print(f"清理模型缓存: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)

            return False

        # 清理下载的模型
        import shutil
        import re
        match = re.match(r'(.*?)snapshots.*', download_path)
        if match:
            cache_dir = match.group(1)
            print(f"清理模型缓存: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始测试 Hugging Face 缓存目录设置...\n")

    results = []
    results.append(("环境变量设置", test_hf_cache_env()))
    results.append(("snapshot_download 缓存目录", test_snapshot_download_cache_dir()))

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
        print("\n所有测试通过！Hugging Face 缓存目录配置正确。")
    else:
        print("\n部分测试失败，需要修复模型下载路径配置。")
        sys.exit(1)

if __name__ == "__main__":
    main()
