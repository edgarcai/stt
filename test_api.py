#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 API 接口
"""

import os
import requests
import tempfile

def test_api():
    """测试 API 接口"""
    print("=== 测试 API 接口 ===")

    # 服务器地址
    url = "http://127.0.0.1:9977/api"

    # 创建一个临时音频文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(b"\x00" * 1024)
        temp_file_path = temp_file.name

    try:
        # 请求参数
        files = {"file": open(temp_file_path, "rb")}
        data = {"language": "zh", "model": "tiny", "response_format": "json"}

        # 发送请求
        response = requests.request("POST", url, timeout=600, data=data, files=files)

        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            print("✓ API 接口测试成功")
            return True
        else:
            print("✗ API 接口测试失败")
            return False

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    finally:
        # 删除临时文件
        os.remove(temp_file_path)

def main():
    """主测试函数"""
    print("开始测试 API 接口...\n")

    test_api()

if __name__ == "__main__":
    main()
