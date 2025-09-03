#!/usr/bin/env python3
"""
运行真实模型测试的脚本

这个脚本用于运行使用真实 DNA 预测模型的测试，而不是模拟模型。
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """运行真实模型测试"""
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧬 运行真实模型测试")
    print("=" * 50)
    
    # 测试文件列表
    test_files = [
        "dnallm/mcp/tests/test_real_models.py",
        "dnallm/mcp/tests/test_integration.py",
        "dnallm/mcp/tests/test_performance.py"
    ]
    
    # 运行选项
    pytest_args = [
        "-v",  # 详细输出
        "-s",  # 不捕获输出
        "-m", "real_model",  # 只运行标记为 real_model 的测试
        "--tb=short",  # 简短的错误回溯
        "--durations=10",  # 显示最慢的10个测试
        "--maxfail=3",  # 最多失败3个测试后停止
    ]
    
    # 添加测试文件
    pytest_args.extend(test_files)
    
    print(f"运行命令: pytest {' '.join(pytest_args)}")
    print()
    
    try:
        # 运行测试
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + pytest_args,
            cwd=project_root,
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n✅ 所有真实模型测试通过!")
        else:
            print(f"\n❌ 测试失败，退出码: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ 运行测试时出错: {e}")
        return 1


def run_specific_test(test_name: str):
    """运行特定的测试"""
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"🧬 运行特定测试: {test_name}")
    print("=" * 50)
    
    pytest_args = [
        "-v",
        "-s",
        "-m", "real_model",
        "--tb=short",
        f"dnallm/mcp/tests/test_real_models.py::{test_name}"
    ]
    
    print(f"运行命令: pytest {' '.join(pytest_args)}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + pytest_args,
            cwd=project_root,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ 运行测试时出错: {e}")
        return 1


def run_quick_tests():
    """运行快速测试（跳过慢速测试）"""
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧬 运行快速真实模型测试")
    print("=" * 50)
    
    pytest_args = [
        "-v",
        "-s",
        "-m", "real_model and not slow",  # 只运行快速的真实模型测试
        "--tb=short",
        "--durations=5",
        "dnallm/mcp/tests/test_real_models.py"
    ]
    
    print(f"运行命令: pytest {' '.join(pytest_args)}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + pytest_args,
            cwd=project_root,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ 运行测试时出错: {e}")
        return 1


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行真实模型测试")
    parser.add_argument(
        "--test", 
        help="运行特定测试",
        choices=[
            "test_real_model_loading",
            "test_real_model_prediction", 
            "test_real_model_batch_prediction",
            "test_real_model_with_task_router",
            "test_real_model_pool",
            "test_multiple_real_models",
            "test_real_model_with_sse",
            "test_real_model_performance"
        ]
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="只运行快速测试"
    )
    
    args = parser.parse_args()
    
    if args.test:
        return run_specific_test(args.test)
    elif args.quick:
        return run_quick_tests()
    else:
        return run_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
