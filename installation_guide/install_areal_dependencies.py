#!/usr/bin/env python3
"""
Install Areal Framework and Dependencies
=======================================

This script installs the Areal framework and other dependencies needed for
enhanced RL algorithms with optimized caching.
"""

import subprocess
import sys
import os


def install_package(package_name: str, pip_name: str = None) -> bool:
    """安装Python包"""
    if pip_name is None:
        pip_name = package_name
    
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pip_name
        ])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False


def check_package_available(package_name: str) -> bool:
    """检查包是否可用"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    """主函数"""
    print("🚀 Installing Areal Framework and Dependencies")
    print("=" * 50)
    
    # 需要安装的包列表
    packages = [
        ("areal", "areal"),  # Areal缓存框架
        ("numpy", "numpy"),  # 数值计算
        ("torch", "torch"),  # PyTorch深度学习框架
        ("psutil", "psutil"),  # 系统监控
        ("matplotlib", "matplotlib"),  # 绘图
        ("seaborn", "seaborn"),  # 统计绘图
        ("plotly", "plotly"),  # 交互式绘图
        ("pandas", "pandas"),  # 数据处理
    ]
    
    # 可选的高级包
    optional_packages = [
        ("redis", "redis"),  # Redis缓存后端
        ("memcached", "python-memcached"),  # Memcached缓存后端
        ("ray", "ray"),  # 分布式计算
        ("dask", "dask"),  # 并行计算
    ]
    
    print("📋 Required packages:")
    for package_name, pip_name in packages:
        if check_package_available(package_name):
            print(f"  ✅ {package_name} (already installed)")
        else:
            success = install_package(package_name, pip_name)
            if not success:
                print(f"  ⚠️  {package_name} installation failed, continuing...")
    
    print("\n📋 Optional packages (for advanced features):")
    for package_name, pip_name in optional_packages:
        if check_package_available(package_name):
            print(f"  ✅ {package_name} (already installed)")
        else:
            print(f"  🔧 {package_name} (optional)")
            response = input(f"    Install {package_name}? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                install_package(package_name, pip_name)
    
    print("\n🔍 Verifying installations...")
    
    # 验证关键包
    critical_packages = ["areal", "numpy"]
    all_available = True
    
    for package in critical_packages:
        if check_package_available(package):
            print(f"  ✅ {package} is available")
        else:
            print(f"  ❌ {package} is NOT available")
            all_available = False
    
    if all_available:
        print("\n🎉 All critical packages installed successfully!")
        print("\n📖 Usage examples:")
        print("  # Run the enhanced RL cache demo")
        print("  python demo/enhanced_rl_cache_demo.py")
        print("  ")
        print("  # Run with specific demo")
        print("  python demo/enhanced_rl_cache_demo.py --demo basic")
        print("  ")
        print("  # Run with custom cache size")
        print("  python demo/enhanced_rl_cache_demo.py --cache-size 20000")
    else:
        print("\n⚠️  Some critical packages are missing.")
        print("Please install them manually or check your Python environment.")
    
    print("\n📚 Documentation:")
    print("  - Areal Framework: https://github.com/areal-framework/areal")
    print("  - Enhanced RL Guide: docs/enhanced_rl_guide.md")
    print("  - Monitoring Guide: docs/monitoring_guide.md")


if __name__ == "__main__":
    main() 