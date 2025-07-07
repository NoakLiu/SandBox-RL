#!/usr/bin/env python3
"""
SandGraph 可视化依赖安装脚本

安装matplotlib和networkx以启用可视化功能
"""

import subprocess
import sys

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False

def main():
    """主安装函数"""
    print("🎨 SandGraph 可视化依赖安装")
    print("=" * 50)
    
    packages = [
        "matplotlib>=3.5.0",
        "networkx>=2.8.0",
        "pillow>=8.0.0"  # 用于动画保存
    ]
    
    success_count = 0
    for package in packages:
        print(f"\n📦 安装 {package}...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 50)
    if success_count == len(packages):
        print("🎉 所有可视化依赖安装完成！")
        print("现在可以运行 python demo/sandbox_optimization.py 来体验完整的可视化功能")
    else:
        print(f"⚠️  {len(packages) - success_count} 个包安装失败")
        print("请检查网络连接或手动安装失败的包")
    
    print("\n📋 安装的功能:")
    print("   • matplotlib: 图表绘制和可视化")
    print("   • networkx: 图结构分析和布局")
    print("   • pillow: 动画和图像处理")

if __name__ == "__main__":
    main() 