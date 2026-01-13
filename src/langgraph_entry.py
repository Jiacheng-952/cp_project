# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
LangGraph服务器入口文件
专门为LangGraph开发服务器提供图实例
"""

import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
print(f"当前入口文件路径: {current_file}")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(current_file))
print(f"项目根目录: {project_root}")

# 强制确保正确的路径在最前面
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)
print(f"已强制添加到Python路径首位: {project_root}")

print(f"当前Python路径: {sys.path[:3]}...")  # 只显示前3个路径
print(f"当前工作目录: {os.getcwd()}")

# 尝试导入并显示详细错误信息
try:
    from src.graph.builder import build_optag_graph, build_optag_graph_with_memory

    print("✅ 成功导入OptAgent构建函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

    # 显示实际查找的路径
    import src.graph.builder as builder_module

    print(f"实际找到的builder模块路径: {builder_module.__file__}")

    # 检查模块中是否有我们需要的函数
    available_functions = [
        attr for attr in dir(builder_module) if not attr.startswith("_")
    ]
    print(f"builder模块中可用的函数: {available_functions}")
    raise

# 为LangGraph服务器提供图实例
optag = build_optag_graph()
optag_with_memory = build_optag_graph_with_memory()

# 保持与原有配置的兼容性
graph = optag  # 默认图
