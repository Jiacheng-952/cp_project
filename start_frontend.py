#!/usr/bin/env python3
"""
前端界面启动脚本
启动一个简单的HTTP服务器来提供前端界面
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
from pathlib import Path

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "frontend"), **kwargs)
    
    def end_headers(self):
        # 添加CORS头信息
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # 处理预检请求
        self.send_response(200)
        self.end_headers()

def start_frontend_server(port=3002):
    """启动前端HTTP服务器"""
    with socketserver.TCPServer(("", port), SimpleHTTPRequestHandler) as httpd:
        print(f"🚀 前端服务器已启动")
        print(f"📁 服务目录: {Path(__file__).parent / 'frontend'}")
        print(f"🌐 访问地址: http://localhost:{port}")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止")

def check_backend_server():
    """检查后端服务器状态"""
    import requests
    
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        print("✅ 后端服务器运行正常 (localhost:8000)")
        return True
    except requests.exceptions.RequestException as e:
        print("❌ 后端服务器连接失败")
        print("   请确保后端服务器正在运行:")
        print("   python server.py")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 OptAgent 前端界面启动器")
    print("=" * 50)
    
    # 检查前端文件是否存在
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print("❌ 前端目录不存在，请检查项目结构")
        return
    
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        print("❌ 前端界面文件不存在")
        return
    
    print("📁 前端文件检查完成")
    
    # 检查后端服务器
    print("\n🔍 检查后端服务器状态...")
    backend_ok = check_backend_server()
    
    if not backend_ok:
        print("\n⚠️  警告: 后端服务器未运行")
        print("   前端界面可以正常显示，但无法与后端API交互")
        print("   请先启动后端服务器: python server.py")
    
    # 启动前端服务器 - 支持环境变量和默认值
    import os
    frontend_port = int(os.environ.get('FRONTEND_PORT', 3002))
    
    print(f"\n🌐 启动前端服务器 (端口 {frontend_port})...")
    
    # 在后台线程中启动服务器
    server_thread = threading.Thread(target=start_frontend_server, args=(frontend_port,))
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    
    # 自动打开浏览器
    try:
        webbrowser.open(f'http://localhost:{frontend_port}')
        print("✅ 浏览器已自动打开")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print(f"   请手动访问: http://localhost:{frontend_port}")
    
    print("\n📋 使用说明:")
    print("1. 在上方输入框输入优化问题描述")
    print("2. 点击'求解优化问题'按钮")
    print("3. 查看右侧的优化结果")
    print("4. 可以使用示例问题快速测试")
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 程序已退出")

if __name__ == "__main__":
    main()