import sys
import os
import time
import asyncio
import csv
from datetime import datetime
from collections import defaultdict

# === 配置 ===
NUM_RUNS = 5  # 每个用例测试 5 遍
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# === 环境路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# === 导入数据 ===
try:
    from dataset import TEST_EXAMPLES
except ImportError:
    print("❌ 请确保 benchmark/dataset.py 文件存在")
    sys.exit(1)

# === 导入架构构建器 ===
try:
    from src.graph.builder import build_optag_graph as build_legacy_graph
except ImportError:
    build_legacy_graph = None
    print("⚠️ 旧架构 (Legacy) 未找到，将跳过。")

try:
    # 注意这里使用了相对引用的修正，确保 graph.py 和 nodes.py 已经改好了
    from src.neuro_optagent.graph import build_optagent as build_neuro_graph
except ImportError:
    build_neuro_graph = None
    print("⚠️ 新架构 (Neuro-Symbolic) 未找到，将跳过。")

# === 核心运行器 ===
async def run_profiled_graph(graph, graph_name, case_id, prompt, run_idx):
    """
    运行图并记录节点耗时 (带实时打印)
    """
    if not graph:
        return []

    print(f"   🚩 Run {run_idx}/{NUM_RUNS} Started...")
    
    inputs = {
        "messages": [{"role": "user", "content": prompt}], 
        "problem_statement": prompt,                       
        "correction_count": 0,
        "max_corrections": 3 # Legacy 架构重试次数
    }
    
    timeline = []
    start_time = time.time()
    last_checkpoint = start_time
    status = "SUCCESS"
    error_msg = ""
    
    try:
        # stream_mode="updates" 返回每个节点完成后的状态增量
        # subgraphs=True 确保我们能捕捉到嵌套图内部的节点（例如 Legacy 内部的节点）
        async for event in graph.astream(inputs, stream_mode="updates", subgraphs=True):
            current_time = time.time()
            
            # event 格式通常是: (namespace, {node_name: update}) 或者直接 {node_name: update}
            # 我们通过解析来获取节点名
            data = event
            if isinstance(event, tuple):
                # 处理子图事件 (namespace, chunk)
                data = event[1]
            
            if isinstance(data, dict):
                for node_name, state_update in data.items():
                    duration = current_time - last_checkpoint
                    
                    # 实时打印：告诉用户跑到了哪里
                    print(f"      ⏱️  Node [{node_name}] finished ({duration:.2f}s)")
                    
                    timeline.append({
                        "case_id": case_id,
                        "run_index": run_idx,
                        "architecture": graph_name,
                        "node": node_name,
                        "duration_seconds": round(duration, 4),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "status": "SUCCESS",
                        "error_msg": ""
                    })
                    last_checkpoint = current_time
                
    except Exception as e:
        status = "ERROR"
        error_msg = str(e)
        print(f"      ❌ Error encountered: {e}")

    total_time = time.time() - start_time
    print(f"   🏁 Run {run_idx} Completed in {total_time:.2f}s\n")
    
    # 记录总耗时 (E2E)
    timeline.append({
        "case_id": case_id,
        "run_index": run_idx,
        "architecture": graph_name,
        "node": "TOTAL_E2E",
        "duration_seconds": round(total_time, 4),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": status,
        "error_msg": error_msg
    })
    
    return timeline

# === 主程序 ===
async def main():
    print("="*60)
    print(f"🚀 Robustness Benchmark (Sequential Batches: {NUM_RUNS} runs/arch)")
    print("="*60)
    
    print("Building Graphs...")
    legacy_agent = build_legacy_graph() if build_legacy_graph else None
    neuro_agent = build_neuro_graph() if build_neuro_graph else None
    
    all_records = []
    summary_data = defaultdict(lambda: defaultdict(list))
    
    # 遍历测试用例
    for case_id, prompt in TEST_EXAMPLES.items():
        print(f"\n" + "="*40)
        print(f"📝 TestCase: {case_id}")
        print("="*40)
        
        # === 1. 批量运行 Legacy (Run 1-5) ===
        if legacy_agent:
            print(f"\n📦 [Batch Testing] Legacy Architecture")
            print("-" * 30)
            for i in range(1, NUM_RUNS + 1):
                recs = await run_profiled_graph(legacy_agent, "Legacy", case_id, prompt, i)
                all_records.extend(recs)
                # 收集摘要数据
                total_node = next((r for r in recs if r['node'] == 'TOTAL_E2E'), None)
                if total_node and total_node['status'] == 'SUCCESS':
                    summary_data[case_id]['Legacy'].append(total_node['duration_seconds'])
        
        # === 2. 批量运行 Neuro-Symbolic (Run 1-5) ===
        if neuro_agent:
            print(f"\n🧠 [Batch Testing] Neuro-Symbolic Architecture")
            print("-" * 30)
            for i in range(1, NUM_RUNS + 1):
                recs = await run_profiled_graph(neuro_agent, "Neuro-Symbolic", case_id, prompt, i)
                all_records.extend(recs)
                # 收集摘要数据
                total_node = next((r for r in recs if r['node'] == 'TOTAL_E2E'), None)
                if total_node and total_node['status'] == 'SUCCESS':
                    summary_data[case_id]['Neuro'].append(total_node['duration_seconds'])

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_sequential_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    fieldnames = ["case_id", "run_index", "architecture", "node", "duration_seconds", "timestamp", "status", "error_msg"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
            
    print("\n" + "="*60)
    print(f"💾 Raw Data Saved: {csv_path}")
    print("="*60)
    
    # 打印对比摘要
    print("\n📊 Average Latency Summary (Avg of 5 runs):")
    print(f"{'Case ID':<25} | {'Legacy Avg(s)':<15} | {'Neuro Avg(s)':<15} | {'Diff'}")
    print("-" * 75)
    
    for case_id in TEST_EXAMPLES.keys():
        leg_times = summary_data[case_id].get('Legacy', [])
        neuro_times = summary_data[case_id].get('Neuro', [])
        
        leg_avg = f"{sum(leg_times)/len(leg_times):.2f}" if leg_times else "N/A"
        neuro_avg = f"{sum(neuro_times)/len(neuro_times):.2f}" if neuro_times else "N/A"
        
        diff_str = "-"
        if leg_times and neuro_times:
            diff = (sum(neuro_times)/len(neuro_times)) - (sum(leg_times)/len(leg_times))
            diff_str = f"{diff:+.2f}s"
        
        print(f"{case_id:<25} | {leg_avg:<15} | {neuro_avg:<15} | {diff_str}")

if __name__ == "__main__":
    asyncio.run(main())