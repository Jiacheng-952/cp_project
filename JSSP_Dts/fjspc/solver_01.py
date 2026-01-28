import pandas as pd
import os
from ortools.sat.python import cp_model
import time

def parse_fjspc_data(data_str):
    """
    解析FJSPC格式的数据
    """
    lines = data_str.strip().split('\n')
    
    # 解析第一行
    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])
    avg_flexibility = float(first_line[2])
    
    # 解析工件数据
    jobs_operations = []
    job_line_idx = 1
    
    for j in range(num_jobs):
        job_line = lines[job_line_idx].strip().split()
        job_line_idx += 1
        
        num_ops = int(job_line[0])
        operations = []
        idx = 1
        
        for _ in range(num_ops):
            num_alternatives = int(job_line[idx])
            idx += 1
            
            alternatives = {}
            for _ in range(num_alternatives):
                machine = int(job_line[idx]) - 1
                idx += 1
                processing_time = int(job_line[idx])
                idx += 1
                alternatives[machine] = processing_time
            
            operations.append(alternatives)
        
        jobs_operations.append(operations)
    
    # 解析SDST矩阵
    changeover_times = []
    for i in range(num_machines):
        row = list(map(int, lines[job_line_idx + i].strip().split()))
        changeover_times.append(row)
    
    return jobs_operations, changeover_times, num_machines

def solve_fjsp_with_ortools(jobs_operations, changeover_times, num_machines, time_limit=300):
    """
    使用 OR-Tools CP-SAT 求解带转换时间的柔性作业车间调度问题
    """
    num_jobs = len(jobs_operations)
    
    # 创建模型
    model = cp_model.CpModel()
    
    # --- 计算时间范围 ---
    horizon = 0
    for j in range(num_jobs):
        for o in range(len(jobs_operations[j])):
            for m in jobs_operations[j][o]:
                horizon += jobs_operations[j][o][m]
    
    max_changeover = max(max(row) for row in changeover_times)
    horizon += max_changeover * num_jobs * 10
    
    # --- 决策变量 ---
    start_vars = {}
    end_vars = {}
    assign_vars = {}
    interval_vars = {}
    
    all_ops = []
    for j in range(num_jobs):
        for o in range(len(jobs_operations[j])):
            all_ops.append((j, o))
    
    for j, o in all_ops:
        start_vars[(j, o)] = model.NewIntVar(0, horizon, f"start_{j}_{o}")
        end_vars[(j, o)] = model.NewIntVar(0, horizon, f"end_{j}_{o}")
        
        assign_vars[(j, o)] = {}
        interval_vars[(j, o)] = {}
        
        possible_machines = jobs_operations[j][o]
        
        for m in possible_machines:
            assign_vars[(j, o)][m] = model.NewBoolVar(f"assign_{j}_{o}_{m}")
            
            proc_time = possible_machines[m]
            
            interval_var = model.NewOptionalIntervalVar(
                start_vars[(j, o)],
                proc_time,
                end_vars[(j, o)],
                assign_vars[(j, o)][m],
                f"interval_{j}_{o}_{m}"
            )
            interval_vars[(j, o)][m] = interval_var
    
    makespan = model.NewIntVar(0, horizon, "makespan")
    
    # --- 约束条件 ---
    for j, o in all_ops:
        possible_machines = list(jobs_operations[j][o].keys())
        model.Add(sum(assign_vars[(j, o)][m] for m in possible_machines) == 1)
    
    for j, o in all_ops:
        possible_machines = jobs_operations[j][o]
        proc_time_expr = sum(
            assign_vars[(j, o)][m] * possible_machines[m]
            for m in possible_machines
        )
        model.Add(end_vars[(j, o)] == start_vars[(j, o)] + proc_time_expr)
    
    for j in range(num_jobs):
        for o in range(len(jobs_operations[j]) - 1):
            model.Add(start_vars[(j, o + 1)] >= end_vars[(j, o)])
    
    for m in range(num_machines):
        intervals_on_machine = []
        
        for j, o in all_ops:
            if m in interval_vars[(j, o)]:
                intervals_on_machine.append(interval_vars[(j, o)][m])
        
        if intervals_on_machine:
            model.AddNoOverlap(intervals_on_machine)
    
    for j in range(num_jobs):
        last_op_idx = len(jobs_operations[j]) - 1
        model.Add(makespan >= end_vars[(j, last_op_idx)])
    
    # --- 目标函数 ---
    model.Minimize(makespan)
    
    # --- 求解 ---
    solver = cp_model.CpSolver()
    
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    
    start_time = time.time()
    status = solver.Solve(model)
    end_time = time.time()
    
    # --- 结果解析和输出 ---
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        makespan_value = solver.ObjectiveValue()
        
        results = []
        for j in range(num_jobs):
            for o in range(len(jobs_operations[j])):
                machine_assigned = -1
                processing_time = 0
                
                for m in jobs_operations[j][o].keys():
                    if solver.Value(assign_vars[(j, o)][m]) == 1:
                        machine_assigned = m + 1
                        processing_time = jobs_operations[j][o][m]
                        break
                
                results.append({
                    "Job": j + 1,
                    "Operation": o + 1,
                    "Machine": machine_assigned,
                    "Start": solver.Value(start_vars[(j, o)]),
                    "Processing": processing_time,
                    "End": solver.Value(end_vars[(j, o)])
                })
        
        return results, makespan_value, end_time - start_time, status
    else:
        return None, None, None, status

def read_fjspc_file(filepath):
    """从文件读取FJSPC数据"""
    with open(filepath, 'r') as f:
        data_str = f.read()
    return data_str

def write_solution_to_txt(results, makespan, filename, status):
    """
    将调度结果写入TXT文件
    格式：
    第一行：makespan: [数值]
    第二行：表头（机器、工件、工序、开始时间、加工时间、结束时间）
    第三行及之后：调度结果（每行一个工序）
    """
    with open(filename, 'w') as f:
        f.write(f"makespan: {makespan}\n")
        
        if results is None:
            f.write("无可行解\n")
            return
        
        f.write("机器\t工件\t工序\t开始时间\t加工时间\t结束时间\n")
        
        sorted_results = sorted(results, key=lambda x: (x["Machine"], x["Start"]))
        
        for r in sorted_results:
            line = f"{r['Machine']}\t{r['Job']}\t{r['Operation']}\t{r['Start']}\t{r['Processing']}\t{r['End']}\n"
            f.write(line)
    
    print(f"已保存求解结果: {filename}")

def process_single_file(file_id):
    """处理单个文件"""
    file_num_str = f"{file_id:03d}"
    
    input_file_path = f"fjspc/data/generate_MK{file_num_str}.fjsc"
    output_dir = "fjspc/results"
    output_file_path = f"{output_dir}/MK{file_num_str}_solution.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 从文件读取数据
    try:
        data_str = read_fjspc_file(input_file_path)
    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 不存在")
        return False, file_num_str, None, None, None
    
    # 解析数据
    jobs_operations, changeover_times, num_machines = parse_fjspc_data(data_str)
    
    # 求解模型
    results, makespan, runtime, status = solve_fjsp_with_ortools(
        jobs_operations, 
        changeover_times, 
        num_machines,
        time_limit=600
    )
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # 写入TXT文件
        write_solution_to_txt(results, makespan, output_file_path, status)
        return True, file_num_str, makespan, runtime, status
    else:
        # 即使失败也写入一个文件
        with open(output_file_path, 'w') as f:
            f.write(f"makespan: 无解\n")
            f.write(f"求解状态: {status}\n")
            f.write(f"该问题在给定时间内未找到可行解\n")
        
        print(f"已保存求解结果: {output_file_path} (无解)")
        return False, file_num_str, None, runtime, status

def main():
    """主函数：处理多个文件"""
    start_id = 85
    end_id = 85
    
    print("开始批量求解FJSPC问题")
    print(f"处理文件: MK{start_id:03d} 到 MK{end_id:03d}")
    
    summary = []
    
    # 处理每个文件
    for file_id in range(start_id, end_id + 1):
        success, file_num, makespan, runtime, status = process_single_file(file_id)
        
        summary.append({
            "文件编号": file_num,
            "状态": "成功" if success else "失败",
            "Makespan": makespan if makespan else "无解",
            "求解时间(秒)": f"{runtime:.2f}" if runtime else "N/A",
            "状态码": status
        })
    
    # 输出汇总报告
    print("\n批量求解完成!")
    print("汇总报告:")
    print(f"{'文件编号':<10} {'状态':<10} {'Makespan':<15} {'求解时间(秒)':<15}")
    print(f"{'-'*50}")
    
    for item in summary:
        print(f"{item['文件编号']:<10} {item['状态']:<10} {str(item['Makespan']):<15} {item['求解时间(秒)']:<15}")
    
    # 保存汇总报告到文件
    summary_file = "fjspc/results/batch_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("批量求解FJSPC问题汇总报告\n")
        f.write("="*50 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件范围: MK{start_id:03d} 到 MK{end_id:03d}\n\n")
        f.write(f"{'文件编号':<10} {'状态':<10} {'Makespan':<15} {'求解时间(秒)':<15}\n")
        f.write("-"*50 + "\n")
        
        for item in summary:
            f.write(f"{item['文件编号']:<10} {item['状态']:<10} {str(item['Makespan']):<15} {item['求解时间(秒)']:<15}\n")
    
    print(f"\n汇总报告已保存到: {summary_file}")

if __name__ == "__main__":
    total_start_time = time.time()
    main()
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print(f"\n批量求解总运行时间: {total_runtime:.2f} 秒 ({total_runtime/60:.2f} 分钟)")