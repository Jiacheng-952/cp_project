import os
import time
from docplex.cp.model import CpoModel

def parse_fjspc_data(data_str):
    """解析FJSPC格式的数据 (保持不变)"""
    lines = data_str.strip().split('\n')
    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])
    
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
    
    changeover_times = []
    for i in range(num_machines):
        row = list(map(int, lines[job_line_idx + i].strip().split()))
        changeover_times.append(row)
    
    print("-" * 30)
    print(f"工件总数 (num_jobs): {num_jobs}")
    print(f"机器总数 (num_machines): {num_machines}")
    print(f"转换矩阵行数: {len(changeover_times)}")
    if len(changeover_times) > 0:
        print(f"转换矩阵列数: {len(changeover_times[0])}")
    print("-" * 30)
    return jobs_operations, changeover_times, num_machines

# def solve_fjsp_with_cplex(jobs_operations, changeover_times, num_machines, time_limit=300):
#     """
#     使用 CPLEX CP Optimizer 求解带转换时间的 FJSP
#     """
#     model = CpoModel(name="FJSPC_Solver")
#     num_jobs = len(jobs_operations)

#     # 1. 定义变量
#     op_intervals = {}
#     m_intervals = {}
#     machine_sequences = [[] for _ in range(num_machines)]
#     machine_interval_types = [[] for _ in range(num_machines)] # 记录每个区间对应的 Job 类型

#     for j in range(num_jobs):
#         for o in range(len(jobs_operations[j])):
#             op_var = model.interval_var(name=f"J{j}_O{o}")
#             op_intervals[(j, o)] = op_var
            
#             m_vars = []
#             possible_machines = jobs_operations[j][o]
#             for m, duration in possible_machines.items():
#                 # 创建可选区间
#                 m_var = model.interval_var(optional=True, size=duration, name=f"J{j}_O{o}_M{m}")
#                 m_vars.append(m_var)
#                 m_intervals[(j, o, m)] = m_var
                
#                 # 关键：记录该机器上的区间及其对应的类型（Job ID）
#                 machine_sequences[m].append(m_var)
#                 machine_interval_types[m].append(j) 
            
#             model.add(model.alternative(op_var, m_vars))

#     # 2. 约束条件
#     # (1) 工序顺序约束
#     for j in range(num_jobs):
#         for o in range(len(jobs_operations[j]) - 1):
#             model.add(model.end_before_start(op_intervals[(j, o)], op_intervals[(j, o + 1)]))

def solve_fjsp_with_cplex(jobs_operations, changeover_times, num_machines, time_limit=300):
    model = CpoModel(name="FJSPC_Solver")
    num_jobs = len(jobs_operations)
    matrix_size = len(changeover_times)

    op_intervals = {}
    m_intervals = {}
    machine_sequences = [[] for _ in range(num_machines)]
    machine_interval_types = [[] for _ in range(num_machines)]

    for j in range(num_jobs):
        for o in range(len(jobs_operations[j])):
            op_var = model.interval_var(name=f"J{j}_O{o}")
            op_intervals[(j, o)] = op_var
            
            m_vars = []
            possible_machines = jobs_operations[j][o]
            for m, duration in possible_machines.items():
                m_var = model.interval_var(optional=True, size=duration, name=f"J{j}_O{o}_M{m}")
                m_vars.append(m_var)
                m_intervals[(j, o, m)] = m_var
                machine_sequences[m].append(m_var)
                
                job_type = j % matrix_size 
                machine_interval_types[m].append(job_type)
            
            model.add(model.alternative(op_var, m_vars))

    # --- 必须补上的约束：工序顺序约束 (Precedence) ---
    for j in range(num_jobs):
        for o in range(len(jobs_operations[j]) - 1):
            # 约束：同一工件的下一道工序必须在前一道工序结束后开始
            model.add(model.end_before_start(op_intervals[(j, o)], op_intervals[(j, o + 1)]))

    # --- 机器不重叠与转换时间 ---
    tmat = model.transition_matrix(changeover_times)
    for m in range(num_machines):
        if machine_sequences[m]:
            seq = model.sequence_var(machine_sequences[m], types=machine_interval_types[m])
            model.add(model.no_overlap(seq, tmat))

    # --- 优化目标：最小化所有工序的最大结束时间 ---
    # 改用更稳妥的定义：所有 op_intervals 的结束时间最大值
    makespan = model.max([model.end_of(itv) for itv in op_intervals.values()])
    model.add(model.minimize(makespan))

    # 4. 求解
    print(f"开始求解 (时限: {time_limit}s)...")
    msol = model.solve(TimeLimit=time_limit, LogVerbosity='Normal')

    # 5. 结果解析
    if msol:
        results = []
        for j in range(num_jobs):
            for o in range(len(jobs_operations[j])):
                # 检查哪个机器区间被选中了
                for m in jobs_operations[j][o].keys():
                    itv_val = msol.get_var_solution(m_intervals[(j, o, m)])
                    if itv_val.is_present():
                        results.append({
                            "Job": j + 1,
                            "Operation": o + 1,
                            "Machine": m + 1,
                            "Start": itv_val.get_start(),
                            "Processing": itv_val.get_size(),
                            "End": itv_val.get_end()
                        })
        
        obj_value = msol.get_objective_values()[0]
        solve_time = msol.get_solve_time()


        return results, obj_value, solve_time, msol.get_solve_status()
    else:
        return None, None, None, "No Solution"

def read_fjspc_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def write_solution_to_txt(results, makespan, filename, status):
    # 使用 utf-8-sig 编码，彻底解决 Windows 下记事本乱码问题
    print(f"DEBUG: 正在写入文件 {filename}, Makespan 变量值 = {makespan}")
    with open(filename, 'w', encoding='utf-8-sig') as f:
        f.write(f"makespan: {makespan}\n")
        
        if results is None:
            f.write(f"状态: {status}\n")
            return
        
        f.write("机器\t工件\t工序\t开始时间\t加工时间\t结束时间\n")
        
        # 排序后输出
        sorted_results = sorted(results, key=lambda x: (x["Machine"], x["Start"]))
        for r in sorted_results:
            line = f"{r['Machine']}\t{r['Job']}\t{r['Operation']}\t{r['Start']}\t{r['Processing']}\t{r['End']}\n"
            f.write(line)
    

def process_single_file(file_id):
    """处理单个文件逻辑 (适配CPLEX)"""
    file_num_str = f"{file_id:03d}"
    input_file_path = f"JSSP_Dts/fjspc/data/generate_MK{file_num_str}.fjsc"
    output_dir = "JSSP_Dts/fjspc/results"
    output_file_path = f"{output_dir}/MK{file_num_str}_solution.txt"
    os.makedirs(output_dir, exist_ok=True)

    try:
        data_str = read_fjspc_file(input_file_path)
    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 不存在")
        return False, file_num_str, None, None, None

    jobs_operations, changeover_times, num_machines = parse_fjspc_data(data_str)
    
    # 调用 CPLEX 求解器
    results, makespan, runtime, status = solve_fjsp_with_cplex(
        jobs_operations, 
        changeover_times, 
        num_machines,
        time_limit=600
    )
    
    if results:
        write_solution_to_txt(results, makespan, output_file_path, status)
        return True, file_num_str, makespan, runtime, status
    else:
        with open(output_file_path, 'w') as f:
            f.write(f"makespan: 无解\n求解状态: {status}\n")
        return False, file_num_str, None, runtime, status

def main(start_id = 2, end_id = 2):
    """主函数 (基本保持不变)"""
    # start_id = 2
    # end_id = 2
    summary = []

# 设定汇总文件路径
    # 这里建议使用 process_single_file 中定义的 output_dir，或者手动指定
    summary_dir = "JSSP_Dts/fjspc/results"
    summary_file_path = os.path.join(summary_dir, "summary.txt")
    
    # 确保目录存在
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. 初始化汇总文件，写入表头（使用 utf-8-sig 防止乱码）
    with open(summary_file_path, 'w', encoding='utf-8-sig') as f_sum:
        f_sum.write(f"{'文件编号':<10}\t{'状态':<10}\t{'Makespan':<15}\t{'求解时间(秒)':<15}\t{'状态码'}\n")
        f_sum.write("-" * 70 + "\n")

    print(f"开始批量求解，汇总将保存至: {summary_file_path}")
    summary = []

    # 2. 循环处理文件
    for file_id in range(start_id, end_id + 1):
        success, file_num, makespan, runtime, status = process_single_file(file_id)
        
        # 整理数据
        item = {
            "文件编号": file_num,
            "状态": "成功" if success else "失败",
            "Makespan": makespan if makespan else "无解",
            "求解时间(秒)": f"{runtime:.2f}" if runtime else "N/A",
            "状态码": status
        }
        summary.append(item)
        
        # 3. 实时追加到 summary.txt 文件
        with open(summary_file_path, 'a', encoding='utf-8-sig') as f_sum:
            line = f"{item['文件编号']:<10}\t{item['状态']:<10}\t{str(item['Makespan']):<15}\t{item['求解时间(秒)']:<15}\t{item['状态码']}\n"
            f_sum.write(line)
        
        # 同时在控制台打印，方便查看进度
        print(f"已完成: {file_num} | 状态: {item['状态']} | Makespan: {item['Makespan']}")

    # 4. 最后在控制台输出总览
    print("\n" + "="*30)
    print("批量处理完成！汇总报告预览:")
    print(f"{'文件编号':<10} {'状态':<10} {'Makespan':<15} {'时间(s)':<15}")
    for item in summary:
        print(f"{item['文件编号']:<10} {item['状态']:<10} {str(item['Makespan']):<15} {item['求解时间(秒)']:<15}")
if __name__ == "__main__":
    main(start_id = 1, end_id = 50)