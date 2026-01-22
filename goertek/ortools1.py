import json
import collections
from ortools.sat.python import cp_model
from datetime import datetime

def solve_factory_scheduling(json_file_path):
    # 1. 加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_jobs = raw_data.get('data', {})
    resources_def = raw_data.get('resources', {}) # 假设文件底部有resources定义，如果没有需从工序中提取
    
    # 如果 JSON 中没有显式的 resources 列表，我们需要从工序中提取所有出现的资源 ID
    all_devices = set()
    
    # 预处理：整理工件、工序和资源信息
    # 结构: jobs[part_id] -> list of stages (sorted by order)
    jobs = []
    
    # 解析日期工具函数
    def parse_time(date_str):
        if not date_str:
            return 0
        try:
            # 假设基准时间为当前或某个固定时间，这里简化处理，将日期字符串转换为相对于基准的分钟数或秒数
            # 实际工程中通常计算相对于项目开始时间的时间差
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            base_dt = datetime(2024, 1, 1) # 示例基准时间
            return int((dt - base_dt).total_seconds() / 60) # 转为分钟
        except:
            return 0

    print("正在解析数据...")
    
    # 提取所有出现的具体设备ID
    for part_id, part_info in data_jobs.items():
        stages = part_info.get('stages', {})
        for stage_id, stage_info in stages.items():
            res_ids = stage_info.get('resources_id')
            if res_ids:
                for rid in res_ids:
                    all_devices.add(rid)
            else:
                # 如果没有指定具体ID，可能使用 stage_code 作为资源池，这里简化假设必须有 device
                pass

    # 将设备列表转为索引映射
    all_devices = sorted(list(all_devices))
    device_to_id = {name: i for i, name in enumerate(all_devices)}
    num_devices = len(all_devices)
    
    if num_devices == 0:
        print("未找到具体资源ID (resources_id)，请检查数据源是否包含设备定义。")
        # 实际情况可能需要根据 stage_code 虚拟出机器资源
        return

    # 构建 Job 列表
    model_jobs = [] 
    for part_id, part_info in data_jobs.items():
        job_stages = []
        raw_stages = part_info.get('stages', {}).values()
        # 按 stage_order 排序
        sorted_stages = sorted(raw_stages, key=lambda x: x.get('stage_order', 0))
        
        for stage in sorted_stages:
            duration = int(stage.get('estimated_time', 0))
            min_start = parse_time(stage.get('min_start_time'))
            
            # 获取该工序可用的机器列表
            possible_devices = []
            res_ids = stage.get('resources_id')
            if res_ids:
                for rid in res_ids:
                    if rid in device_to_id:
                        possible_devices.append(device_to_id[rid])
            
            # 如果数据中某些工序没有列出 specific resources_id，
            # 实际项目中通常需要根据 stage_code 去 resources 字典里查找具备该能力的机器。
            # 这里为演示代码健壮性，若无匹配机器则跳过或报错，用户需根据实际 JSON 完善此逻辑。
            if not possible_devices:
                continue

            job_stages.append({
                'duration': duration,
                'min_start': min_start,
                'possible_devices': possible_devices,
                'name': stage.get('stage_code', 'Unknown')
            })
        
        if job_stages:
            model_jobs.append(job_stages)

    print(f"解析完成: 共有 {len(model_jobs)} 个工件, {num_devices} 台设备。")

    # 2. 创建 CP 模型
    model = cp_model.CpModel()

    # 3. 定义变量
    # machine_to_intervals[device_id] = list of optional intervals
    machine_to_intervals = collections.defaultdict(list)
    
    # 存储所有工件的最后一个工序的结束时间，用于计算 Makespan
    job_ends = []

    for job_id, job in enumerate(model_jobs):
        previous_end_var = None
        
        for stage_id, stage in enumerate(job):
            suffix = f'_{job_id}_{stage_id}'
            duration = stage['duration']
            min_start = stage['min_start']
            
            # 创建主区间变量 (Master Interval)
            start_var = model.NewIntVar(0, 1000000, f'start{suffix}') # 上界需根据实际情况估算
            end_var = model.NewIntVar(0, 1000000, f'end{suffix}')
            interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval{suffix}')
            
            # 约束：最早开始时间
            if min_start > 0:
                model.Add(start_var >= min_start)

            # 约束：工艺顺序 (上一个工序结束 <= 当前工序开始)
            if previous_end_var is not None:
                model.Add(previous_end_var <= start_var)
            previous_end_var = end_var

            # 柔性资源选择：为每台可选机器创建一个可选区间
            optional_intervals = []
            for dev_id in stage['possible_devices']:
                dev_suffix = f'_{job_id}_{stage_id}_{dev_id}'
                
                # 这是一个布尔变量，表示是否选择了这台机器
                is_present = model.NewBoolVar(f'pres{dev_suffix}')
                
                # 可选区间：如果 is_present 为真，则该区间生效，且时间与主区间一致
                # 注意：可选区间的 start/end 不需要新的变量，直接用主区间的 start/end 即可
                # 但为了标准 FJSP 建模，通常每个可选分支有自己的 start/end，通过 Alternative 约束同步
                # 这里使用简化版：如果选中，占用该机器的时间段就是 [start_var, end_var]
                
                opt_interval = model.NewOptionalIntervalVar(
                    start_var, duration, end_var, is_present, f'opt_interval{dev_suffix}'
                )
                
                machine_to_intervals[dev_id].append(opt_interval)
                optional_intervals.append(is_present)
            
            # 约束：必须且只能选择一台机器
            model.Add(sum(optional_intervals) == 1)

        # 记录该工件完成时间
        job_ends.append(previous_end_var)

    # 4. 资源不重叠约束
    for dev_id in range(num_devices):
        # 对每台机器的所有被选中的区间添加 NoOverlap 约束
        model.AddNoOverlap(machine_to_intervals[dev_id])

    # 5. 目标函数：最小化最大完工时间 (Makespan)
    makespan = model.NewIntVar(0, 1000000, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # 6. 求解
    solver = cp_model.CpSolver()
    # 设置求解时间限制（例如 30 秒）
    solver.parameters.max_time_in_seconds = 30.0
    print("开始求解...")
    status = solver.Solve(model)

    # 7. 输出结果
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'求解状态: {solver.StatusName(status)}')
        print(f'最大完工时间 (Makespan): {solver.ObjectiveValue()}')
        
        # 可视化或打印详细排程
        # 这里简单打印部分结果
        count = 0
        for job_id, job in enumerate(model_jobs):
            if count > 5: break # 只打印前5个工件
            print(f"工件 {job_id}:")
            last_end = 0
            for stage_id, stage in enumerate(job):
                suffix = f'_{job_id}_{stage_id}'
                start_val = solver.Value(model.GetIntVar(f'start{suffix}'))
                end_val = solver.Value(model.GetIntVar(f'end{suffix}'))
                
                # 找出是在哪台机器上做的
                selected_dev = "Unknown"
                for dev_id in stage['possible_devices']:
                    if solver.Value(model.GetBoolVar(f'pres_{job_id}_{stage_id}_{dev_id}')):
                        selected_dev = all_devices[dev_id]
                        break
                
                print(f"  - 工序 {stage['name']}: {start_val} -> {end_val} (时长 {stage['duration']}) @ 设备 {selected_dev}")
            count += 1
    else:
        print('未找到可行解。')

if __name__ == '__main__':
    # 请将此处替换为你实际下载的文件路径
    file_path = r"E:\01SHU\01_work\06_CP\schedule-goertek-main\examples\data\TEST1.json"
    solve_factory_scheduling(file_path)