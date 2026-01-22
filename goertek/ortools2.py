import json
import collections
from ortools.sat.python import cp_model
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
# 如果中文报错，改用这行
plt.rcParams['font.family'] = 'sans-serif'
def solve_factory_scheduling(json_file_path):
    # 1. 加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_jobs = raw_data.get('data', {})
    
    # 如果 JSON 中没有显式的 resources 列表，我们需要从工序中提取所有出现的资源 ID
    all_devices = set()
    
    # 解析日期工具函数
    def parse_time(date_str):
        if not date_str:
            return 0
        try:
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

    # 将设备列表转为索引映射
    all_devices = sorted(list(all_devices))
    device_to_id = {name: i for i, name in enumerate(all_devices)}
    num_devices = len(all_devices)
    
    if num_devices == 0:
        print("未找到具体资源ID (resources_id)，请检查数据源是否包含设备定义。")
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
    
    job_ends = []
    
    # --- 关键修改：建立字典来存储变量，以便后续查询 ---
    starts = {}      # 存储开始时间变量
    ends = {}        # 存储结束时间变量
    presences = {}   # 存储“是否选择某机器”的布尔变量: key=(job_id, stage_id, dev_id)

    for job_id, job in enumerate(model_jobs):
        previous_end_var = None
        
        # 注意：这里用 stage_id
        for stage_id, stage in enumerate(job):
            suffix = f'_{job_id}_{stage_id}'
            duration = stage['duration']
            min_start = stage['min_start']
            
            # 创建主区间变量 (Master Interval)
            start_var = model.NewIntVar(0, 1000000, f'start{suffix}') 
            end_var = model.NewIntVar(0, 1000000, f'end{suffix}')
            interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval{suffix}')
            
            # --- 修复 1: 存入字典 (使用 stage_id) ---
            starts[(job_id, stage_id)] = start_var
            ends[(job_id, stage_id)] = end_var

            # 约束：最早开始时间
            if min_start > 0:
                model.Add(start_var >= min_start)

            # 约束：工艺顺序
            if previous_end_var is not None:
                model.Add(previous_end_var <= start_var)
            previous_end_var = end_var

            # 柔性资源选择
            optional_intervals = []
            for dev_id in stage['possible_devices']:
                dev_suffix = f'_{job_id}_{stage_id}_{dev_id}'
                
                is_present = model.NewBoolVar(f'pres{dev_suffix}')
                
                # --- 修复 2: 将布尔变量存入字典 ---
                presences[(job_id, stage_id, dev_id)] = is_present

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
        model.AddNoOverlap(machine_to_intervals[dev_id])

    # 5. 目标函数
    makespan = model.NewIntVar(0, 1000000, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # 6. 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    print("开始求解...")
    status = solver.Solve(model)

    # 7. 输出结果
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'求解状态: {solver.StatusName(status)}')
        print(f'最大完工时间 (Makespan): {solver.ObjectiveValue()}')
        
        count = 0
        for job_id, job in enumerate(model_jobs):
            if count > 5: break 
            print(f"工件 {job_id}:")
            for stage_id, stage in enumerate(job):
                
                # --- 修复 3: 从字典中获取值 (不再使用 GetIntVar) ---
                start_val = solver.Value(starts[(job_id, stage_id)])
                end_val = solver.Value(ends[(job_id, stage_id)])
                
                # 找出是在哪台机器上做的
                selected_dev = "Unknown"
                for dev_id in stage['possible_devices']:
                    # 从字典获取布尔值
                    if solver.Value(presences[(job_id, stage_id, dev_id)]):
                        selected_dev = all_devices[dev_id]
                        break
                
                print(f"  - 工序 {stage['name']}: {start_val} -> {end_val} (时长 {stage['duration']}) @ 设备 {selected_dev}")
            count += 1
    else:
        print('未找到可行解。')
    # 7. 可视化结果：甘特图
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'求解状态: {solver.StatusName(status)}')
        print(f'最大完工时间 (Makespan): {solver.ObjectiveValue()}')

        # --- 配置中文字体 (防止中文乱码) ---
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 准备绘图数据
        # 结构: machine_schedule[machine_name] = [(start, duration, job_id, stage_name), ...]
        machine_schedule = collections.defaultdict(list)
        
        # 收集数据
        for job_id, job in enumerate(model_jobs):
            for stage_id, stage in enumerate(job):
                start_val = solver.Value(starts[(job_id, stage_id)])
                duration = stage['duration']
                # 如果时长为0，通常不绘制，或者绘制一条细线
                if duration == 0: continue
                
                # 找到被选中的机器
                selected_dev_name = None
                for dev_id in stage['possible_devices']:
                    if solver.Value(presences[(job_id, stage_id, dev_id)]):
                        selected_dev_name = all_devices[dev_id]
                        break
                
                if selected_dev_name:
                    machine_schedule[selected_dev_name].append({
                        'start': start_val,
                        'duration': duration,
                        'job_id': job_id,
                        'name': stage['name']
                    })

        # --- 开始绘图 ---
        # 动态设置图片高度：机器越多，图片越长
        fig_height = max(10, num_devices * 0.4) 
        fig, ax = plt.subplots(figsize=(20, fig_height))

        # 生成颜色池 (为每个工件分配一个颜色)
        # 使用 tab20 调色板，如果工件超过20个则循环使用
        colors = plt.cm.tab20.colors 
        
        # 获取所有机器名称并在Y轴排序
        y_labels = sorted(list(machine_schedule.keys()))
        y_ticks = range(len(y_labels))
        
        # 绘制条形
        for i, machine_name in enumerate(y_labels):
            tasks = machine_schedule[machine_name]
            for task in tasks:
                # 颜色由 job_id 决定
                color = colors[task['job_id'] % len(colors)]
                
                # 绘制水平条: (start, i) 是左下角坐标
                # width=duration, height=0.8
                ax.broken_barh([(task['start'], task['duration'])], (i - 0.4, 0.8), 
                               facecolors=color, edgecolor='black', linewidth=0.5)
                
                # 如果条形足够宽，在里面写上工件ID
                if task['duration'] > 50: # 这里的50是阈值，可根据时间跨度调整
                    mid_point = task['start'] + task['duration']/2
                    ax.text(mid_point, i, f"J{task['job_id']}", 
                            ha='center', va='center', color='white', fontsize=8, fontweight='bold')

        # 设置坐标轴
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('时间 (Time)')
        ax.set_title(f'工厂排程甘特图 (Makespan: {solver.ObjectiveValue()})')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        # 保存图片而不是直接显示 (防止窗口卡死)
        output_img = 'schedule_gantt.png'
        plt.savefig(output_img, dpi=150, bbox_inches='tight')
        print(f"甘特图已保存为: {output_img}")
        
        # 如果你想弹窗显示，请取消下面这行的注释
        # plt.show()

    else:
        print('未找到可行解。')
if __name__ == '__main__':
    # 请确保路径正确
    file_path = r"E:\01SHU\01_work\06_CP\schedule-goertek-main\examples\data\TEST1.json"
    solve_factory_scheduling(file_path)