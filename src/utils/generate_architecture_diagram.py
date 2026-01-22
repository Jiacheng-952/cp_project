# ./src/utils/generate_architecture_diagram.py
import graphviz

def render_mas_architecture(output_filename='mas_architecture'):
    """
    使用 Graphviz 生成多智能体系统架构图。
    """
    # 初始化有向图，设置全局样式
    dot = graphviz.Digraph('MAS_Workflow', comment='Multi-Agent System Architecture')
    dot.attr(rankdir='TB', compound='true', fontname='Microsoft YaHei')
    dot.attr('node', shape='box', style='filled', fontname='Microsoft YaHei')
    dot.attr('edge', fontname='Microsoft YaHei')

    # 定义颜色方案
    colors = {
        'orchestrator': '#ffccff', # 粉色
        'agent': '#cce5ff',        # 蓝色
        'data': '#fff2cc',         # 黄色
        'ext': '#d5e8d4',          # 绿色
        'subgraph': '#fcfcfc'      # 浅灰背景
    }

    # --- 主要节点定义 ---
    dot.node('Start', '用户/触发器 Initiator', shape='ellipse', style='filled', fillcolor='#e0e0e0')
    dot.node('MainFlow', 'workflow.py\n(主流程编排器 Orchestrator)', fillcolor=colors['orchestrator'], penwidth='2')
    dot.node('End', '结束 / 展示结果 End', shape='ellipse', style='filled', fillcolor='#e0e0e0')

    # --- 阶段一：数据准备 ---
    with dot.subgraph(name='cluster_PG1') as c:
        c.attr(label='阶段一：数据准备 (Data Preparation)', style='filled', fillcolor=colors['subgraph'])
        c.node('DataAgent', '数据预处理智能体\n(Data Agent)', fillcolor=colors['agent'])
        c.node('RawData', '原始业务数据源\n(订单, BOM, 机台, 日历)', shape='cylinder', fillcolor=colors['data'], style='dashed,filled')
        c.node('StructuredData', '结构化调度问题对象\n(Structured Problem Data)', shape='note', fillcolor=colors['data'], style='dashed,filled')
        
        c.edge('RawData', 'DataAgent', label='读取', style='dashed')
        c.edge('DataAgent', 'StructuredData', label='2. 清洗、转换、验证')

    # --- 阶段二：CP建模核心 ---
    with dot.subgraph(name='cluster_PG2') as c:
        c.attr(label='阶段二：CP建模核心 (CP Modeling Core)', style='filled', fillcolor=colors['subgraph'])
        c.node('ModelAgent', 'CP建模智能体\n(Modeling Agent)', fillcolor=colors['agent'])
        c.node('CPModel', '实例化 CP 模型对象\n(Instantiated CP Model)', shape='note', fillcolor=colors['data'], style='dashed,filled')
        # 添加注释节点
        note_text = "包含：\\n- 决策变量 (start_vars)\\n- 硬约束 (时序, 能力)\\n- 目标函数 (最小化 Makespan)"
        c.node('ModelNote', note_text, shape='plain', fontsize='10')
        c.edge('CPModel', 'ModelNote', style='dotted', arrowhead='none')

        c.edge('StructuredData', 'ModelAgent', label='作为输入', style='dashed', ltail='cluster_PG1', lhead='cluster_PG2')
        c.edge('ModelAgent', 'CPModel', label='业务约束 -> 数学模型映射')

    # --- 阶段三：求解优化 ---
    with dot.subgraph(name='cluster_PG3') as c:
        c.attr(label='阶段三：求解优化 (Optimization Solving)', style='filled', fillcolor=colors['subgraph'])
        c.node('SolverAgent', '求解器接口智能体\n(Solver Interface Agent)', fillcolor=colors['agent'])
        c.node('ExternalSolver', '外部 CP 求解器引擎\ne.g., OR-Tools CP-SAT / CPLEX', shape='component', fillcolor=colors['ext'])
        c.node('RawSolution', '原始求解结果与状态\n(Raw Solution Values & Status)', shape='note', fillcolor=colors['data'], style='dashed,filled')

        c.edge('CPModel', 'SolverAgent', label='加载模型', style='dashed')
        c.edge('SolverAgent', 'ExternalSolver', label='调用底层API, 配置参数')
        c.edge('ExternalSolver', 'SolverAgent', label='返回状态和解值')
        c.edge('SolverAgent', 'RawSolution', label='5. 提取原始数学解')

    # --- 阶段四：结果后处理 ---
    with dot.subgraph(name='cluster_PG4') as c:
        c.attr(label='阶段四：结果后处理与交付 (Post-processing & Delivery)', style='filled', fillcolor=colors['subgraph'])
        c.node('ResultAgent', '结果后处理智能体\n(Result Processing Agent)', fillcolor=colors['agent'])
        c.node('FinalOutput', '最终排程方案 / Gantt视图数据\n(Final Schedule)', shape='note', fillcolor=colors['data'], style='dashed,filled')

        c.edge('RawSolution', 'ResultAgent', label='输入原始解', style='dashed')
        # 跨子图的连接需要特别处理，这里简化连接
        c.edge('StructuredData', 'ResultAgent', label='结合业务上下文', style='dashed', constraint='false') 
        c.edge('ResultAgent', 'FinalOutput', label='业务逻辑验证、格式化')

    # --- 主流程控制连接 ---
    dot.edge('Start', 'MainFlow')
    dot.edge('MainFlow', 'DataAgent', label='1. 调度任务启动', lhead='cluster_PG1')
    dot.edge('MainFlow', 'ModelAgent', label='3. 请求建模', lhead='cluster_PG2')
    dot.edge('MainFlow', 'SolverAgent', label='4. 启动求解', lhead='cluster_PG3')
    dot.edge('MainFlow', 'ResultAgent', label='6. 请求结果校验', lhead='cluster_PG4')
    dot.edge('MainFlow', 'End', label='7. 流程结束，返回结果')
    dot.edge('FinalOutput', 'End', style='dashed')

    # 渲染并保存
    try:
        # format='png' 或 'svg'
        output_path = dot.render(output_filename, view=False, format='png')
        print(f"架构图已成功生成于: {output_path}")
    except graphviz.backend.ExecutableNotFound:
        print("错误: 未找到 Graphviz 可执行文件。请确保已安装 Graphviz 并将其添加到系统路径中。")
    except Exception as e:
        print(f"生成图表时出错: {e}")

if __name__ == '__main__':
    # 运行此脚本将在当前目录下生成 png 图片
    render_mas_architecture()
