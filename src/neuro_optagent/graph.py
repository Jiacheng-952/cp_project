from langgraph.graph import StateGraph, START, END
# 修改下面这两行，加上点号 .
from .state import OptAgentState
from .nodes import (
    architect_node, builder_a_node, builder_b_node, 
    verifier_node, corrector_node, converter_node, reporter_node
)

def build_optagent():
    workflow = StateGraph(OptAgentState)
    
    # 1. 添加节点
    workflow.add_node("architect", architect_node)
    workflow.add_node("builder_a", builder_a_node)
    workflow.add_node("builder_b", builder_b_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("corrector", corrector_node)
    workflow.add_node("converter", converter_node)
    workflow.add_node("reporter", reporter_node)
    
    # 2. 定义边
    workflow.add_edge(START, "architect")
    
    # 并行：Architect -> Builders
    workflow.add_edge("architect", "builder_a")
    workflow.add_edge("architect", "builder_b")
    
    # 汇聚：Builders -> Verifier
    workflow.add_edge("builder_a", "verifier")
    workflow.add_edge("builder_b", "verifier")
    
    # 条件路由：verifier_condition
    def router(state):
        if state["verification_passed"]:
            return "converter"
        elif state["correction_count"] >= state["max_corrections"]:
            return "reporter"  # 失败退出
        else:
            return "corrector" # 循环纠错

    workflow.add_conditional_edges(
        "verifier",
        router,
        {
            "converter": "converter",
            "reporter": "reporter",
            "corrector": "corrector"
        }
    )
    
    # 闭环：Corrector -> Architect
    workflow.add_edge("corrector", "architect")
    
    # 成功路径
    workflow.add_edge("converter", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()

# === 运行测试 ===
import asyncio

# async def main():
#     app = build_optagent()
    
#     # 测试问题
#     problem = "A farmer has chickens and rabbits. There are 35 heads and 94 legs. Minimize the cost if a chicken costs 2 and rabbit costs 3."
    
#     initial_state = {
#         "problem_statement": problem,
#         "correction_count": 0,
#         "max_corrections": 3
#     }
    
#     print("Starting OptAgent...")
#     final_state = await app.ainvoke(initial_state)
#     print("\n\n=== FINAL RESULT ===")
#     print(final_state["final_report"])

async def main():
    app = build_optagent()
    
    # === 复杂测试：作业车间调度 (Job Shop) 的模糊描述 ===
    # 模糊点：
    # 1. 没说一台机器同一时间只能做一个作业（Disjunctive constraint）。
    # 2. 没说作业是否可以中断（Preemptive）。
    # 预期：Builder A 和 B 可能会做出不同的假设，导致 Z3 验证不通过，触发 Corrector。
    
    # problem = """
    # We have 3 jobs (J1, J2, J3) and 2 machines (M1, M2).
    # Processing times:
    # - J1: M1(3h), M2(2h)
    # - J2: M1(2h), M2(4h)
    # - J3: M1(1h), M2(3h)
    
    # Sequence: Each job must visit M1 first, then M2.
    # Minimize the total completion time (Makespan).
    # """
    
    problem = """
    员工排班问题：

    某公司有20名员工需要分配到6个不同的班次。每个班次有不同的需求人数：
    - 班次1：需要4人
    - 班次2：需要6人
    - 班次3：需要8人
    - 班次4：需要7人
    - 班次5：需要5人
    - 班次6：需要3人
    约束条件：
        1. 每位员工只能被分配到一个班次
        2. 每个班次的需求人数必须得到满足
        3. 每位员工最多只能连续工作两个班次

    请为这20名员工安排合适的班次分配方案。
    """

    initial_state = {
        "problem_statement": problem,
        "correction_count": 0,
        "max_corrections": 3 # 允许重试3次
    }
    
    print(f"Starting OptAgent with problem:\n{problem}\n")
    final_state = await app.ainvoke(initial_state)
    print("\n\n=== FINAL RESULT ===")
    print(final_state["final_report"])
    

if __name__ == "__main__":
    asyncio.run(main())