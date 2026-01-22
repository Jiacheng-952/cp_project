from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator

class OptAgentState(TypedDict):
    # 输入
    problem_statement: str
    
    # 阶段 1: Architect 输出
    math_spec: Dict[str, Any]       # JSON 格式的数学蓝图
    architect_feedback: Optional[str] # 纠错意见
    
    # 阶段 2: Builder 输出
    z3_code_a: Optional[str]
    z3_code_b: Optional[str]
    
    # 阶段 3: Verifier 输出
    verification_passed: bool
    verification_log: str
    
    # 阶段 4: 最终输出
    cmpy_code: Optional[str]
    final_report: Optional[str]
    
    # 状态控制
    correction_count: int
    max_corrections: int