import z3
from z3 import *
import multiprocessing

def check_z3_equivalence(code_a: str, code_b: str, timeout_ms: int = 100000) -> dict:
    """
    Check if two Z3 code snippets are logically equivalent.
    
    Args:
        code_a: Source code from Builder A
        code_b: Source code from Builder B
        timeout_ms: Timeout in milliseconds (default 10s)
        
    Returns:
        dict: {"equivalent": bool, "reason": str}
    """
    
    # 定义一个内部函数来执行具体的 Z3 逻辑
    def verify_logic(q, code_a, code_b):
        try:
            # Create separate contexts to avoid pollution
            ctx = Context()
            
            # Helper to execute code and extract constraints
            def get_constraints(code_str, context):
                # Sandbox environment
                local_env = {'z3': z3, 'constraints': []}
                # Inject context-specific Z3 creators if needed, 
                # but standard Z3 python API uses global context usually.
                # For safety, we just exec. 
                # Note: Z3 Python objects are tied to a context. 
                # Simple exec usually works with default context.
                try:
                    exec(code_str, {}, local_env)
                    return local_env.get('constraints', [])
                except Exception as e:
                    return str(e)

            cons_a = get_constraints(code_a, ctx)
            cons_b = get_constraints(code_b, ctx)

            if isinstance(cons_a, str):
                q.put({"equivalent": False, "reason": f"Code A Error: {cons_a}"})
                return
            if isinstance(cons_b, str):
                q.put({"equivalent": False, "reason": f"Code B Error: {cons_b}"})
                return

            # Construct the miter: (A AND NOT B) OR (NOT A AND B)
            # In constraint terms: NOT (A <-> B)
            # Which means: (A and !B) OR (!A and B)
            
            s = Solver()
            s.set("timeout", timeout_ms) # 设置超时时间

            # Make simple assumption: combining all constraints implies logical AND
            f_a = And(cons_a)
            f_b = And(cons_b)

            # Equivalence check: Is it IMPOSSIBLE for them to be different?
            # We want to prove: Forall X, A(X) == B(X)
            # We check negation: Exists X, A(X) != B(X)
            # condition: (f_a != f_b)
            # In Z3: Xor(f_a, f_b)
            
            s.add(Xor(f_a, f_b))
            
            result = s.check()
            
            if result == unsat:
                # UNSAT means NO counter-example exists -> They are Equivalent
                q.put({"equivalent": True, "reason": "SUCCESS: Logically Equivalent (UNSAT)"})
            elif result == sat:
                # SAT means Counter-example found -> Not Equivalent
                q.put({"equivalent": False, "reason": "FAILED: Found Counter-example (Models diverge)"})
            elif result == unknown:
                 q.put({"equivalent": False, "reason": "TIMEOUT: Verification took too long"})
            else:
                q.put({"equivalent": False, "reason": f"Unknown Z3 state: {result}"})

        except Exception as e:
            q.put({"equivalent": False, "reason": f"Verification Crash: {str(e)}"})

    # 由于 Z3 在主进程中可能会阻塞 asyncio，最好用进程或简单处理
    # 这里为了简单起见，我们直接在当前进程跑，依靠 Z3 内部的 timeout
    # 但为了防止 Z3 彻底卡死 Python GIL，理想情况是用 multiprocessing
    # 下面是一个简化版，直接利用 Z3 的 timeout 参数
    
    # --- 简化版实现 (直接运行) ---
    try:
        # 1. 提取约束
        local_env_a = {'z3': z3, 'constraints': []}
        exec(code_a, {}, local_env_a)
        cons_a = local_env_a.get('constraints', [])
        
        local_env_b = {'z3': z3, 'constraints': []}
        exec(code_b, {}, local_env_b)
        cons_b = local_env_b.get('constraints', [])
        
        # 2. 构建求解器
        s = Solver()
        s.set("timeout", timeout_ms)  # <--- 关键修改：设置超时 (毫秒)
        
        f_a = And(cons_a)
        f_b = And(cons_b)
        
        # 3. 验证 (A XOR B)
        s.add(Xor(f_a, f_b))
        
        check_res = s.check()
        
        if check_res == unsat:
             return {"equivalent": True, "reason": "SUCCESS: Logically Equivalent (UNSAT)"}
        elif check_res == sat:
             return {"equivalent": False, "reason": "FAILED: Divergence Found"}
        elif check_res == unknown:
             return {"equivalent": False, "reason": "WARNING: Verification Timeout (Too Complex)"}
             
    except Exception as e:
        return {"equivalent": False, "reason": f"Execution Error: {e}"}

    return {"equivalent": False, "reason": "Unknown Error"}