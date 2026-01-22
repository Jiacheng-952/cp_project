import json
import re
import os
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import OptAgentState
from .formal_logic import check_z3_equivalence
# === 0. 配置加载工具 ===
def load_llm_from_config():
    """
    自动寻找并读取 conf.yaml 配置 LLM
    """
    current_file = Path(__file__).resolve()
    config_path = None
    for i in range(5):
        potential_path = current_file.parents[i] / "conf.yaml"
        if potential_path.exists():
            config_path = potential_path
            break
            
    if not config_path:
        if Path("conf.yaml").exists():
            config_path = Path("conf.yaml")
    
    if not config_path:
        raise FileNotFoundError("Could not find conf.yaml in project root or parent directories.")

    print(f"--- [System] Loading config from: {config_path} ---")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    basic_model = config.get("BASIC_MODEL", {})
    
    return ChatOpenAI(
        base_url=basic_model.get("base_url"),
        api_key=basic_model.get("api_key"),
        model=basic_model.get("model", "gpt-4"),
        temperature=0, 
        max_retries=basic_model.get("max_retries", 3)
    )

# === 初始化 LLM ===
try:
    llm = load_llm_from_config()
except Exception as e:
    print(f"Error loading config: {e}")
    llm = ChatOpenAI(temperature=0)

# === 辅助函数 ===
def extract_code(text: str, tag: str = "python") -> str:
    pattern = f"```{tag}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern_generic = r"```\n(.*?)```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    return text.strip()

def extract_json(text: str) -> dict:
    pattern = r"```json\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    json_str = match.group(1).strip() if match else text
    
    json_str = json_str.strip()
    if not json_str.startswith("{") and "{" in json_str:
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        json_str = json_str[start:end]

    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Error: {e}\nContent: {json_str}")
        return {}

# ========================================================
# 核心修复：使用 {variable} 占位符，而不是 f-string 直接拼接
# ========================================================

# === 1. Architect Node (立法) ===
async def architect_node(state: OptAgentState):
    print("\n--- [Architect] Drafting Math Blueprint ---")
    problem = state["problem_statement"]
    feedback = state.get("architect_feedback", "") or "None"
    
    system_msg = """You are the Chief Optimization Architect. 
    Goal: Create a formal Mathematical Specification.
    Format: Output strictly a JSON block.
    CRITICAL: You MUST define explicit variable names (e.g., 'x_i', 'flow_ij') and types (Int/Real).
    The Builders will assume your variable names are the 'Source of Truth'."""
    
    # 修复：使用 {problem} 和 {feedback} 占位符
    user_template = "Problem: {problem}\n\nFeedback from previous failure: {feedback}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("user", user_template)])
    chain = prompt | llm | StrOutputParser()
    
    # 修复：将内容作为字典传入 invoke
    response = await chain.ainvoke({"problem": problem, "feedback": feedback})
    
    return {"math_spec": extract_json(response)}

# === 2. Builder Node (执行) ===
async def builder_node(state: OptAgentState, builder_id: str):
    print(f"\n--- [Builder {builder_id}] Generating Z3 Code ---")
    spec = state["math_spec"]
    
    # 将 JSON 转为字符串，准备注入
    spec_str = json.dumps(spec, indent=2)
    
    system_msg = f"""You are Builder {builder_id}. 
    Task: Convert the JSON Math Spec into executable Python Z3 code.
    RULES:
    1. STRICTLY use the variable names defined in the Spec.
    2. Do NOT solve the model. Just define variables and constraints.
    3. End your code by creating a list named 'constraints' containing all Z3 constraint objects.
    4. Start with `import z3`.
    """
    
    # 修复：使用 {spec_str} 占位符，避免 LangChain 解析 JSON 中的花括号
    user_template = "Math Spec JSON:\n{spec_str}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("user", user_template)])
    chain = prompt | llm | StrOutputParser()
    
    # 修复：传入 spec_str
    response = await chain.ainvoke({"spec_str": spec_str})
    
    return extract_code(response, "python")

# 包装器
async def builder_a_node(state: OptAgentState):
    code = await builder_node(state, "A")
    return {"z3_code_a": code}

async def builder_b_node(state: OptAgentState):
    code = await builder_node(state, "B")
    return {"z3_code_b": code}

# === 3. Verifier Node (司法) ===
async def verifier_node(state: OptAgentState):
    print("\n--- [Verifier] Checking Equivalence ---")
    code_a = state["z3_code_a"]
    code_b = state["z3_code_b"]
    
    result = check_z3_equivalence(code_a, code_b)
    
    print(f"Verifier Result: {result['reason']}")
    return {
        "verification_passed": result["equivalent"],
        "verification_log": result["reason"]
    }

# === 4. Corrector Node (纠错) ===
async def corrector_node(state: OptAgentState):
    print("\n--- [Corrector] Analyzing Failure ---")
    spec = state["math_spec"]
    log = state["verification_log"]
    
    spec_str = json.dumps(spec, indent=2)
    
    # 修复：使用占位符
    user_template = """The Builders produced inconsistent Z3 codes based on your Spec.
    Verifier Log: {log}
    
    Original Spec:
    {spec_str}
    
    Analyze the ambiguity in the original Spec that allowed this divergence.
    Provide concise instructions to the Architect to fix the Spec.
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", "You are a QA Analyst."), ("user", user_template)])
    chain = prompt | llm | StrOutputParser()
    
    response = await chain.ainvoke({"log": log, "spec_str": spec_str})
    
    return {
        "architect_feedback": response,
        "correction_count": state["correction_count"] + 1
    }

# === 5. Converter Node (转换) ===
async def converter_node(state: OptAgentState):
    print("\n--- [Converter] Translating to CPMpy ---")
    spec = state["math_spec"]
    z3_code = state["z3_code_a"]
    
    spec_str = json.dumps(spec, indent=2)
    
    # 【修改点】强制要求生成 CPMpy 代码
    user_template = """Task: Convert the verified Z3 Logic and Math Spec into a standard CPMpy model.

    Input Spec:
    {spec_str}

    Reference Z3 Logic:
    {z3_code}

    REQUIREMENTS:
    1. STRICTLY use the `cpmpy` library.
    2. Use `cpmpy.Model()`, `cpmpy.intvar()`, `cpmpy.boolvar()` etc.
    3. Ensure the code is complete and runnable.
    4. Include a `solve()` block that prints the solution clearly.
    5. Do NOT use Gurobi or COPT specific syntax; use high-level CPMpy syntax.
    """
    
    prompt = ChatPromptTemplate.from_messages([("user", user_template)])
    chain = prompt | llm | StrOutputParser()
    
    response = await chain.ainvoke({"spec_str": spec_str, "z3_code": z3_code})
    
    return {"cmpy_code": extract_code(response, "python")}

# === 6. Reporter Node (报告 & 保存) ===
async def reporter_node(state: OptAgentState):
    print("\n--- [Reporter] Finalizing & Saving ---")
    
    # 1. 生成报告内容
    if state["verification_passed"]:
        cmpy_code = state['cmpy_code']
        report = f"SUCCESS! Model Verified.\n\nCode saved to 'generated_model.py'"
        
        # 2. 【新增】保存代码到本地文件
        # 自动去除可能的 markdown 标记
        clean_code = cmpy_code.replace("```python", "").replace("```", "").strip()
        
        output_file = "generated_model.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
        print(f"✅ Code written to {os.path.abspath(output_file)}")
        
    else:
        report = "FAILED. Max retries reached."
        
    return {"final_report": report}