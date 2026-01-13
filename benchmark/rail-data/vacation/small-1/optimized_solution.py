import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# --- Data Loading ---
file_path = "/Users/luhongliang/Desktop/Research/LLM4OPT/OptAgent-langgraph/benchmark/rail-data/vacation/small-1/data.xlsx"

# 1. Station data
df_station = pd.read_excel(file_path, sheet_name="station")
stations = list(df_station["station"])
S_first = stations[0]
S_last = stations[-1]

# 2. Train data
df_train = pd.read_excel(file_path, sheet_name="train")
trains = list(df_train["trainNO"])
train_speeds = dict(zip(df_train["trainNO"], df_train["speed"]))
is_stop = {}
for _, row in df_train.iterrows():
    train_no = row["trainNO"]
    for s in stations:
        is_stop[(train_no, s)] = row[s]

# 3. Runtime data
df_runtime = pd.read_excel(file_path, sheet_name="runtime")
runtime_data = {}
for _, row in df_runtime.iterrows():
    station_interval = row["station"].split("-")
    prev_s, next_s = station_interval[0], station_interval[1]
    runtime_data[(prev_s, next_s, "300")] = row["300"]
    runtime_data[(prev_s, next_s, "350")] = row["350"]

# 4. Parameter data
df_param = pd.read_excel(file_path, sheet_name="parameter")
params = df_param.iloc[0].to_dict()
T = params["T"]
H = params["H"]
MINSTOP = params["MINSTOP"]
MAXSTOP = params["MAXSTOP"]
alpha = 1

# 5. Value data
df_value = pd.read_excel(file_path, sheet_name="value")
train_values = dict(zip(df_value["trainNO"], df_value["value"]))
train_expects = dict(zip(df_value["trainNO"], df_value["expect"]))

print(f"Problem scale: {len(trains)} trains, {len(stations)} stations")
print(f"Binary variables needed: {len(trains) * (len(trains) - 1) // 2}")

# Calculate tighter Big-M
max_runtime_300 = sum(
    runtime_data[(stations[i], stations[i + 1], "300")]
    for i in range(len(stations) - 1)
)
max_runtime_350 = sum(
    runtime_data[(stations[i], stations[i + 1], "350")]
    for i in range(len(stations) - 1)
)
max_travel = max(max_runtime_300, max_runtime_350) + MAXSTOP * (len(stations) - 2)
M = min(T, max_travel + 2 * H)  # Tighter Big-M
print(f"Using Big-M = {M} (original: {T + H})")

# --- Gurobi Model ---
model = gp.Model("TrainScheduling")

# 设置Gurobi参数 - 关键优化！
model.setParam("TimeLimit", 180)  # 3分钟求解限制
model.setParam("MIPGap", 0.02)  # 2%的优化间隙
model.setParam("Threads", 4)  # 使用4个线程
model.setParam("MIPFocus", 1)  # 重点寻找可行解
model.setParam("Presolve", 2)  # 激进预处理
model.setParam("Cuts", 2)  # 激进切平面
model.setParam("Heuristics", 0.2)  # 启发式时间占比
model.setParam("OutputFlag", 1)  # 显示日志

# Decision variables
Arr = model.addVars(trains, stations, vtype=GRB.CONTINUOUS, name="Arr", lb=0, ub=T)
Dep = model.addVars(trains, stations, vtype=GRB.CONTINUOUS, name="Dep", lb=0, ub=T)
y = model.addVars(
    [(r1, r2) for r1 in trains for r2 in trains if trains.index(r1) < trains.index(r2)],
    vtype=GRB.BINARY,
    name="y",
)

# Objective Function
objective_expr = gp.LinExpr()
for r in trains:
    d_r = Dep[r, S_first] - train_expects[r]
    g_r = gp.quicksum(Dep[r, s] - Arr[r, s] for s in stations)
    objective_expr += train_values[r] - (d_r + g_r) * alpha

model.setObjective(objective_expr, GRB.MAXIMIZE)

# Constraints
print("Adding constraints...")

# 1. Order of events at a station
for r in trains:
    for s in stations:
        model.addConstr(Arr[r, s] <= Dep[r, s], name=f"Arr_le_Dep_{r}_{s}")

# 2. Stop time constraints
for r in trains:
    for s in stations:
        if s == S_first or s == S_last:
            model.addConstr(Dep[r, s] == Arr[r, s], name=f"OriginDest_{r}_{s}")
        elif is_stop[(r, s)] == 0:
            model.addConstr(Dep[r, s] == Arr[r, s], name=f"PassThrough_{r}_{s}")
        else:
            model.addConstr(Dep[r, s] - Arr[r, s] >= MINSTOP, name=f"MinStop_{r}_{s}")
            model.addConstr(Dep[r, s] - Arr[r, s] <= MAXSTOP, name=f"MaxStop_{r}_{s}")

# 3. Running Time Constraint
for r in trains:
    for i in range(len(stations) - 1):
        s_prev = stations[i]
        s_next = stations[i + 1]
        speed_str = str(train_speeds[r])
        run_t = runtime_data[(s_prev, s_next, speed_str)]
        model.addConstr(
            Dep[r, s_prev] + run_t == Arr[r, s_next],
            name=f"RunTime_{r}_{s_prev}_{s_next}",
        )

# 4. 提供启发式初始解（基于期望出发时间排序）
sorted_trains = sorted(trains, key=lambda r: train_expects[r])
train_order = {t: i for i, t in enumerate(sorted_trains)}

for r1 in trains:
    for r2 in trains:
        i1, i2 = trains.index(r1), trains.index(r2)
        if i1 < i2:
            # 根据期望出发时间给出初始顺序
            if train_order[r1] < train_order[r2]:
                y[r1, r2].start = 1
            else:
                y[r1, r2].start = 0

print("Adding interval constraints (this may take a moment)...")
# 5. Minimum Interval Time and No Overtaking
constraint_count = 0
for i, r1 in enumerate(trains):
    for j, r2 in enumerate(trains):
        if i < j:
            for s in stations:
                # Case 1: r1 precedes r2
                model.addConstr(
                    Arr[r2, s] >= Arr[r1, s] + H - M * (1 - y[r1, r2]),
                    name=f"IntArr1_{r1}_{r2}_{s}",
                )
                model.addConstr(
                    Dep[r2, s] >= Dep[r1, s] + H - M * (1 - y[r1, r2]),
                    name=f"IntDep1_{r1}_{r2}_{s}",
                )

                # Case 2: r2 precedes r1
                model.addConstr(
                    Arr[r1, s] >= Arr[r2, s] + H - M * y[r1, r2],
                    name=f"IntArr2_{r1}_{r2}_{s}",
                )
                model.addConstr(
                    Dep[r1, s] >= Dep[r2, s] + H - M * y[r1, r2],
                    name=f"IntDep2_{r1}_{r2}_{s}",
                )
                constraint_count += 4

print(f"Added {constraint_count} interval constraints")
print(f"Total variables: {model.NumVars} ({model.NumBinVars} binary)")
print(f"Total constraints: {model.NumConstrs}")

# Solve the model
print("\n" + "=" * 60)
print("Starting optimization...")
print("=" * 60)
model.optimize()

# Extract and print results
print("\n" + "=" * 60)
if model.Status == GRB.OPTIMAL:
    print(f"✓ OPTIMAL SOLUTION FOUND")
    print(f"optimal_value = {model.ObjVal:.4f}")
    print(f"objective = {model.ObjVal:.4f}")
    print(f"result = {model.ObjVal:.4f}")
    print(f"MIP Gap: {model.MIPGap*100:.2f}%")
    print(f"Solution time: {model.Runtime:.2f}s")

elif model.Status == GRB.TIME_LIMIT:
    print(f"⚠ TIME LIMIT REACHED")
    if model.SolCount > 0:
        print(f"Best solution found:")
        print(f"optimal_value = {model.ObjVal:.4f}")
        print(f"objective = {model.ObjVal:.4f}")
        print(f"result = {model.ObjVal:.4f}")
        print(f"MIP Gap: {model.MIPGap*100:.2f}%")
        print(f"Solution time: {model.Runtime:.2f}s")
    else:
        print("No feasible solution found within time limit")
        print("optimal_value = NO_SOLUTION")

elif model.Status == GRB.INFEASIBLE:
    print(f"✗ MODEL IS INFEASIBLE")
    print("optimal_value = INFEASIBLE")
    model.computeIIS()
    print("\nIrreducible Inconsistent Subsystem (IIS):")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"  {c.ConstrName}")

elif model.Status == GRB.UNBOUNDED:
    print(f"✗ MODEL IS UNBOUNDED")
    print("optimal_value = UNBOUNDED")

else:
    print(f"✗ OPTIMIZATION FAILED")
    print(f"Status code: {model.Status}")
    print("optimal_value = ERROR")

print("=" * 60)
