import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np

# Create model
model = gp.Model("optimization_problem")

# --- Data Reading ---
# Use CSV files from data/ directory
data_dir = "data/"

# 1. Read 'station' CSV
df_station = pd.read_csv(f"{data_dir}station.csv")
stations = df_station["station"].tolist()

# 2. Read 'train' CSV
df_train = pd.read_csv(f"{data_dir}train.csv")
trains = df_train["trainNO"].tolist()
train_speeds_map = (
    df_train.set_index("trainNO")["speed"].astype(str).to_dict()
)  # Speed class as string for lookup

# Prepare train_stop_flags: {(trainNO, station): 1/0}
train_stop_flags = {}
for _, row in df_train.iterrows():
    train_no = row["trainNO"]
    for station_name in stations:
        if station_name in row.index:
            train_stop_flags[(train_no, station_name)] = row[station_name]
        else:
            train_stop_flags[(train_no, station_name)] = (
                0  # Default to not stopping if column not found
            )


# 3. Read 'runtime' CSV
df_runtime = pd.read_csv(f"{data_dir}runtime.csv")
runtimes_map = {}  # {(from_station, to_station, speed_class_str): runtime_value}
for _, row in df_runtime.iterrows():
    station_pair_str = row["station"]
    from_station, to_station = station_pair_str.split("-")
    for speed_col in df_runtime.columns:
        if speed_col != "station":
            runtimes_map[(from_station, to_station, speed_col)] = row[speed_col]

# 4. Read 'parameter' CSV (transposed format)
df_parameter_raw = pd.read_csv(f"{data_dir}parameter.csv", header=None)
parameter_names = df_parameter_raw.iloc[0].tolist()
parameter_values = df_parameter_raw.iloc[1].tolist()
parameters = dict(zip(parameter_names, parameter_values))

MAX_TIME = int(parameters["T"])
MIN_HEADWAY = int(parameters["H"])
MIN_STOP_TIME = int(parameters["MINSTOP"])
MAX_STOP_TIME = int(parameters["MAXSTOP"])

# Derived sets and parameters
first_station = stations[0]
last_station = stations[-1]
intermediate_stations = [s for s in stations if s not in [first_station, last_station]]
BIG_M = MAX_TIME  # A sufficiently large number for Big-M constraints

# Configure Gurobi logging
import os

os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
model.setParam("LogFile", "logs/solver.log")  # Save solver log to file
model.setParam("OutputFlag", 1)  # Enable console output

# Decision variables
# A_is: Arrival time of train i at station s
A = model.addVars(trains, stations, vtype=GRB.CONTINUOUS, lb=0, ub=MAX_TIME, name="A")
# D_is: Departure time of train i from station s
D = model.addVars(trains, stations, vtype=GRB.CONTINUOUS, lb=0, ub=MAX_TIME, name="D")

# x_ij: Binary variable for train order (1 if i precedes j, 0 otherwise)
train_pairs = [
    (i, j)
    for idx_i, i in enumerate(trains)
    for idx_j, j in enumerate(trains)
    if idx_i < idx_j
]
x = model.addVars(train_pairs, vtype=GRB.BINARY, name="x")

# Objective function
# Minimize total travel time for all trains
obj_expr = gp.quicksum(A[i, last_station] - D[i, first_station] for i in trains)
model.setObjective(obj_expr, GRB.MINIMIZE)

# Constraints
# 2. Running Time Constraints
for i in trains:
    for s_idx in range(len(stations) - 1):
        s_current = stations[s_idx]
        s_next = stations[s_idx + 1]
        speed_class = train_speeds_map[i]

        run_duration = runtimes_map.get((s_current, s_next, speed_class))

        if run_duration is not None:
            model.addConstr(
                D[i, s_current] + run_duration == A[i, s_next],
                name=f"RunTime_T{i}_S{s_current}_S{s_next}",
            )
        else:
            print(
                f"WARNING: No runtime for train {i} (speed {speed_class}) from {s_current} to {s_next}."
            )

# 3. Dwell Time Constraints
for i in trains:
    for s in stations:
        # a. Departure must be after or at arrival time
        model.addConstr(A[i, s] <= D[i, s], name=f"DepAfterArr_T{i}_S{s}")

        # b. and c. Handle intermediate stations based on stop flag
        if s in intermediate_stations:
            if train_stop_flags[(i, s)] == 1:  # Train stops at intermediate station
                model.addConstr(
                    D[i, s] - A[i, s] >= MIN_STOP_TIME, name=f"MinStop_T{i}_S{s}"
                )
                model.addConstr(
                    D[i, s] - A[i, s] <= MAX_STOP_TIME, name=f"MaxStop_T{i}_S{s}"
                )
            else:  # Train passes through intermediate station (stop_flag 0)
                model.addConstr(D[i, s] - A[i, s] == 0, name=f"PassThrough_T{i}_S{s}")
        # d. Special handling for origin and destination stations (dwell time always 0)
        elif s == first_station or s == last_station:
            model.addConstr(D[i, s] - A[i, s] == 0, name=f"DwellZero_T{i}_S{s}")

# 4. Headway and No-Overtaking Constraints
for i, j in train_pairs:
    for s in stations:
        # If x[i,j] = 1 (train i precedes train j):
        model.addConstr(
            A[j, s] - A[i, s] >= MIN_HEADWAY - BIG_M * (1 - x[i, j]),
            name=f"HeadwayArr_ij_T{i}_T{j}_S{s}",
        )
        model.addConstr(
            D[j, s] - D[i, s] >= MIN_HEADWAY - BIG_M * (1 - x[i, j]),
            name=f"HeadwayDep_ij_T{i}_T{j}_S{s}",
        )

        # If x[i,j] = 0 (train j precedes train i):
        model.addConstr(
            A[i, s] - A[j, s] >= MIN_HEADWAY - BIG_M * x[i, j],
            name=f"HeadwayArr_ji_T{i}_T{j}_S{s}",
        )
        model.addConstr(
            D[i, s] - D[j, s] >= MIN_HEADWAY - BIG_M * x[i, j],
            name=f"HeadwayDep_ji_T{i}_T{j}_S{s}",
        )

# Solve
model.optimize()


def plot_train_schedule_gantt(
    A_vars, D_vars, trains, stations, train_stop_flags, model_obj_val
):
    """
    Plot train schedule Gantt chart

    Args:
        A_vars: Arrival time variables dictionary {(train, station): value}
        D_vars: Departure time variables dictionary {(train, station): value}
        trains: List of trains
        stations: List of stations
        train_stop_flags: Stop flags dictionary {(train, station): 0/1}
        model_obj_val: Model optimal value
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, len(trains)))
    train_colors = {train: colors[i] for i, train in enumerate(trains)}

    # Plot train operation lines for each train
    for i, train in enumerate(trains):
        train_times = []
        train_stations = []

        # Collect time information for this train at each station
        for station in stations:
            arrival_time = A_vars.get((train, station), 0)
            departure_time = D_vars.get((train, station), 0)

            train_times.append((arrival_time, departure_time))
            train_stations.append(station)

        # Plot operation line segments
        for j, station in enumerate(stations):
            arrival_time, departure_time = train_times[j]

            # Dwell time at station (if any)
            if (
                train_stop_flags.get((train, station), 0) == 1
                and arrival_time != departure_time
            ):
                # Stop: plot horizontal line segment to represent dwell time
                ax.barh(
                    j,
                    departure_time - arrival_time,
                    left=arrival_time,
                    height=0.3,
                    color=train_colors[train],
                    alpha=0.8,
                    label=f"{train}" if j == 0 else "",
                )

                # Add stop markers
                ax.plot(arrival_time, j, "o", color=train_colors[train], markersize=6)
                ax.plot(departure_time, j, "s", color=train_colors[train], markersize=6)
            else:
                # Pass through station: plot point
                ax.plot(
                    arrival_time,
                    j,
                    "o",
                    color=train_colors[train],
                    markersize=4,
                    alpha=0.7,
                )

            # Plot section operation line (connecting adjacent stations)
            if j < len(stations) - 1:
                next_arrival_time = train_times[j + 1][0]
                ax.plot(
                    [departure_time, next_arrival_time],
                    [j, j + 1],
                    "-",
                    color=train_colors[train],
                    linewidth=2,
                    alpha=0.8,
                )

    # Set figure properties
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations)
    ax.set_ylabel("Stations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Train Schedule Gantt Chart\nTotal Travel Time: {model_obj_val:.1f} minutes",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Set time axis range
    all_times = []
    for train in trains:
        for station in stations:
            all_times.extend(
                [A_vars.get((train, station), 0), D_vars.get((train, station), 0)]
            )

    if all_times:
        min_time = max(0, min(all_times) - 5)
        max_time = max(all_times) + 10
        ax.set_xlim(min_time, max_time)

    # Add legend
    legend_elements = []
    for train in trains:
        legend_elements.append(
            mpatches.Patch(color=train_colors[train], label=f"Train {train}")
        )

    # Add symbol explanations
    legend_elements.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle="None",
                markersize=6,
                label="Arrival Time",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="gray",
                linestyle="None",
                markersize=6,
                label="Departure Time",
            ),
            mpatches.Patch(color="gray", alpha=0.8, label="Dwell Time"),
        ]
    )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    filename = f"logs/train_schedule_gantt_chart.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Gantt chart saved as: {filename}")

    # Show figure
    plt.show()

    return filename


def print_detailed_schedule(A_vars, D_vars, trains, stations, train_stop_flags):
    """
    Print detailed train schedule
    """
    print("\n" + "=" * 80)
    print("Detailed Train Schedule")
    print("=" * 80)

    for train in trains:
        print(f"\nTrain {train} Schedule:")
        print("-" * 50)
        print(
            f"{'Station':<8} {'Arrival':<10} {'Departure':<10} {'Dwell':<10} {'Status':<8}"
        )
        print("-" * 50)

        total_travel_time = 0
        for station in stations:
            arrival = A_vars.get((train, station), 0)
            departure = D_vars.get((train, station), 0)
            dwell_time = departure - arrival
            stop_flag = train_stop_flags.get((train, station), 0)
            stop_status = "Stop" if stop_flag == 1 else "Pass"

            print(
                f"{station:<8} {arrival:<10.1f} {departure:<10.1f} {dwell_time:<10.1f} {stop_status:<8}"
            )

        # Calculate total travel time for this train
        first_station = stations[0]
        last_station = stations[-1]
        travel_time = A_vars.get((train, last_station), 0) - D_vars.get(
            (train, first_station), 0
        )
        print(f"\nTrain {train} Total Travel Time: {travel_time:.1f} minutes")
        total_travel_time += travel_time

    print(f"\nAll Trains Total Travel Time: {total_travel_time:.1f} minutes")


# Extract results
if model.Status == GRB.OPTIMAL:
    print(f"optimal_value = {model.ObjVal}")
    print(f"objective = {model.ObjVal}")
    print(f"result = {model.ObjVal}")

    # Extract solution values for visualization
    A_solution = {}
    D_solution = {}

    for var in model.getVars():
        if var.VarName.startswith("A["):
            # Parse variable name A[train,station]
            var_name = var.VarName[2:-1]  # Remove 'A[' and ']'
            train, station = var_name.split(",")
            A_solution[(train, station)] = var.X
        elif var.VarName.startswith("D["):
            # Parse variable name D[train,station]
            var_name = var.VarName[2:-1]  # Remove 'D[' and ']'
            train, station = var_name.split(",")
            D_solution[(train, station)] = var.X

    # Print detailed schedule
    print_detailed_schedule(A_solution, D_solution, trains, stations, train_stop_flags)

    # Generate and save Gantt chart
    print("\n" + "=" * 80)
    print("Generating train schedule Gantt chart...")
    try:
        chart_filename = plot_train_schedule_gantt(
            A_solution, D_solution, trains, stations, train_stop_flags, model.ObjVal
        )
        print(f"✅ Gantt chart generated successfully!")
    except Exception as e:
        print(f"❌ Failed to generate Gantt chart: {str(e)}")

    # Print all decision variables with their optimal values (optional, can be commented out)
    print("\n" + "=" * 80)
    print("Optimal values of all decision variables:")
    print("=" * 80)
    for var in model.getVars():
        if var.X > 1e-6 or (
            var.VarName.startswith("x") and var.X < 0.5
        ):  # Print all binary vars, and non-zero continuous vars
            print(f"{var.VarName} = {var.X}")

else:
    print(f"Optimization status: {model.Status}")
    if model.Status == GRB.INFEASIBLE:
        print("optimal_value = INFEASIBLE")
        print("❌ Problem is infeasible, cannot generate Gantt chart")
    elif model.Status == GRB.UNBOUNDED:
        print("optimal_value = UNBOUNDED")
        print("❌ Problem is unbounded, cannot generate Gantt chart")
    else:
        print("optimal_value = ERROR")
        print("❌ Solver error, cannot generate Gantt chart")
