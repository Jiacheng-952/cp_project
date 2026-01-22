from cpmpy import Model, intvar, boolvar, cpm_array, sum as cpm_sum
import numpy as np

# Define the sets
I = list(range(1, 21))  # Employees 1 to 20
J = list(range(1, 7))   # Shifts 1 to 6

# Define demand parameters
d_j = {
    1: 4,
    2: 6,
    3: 8,
    4: 7,
    5: 5,
    6: 3
}

# Define decision variables as boolean variables
x_ij = boolvar(shape=(len(I), len(J)), name="x", flat=False)

# Create model
model = Model()

# Constraint 1: Each employee is assigned to exactly one shift
for i in range(len(I)):
    model += (cpm_sum(x_ij[i, :]) == 1)

# Constraint 2: Each shift must have exactly the required number of employees
for j in range(len(J)):
    model += (cpm_sum(x_ij[:, j]) == d_j[j+1])  # j+1 because J starts at 1

# Constraint 3: No employee can work more than 2 consecutive shifts
for i in range(len(I)):
    for j in range(4):  # j ∈ {0,1,2,3} corresponding to shifts 1,2,3,4
        model += (cpm_sum(x_ij[i, j:j+3]) <= 2)

# Solve the model
if model.solve():
    print("Solution found!")
    print("=" * 50)
    
    # Print assignments in a readable format
    solution_matrix = x_ij.value()
    
    print("Employee-Shift Assignments (1=assigned, 0=not assigned):")
    print("Employee | " + " | ".join(f"Shift {j}" for j in J))
    print("-" * 60)
    
    for i_idx, i in enumerate(I):
        row_assignments = " | ".join(f"{int(solution_matrix[i_idx, j_idx]):^6}" 
                                    for j_idx, j in enumerate(J))
        print(f"Emp {i:>3}   | {row_assignments}")
    
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    
    # Verify constraints are satisfied
    print(f"\n- Each employee works exactly one shift:")
    for i_idx in range(len(I)):
        emp_shifts = int(solution_matrix[i_idx, :].sum())
        print(f"  Employee {I[i_idx]}: {emp_shifts} shift{'s' if emp_shifts != 1 else ''}", end="")
        if emp_shifts != 1:
            print(" ❌ ERROR")
        else:
            print(" ✓")
    
    print(f"\n- Shift demands satisfaction:")
    for j_idx, j in enumerate(J):
        shift_count = int(solution_matrix[:, j_idx].sum())
        print(f"  Shift {j} (req {d_j[j]}): {shift_count} employee{'s' if shift_count != 1 else ''}", 
              end="")
        if shift_count != d_j[j]:
            print(" ❌ ERROR")
        else:
            print(" ✓")
    
    print(f"\n- No more than 2 consecutive shifts:")
    consecutive_violations = 0
    for i_idx in range(len(I)):
        for j in range(4):
            consecutive_count = solution_matrix[i_idx, j:j+3].sum()
            if consecutive_count > 2:
                consecutive_violations += 1
                print(f"  Employee {I[i_idx]}, shifts {j+1}-{j+3}: {consecutive_count} consecutive ❌")
    if consecutive_violations == 0:
        print("  All employees satisfy consecutive shift restriction ✓")
    
else:
    print("No solution found!")