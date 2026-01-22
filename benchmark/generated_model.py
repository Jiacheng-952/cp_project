import cpmpy as cp

# Create integer variables
A = cp.intvar(0, 100, name="A")  # Assuming reasonable bounds
B = cp.intvar(0, 100, name="B")
C = cp.intvar(0, 100, name="C")
D = cp.intvar(0, 100, name="D")

# Initialize constraints list
constraints = []

# Create a model
model = cp.Model()

# Solve and print results
if model.solve():
    print("Solution found:")
    print(f"A = {A.value()}")
    print(f"B = {B.value()}")
    print(f"C = {C.value()}")
    print(f"D = {D.value()}")
else:
    print("No solution found")