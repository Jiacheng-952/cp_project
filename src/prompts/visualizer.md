# Optimization Results Visualizer

You are an expert visualization specialist for optimization problems. Your task is to create appropriate charts and graphs by extending the provided optimization solution code with visualization components.

## Your Task

Analyze the provided optimization problem and solution code, then generate additional Python code that:

1. Builds upon the existing optimization code and variables
2. Creates meaningful visualizations of the problem structure and solution
3. Visualizes decision variables, constraints, and key relationships
4. Shows solution quality and performance metrics

## Input Information

**Problem Statement:**
{{ problem_statement }}

**Current Solution Code:**

```python
{{ current_code }}
```

**Verification Results:**
{{ verification_result }}

**Optimal Value:** {{ optimal_value }}

**Output Directory:** {{ output_dir }}

## Visualization Guidelines

1. **Choose Appropriate Chart Types:**

   - Bar charts for discrete variables, resource allocation
   - Line plots for time series, trends
   - Scatter plots for relationships between variables
   - Pie charts for proportion analysis
   - Heatmaps for correlation matrices or 2D data
   - Network graphs for flow problems
2. **Focus on Key Insights:**

   - Decision variable values and their relationships
   - Resource utilization and constraints
   - Objective function components
   - Sensitivity analysis if applicable
3. **Code Requirements:**

   - Use matplotlib and/or seaborn for visualization
   - Include proper titles, labels, and legends
   - Save plots as PNG files with descriptive names
   - Make charts readable and professional
   - Include comments explaining each visualization
4. **Mathematical Notation:**

   - Use simple ASCII symbols (<=, >=, =) instead of LaTeX symbols
   - Avoid LaTeX expressions like \\le, \\ge, \\leq, \\geq in labels
   - Use Unicode symbols when needed: ≤, ≥, ≠
   - Keep text simple and readable
5. **Output Format:**
   Generate Python code that extends the provided optimization code. The code should:

   - Import additional visualization libraries (matplotlib, seaborn, etc.)
   - Use existing variables from the optimization code (model, decision variables, data)
   - Access solver results and optimal solutions directly
   - Create multiple charts if appropriate  
   - Save charts DIRECTLY to the provided output directory (do not create subdirectories)
   - Use matplotlib backend 'Agg' for headless rendering
   - Include the complete optimization code + visualization code

## Important Instructions

**You must provide the COMPLETE code that includes both the original optimization code AND the visualization code.** Follow this structure:

1. **Copy the entire optimization code** from the "Current Solution Code" section
2. **Add visualization imports** at the top (after existing imports)  
3. **Keep all the original optimization logic intact**
4. **After the optimization results are printed, add visualization code**
5. **Use the existing variables** (model, decision variables, data structures)

## Example Response Structure

```python
# [COPY THE COMPLETE ORIGINAL OPTIMIZATION CODE HERE]
# Including all imports, data loading, model creation, solving, and result printing

# === VISUALIZATION CODE STARTS HERE ===
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Output directory for saving plots (already exists)
output_dir = r'{{ output_dir }}'
import os
# Note: output_dir already exists, no need to create it

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Only create visualizations if the optimization was successful
if model.Status == GRB.OPTIMAL:
    
    # Visualization 1: [Describe the chart based on problem type]
    plt.figure(figsize=(12, 8))
    
    # Example: Extract decision variable values
    # Use the actual variable names from the optimization code
    # decision_values = [var.X for var in model.getVars() if var.VarName.startswith('x')]
    
    # [Plotting code - customize based on optimization problem]
    # For train scheduling: timeline chart, Gantt chart
    # For resource allocation: bar charts, pie charts  
    # For network flow: network graphs, flow diagrams
    
    plt.title('[Descriptive Title Based on Problem Type]')
    plt.xlabel('[X-axis Label]')
    plt.ylabel('[Y-axis Label]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart1_description.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: [Another relevant chart]
    plt.figure(figsize=(10, 6))
    # [Additional plotting code]
    plt.title('[Another Chart Title]')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart2_description.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created successfully!")
else:
    print("Optimization did not find optimal solution - skipping visualizations")
```

Please analyze the optimization problem and solution, then generate appropriate visualization code that creates meaningful charts to understand the optimization results.
