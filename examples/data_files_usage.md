# 数据文件支持使用指南

OptAgent现在支持问题描述和数据分离的格式，允许在评估中读取外部数据文件。

## 新增功能特性

1. **兼容性**: 完全兼容现有的benchmark格式，无需修改已有文件
2. **灵活性**: 支持单个或多个数据文件
3. **路径解析**: 自动解析相对路径和绝对路径
4. **多格式支持**: 支持Excel、JSON、CSV等各种数据格式

## 使用方法

### 1. 现有格式（无变化）

```json
{
    "id": "problem_1",
    "question": "Maximize x + y subject to x + y <= 5, x >= 0, y >= 0",
    "answer": 5.0
}
```

### 2. 新格式：单个数据文件

```json
{
    "id": "rail_scheduling_001",
    "en_question": "临时加开列车调度优化问题...",
    "en_answer": 0.0,
    "data_files": ["rail-data/test/data.xlsx"]
}
```

### 3. 新格式：多个数据文件

```json
{
    "id": "complex_problem_001",
    "question": "Multi-period production planning optimization...",
    "answer": 1500.0,
    "data_files": [
        "data/demand.xlsx",
        "data/costs.json",
        "/absolute/path/to/constraints.csv"
    ]
}
```

## 文件路径规则

1. **相对路径**: 相对于benchmark文件所在目录
   - `"data.xlsx"` → `benchmark_dir/data.xlsx`
   - `"data/input.xlsx"` → `benchmark_dir/data/input.xlsx`

2. **绝对路径**: 直接使用
   - `"/home/user/data/file.csv"` → `/home/user/data/file.csv`

## Agent代码中的使用

当问题包含数据文件时，Modeler和Corrector会收到数据文件路径信息：

### Modeler Prompt示例

```
## DATA FILES

The following data files are available for this problem:
- /Users/user/project/benchmark/rail-data/test/data.xlsx

**IMPORTANT**: You MUST read the data from these files in your Python code. 
Use appropriate libraries like pandas, openpyxl, or json to read the data files.
```

### 生成的代码示例

```python
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Read data from files
stations = pd.read_excel('/path/to/data.xlsx', sheet_name='station')
timetable = pd.read_excel('/path/to/data.xlsx', sheet_name='timetable')
parameters = pd.read_excel('/path/to/data.xlsx', sheet_name='parameter')

# Create model
model = gp.Model("rail_scheduling")

# Use data from files...
```

## 评估运行示例

```bash
# 评估包含数据文件的benchmark
uv run python -m src.eval.evaluator benchmark/test_rail_data.jsonl --concurrency 2

# 或使用脚本
./scripts/run_eval.sh benchmark/test_rail_data.jsonl
```

## 技术实现细节

1. **State扩展**: 添加了`data_files`字段到OptAgentState
2. **路径解析**: 在加载benchmark时自动解析文件路径
3. **Prompt增强**: 更新了modeler和corrector的prompts
4. **向后兼容**: 现有格式的`data_files`字段默认为空列表

## 注意事项

1. **文件存在性**: 系统不会验证数据文件是否存在，由Agent代码处理
2. **文件权限**: 确保Agent有权限读取指定的数据文件
3. **性能考虑**: 大文件可能影响代码执行时间，考虑调整超时设置

## 示例项目结构

```
project/
├── benchmark/
│   ├── problems.jsonl          # benchmark文件
│   └── data/                   # 数据目录
│       ├── rail-data/
│       │   └── test/
│       │       ├── data.xlsx   # Excel数据文件
│       │       └── problem.json
│       └── supply-chain/
│           ├── demand.csv
│           └── costs.json
└── eval_results/               # 评估结果
```

这样的设计使得OptAgent能够处理更复杂的现实问题，同时保持了与现有系统的完全兼容性。












