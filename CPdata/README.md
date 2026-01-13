# CPdata - 约束规划(CP)案例数据集

## 概述

本文件夹包含了用于测试和开发约束规划(Constraint Programming, CP)模型的各种案例和数据集。所有案例都按照问题类型进行了分类整理。

## 文件夹结构

```
CPdata/
├── scheduling/          # 调度问题案例
├── assignment/          # 分配问题案例  
├── routing/             # 路径问题案例
├── tsp/                 # 旅行商问题案例
└── templates/           # CP建模模板和配置
```

## 案例详情

### 1. 调度问题 (Scheduling)

**文件**: `scheduling/flowshop_scheduling.py`
- **问题类型**: 流车间调度问题
- **求解器**: Gurobi
- **描述**: 优化作业在多个机器上的处理顺序，最小化总完成时间(makespan)
- **关键参数**: 作业列表、机器列表、处理时间矩阵

### 2. 分配问题 (Assignment)

**文件**: `assignment/aircraft_assignment.py`
- **问题类型**: 飞机分配问题
- **求解器**: Gurobi
- **描述**: 将飞机分配到航线，最小化总成本
- **关键参数**: 飞机可用性、航线需求、飞机能力、分配成本

### 3. 路径问题 (Routing)

**文件夹**: `routing/VRPTW/`
- **问题类型**: 带时间窗口的车辆路径问题(VRPTW)
- **描述**: 优化车辆路径，满足客户需求和时间窗口约束
- **包含文件**:
  - `input_targets.json`: 问题背景、约束和目标定义
  - `input.json`: 详细参数描述
  - `data.json`: 具体数据实例

### 4. 旅行商问题 (TSP)

**文件**: `tsp/`
- **问题类型**: 旅行商问题
- **描述**: 寻找访问所有城市一次并返回起点的最短路径
- **包含案例**:
  - `IndustryOR_TSP.jsonl`: 4城市TSP问题
  - `MAMO_TSP.jsonl`: 4城市TSP问题
  - `IndustryOR_5city_TSP.jsonl`: 5城市TSP问题

### 5. 建模模板 (Templates)

**文件**: `templates/`
- `cp_modeler.md`: CP建模详细指南和示例代码
- `cp_solvers.py`: CP求解器配置和问题分类

## 使用说明

### 对于CP模型测试

1. **选择问题类型**: 根据您的测试需求选择相应的文件夹
2. **查看参数定义**: 每个案例都提供了完整的参数说明
3. **实现CP模型**: 使用OR-Tools CP-SAT求解器实现模型
4. **测试求解**: 运行模型并验证结果

### 推荐的测试顺序

1. **入门测试**: 从TSP问题开始，理解基本的路径约束
2. **中级测试**: 尝试分配问题和调度问题
3. **高级测试**: 挑战VRPTW等复杂路径问题

## 技术栈建议

- **主要求解器**: OR-Tools CP-SAT
- **备选求解器**: Gurobi (对于某些案例)
- **编程语言**: Python
- **建模框架**: 遵循`cp_modeler.md`中的指导原则

## 注意事项

1. 所有案例都提供了完整的参数定义，但部分案例可能需要数据预处理
2. VRPTW案例是典型的CP问题，适合测试复杂的约束处理能力
3. 调度和分配问题展示了CP在资源分配方面的优势
4. 建议先阅读`templates/cp_modeler.md`了解CP建模最佳实践

## 扩展建议

如需更多CP测试案例，可以考虑：
- 装箱问题(Bin Packing)
- 作业车间调度(Job Shop Scheduling)
- 体育赛程安排(Sports Scheduling)
- 人员排班(Staff Rostering)

---

*最后更新: 2025-12-18*