# Mathematical Model: Single-Direction Railway Scheduling Problem

## Problem Overview

The single-direction railway scheduling problem involves optimizing arrival and departure times for trains at stations while maintaining minimum headway intervals and satisfying stopping requirements.

## Mathematical Formulation

### 1. Sets and Indices

- $\mathcal{T}$: Set of trains, indexed by $i, j \in \mathcal{T}$
- $\mathcal{S}$: Set of stations, indexed by $s \in \mathcal{S}$
- $\mathcal{S}_{int}$: Set of intermediate stations (excluding origin and destination)
- $\mathcal{R}$: Set of railway sections connecting adjacent stations

where:

- $s_1$: Origin station (first station)
- $s_{|\mathcal{S}|}$: Destination station (last station)

### 2. Decision Variables

- $A_{is} \geq 0$: Continuous variable, arrival time of train $i$ at station $s$
- $D_{is} \geq 0$: Continuous variable, departure time of train $i$ from station $s$
- $x_{ij} \in \{0, 1\}$: Binary variable, 1 if train $i$ precedes train $j$, 0 otherwise

### 3. Parameters

- $t_{s,s+1}^v$: Running time from station $s$ to $s+1$ for trains of speed class $v$
- $v_i$: Speed class of train $i$
- $s_{is} \in \{0, 1\}$: Stopping flag, 1 if train $i$ stops at station $s$, 0 if it passes through
- $T$: Time limit (all times must be within $[0, T]$)
- $H$: Minimum headway interval between trains at any station
- $d_{min}$: Minimum dwell time at intermediate stations
- $d_{max}$: Maximum dwell time at intermediate stations
- $M$: Sufficiently large constant for Big-M constraints

### 4. Objective Function

$$
\text{minimize} \quad Z = \sum_{i \in \mathcal{T}} (A_{i,s_{|\mathcal{S}|}} - D_{i,s_1})
$$

**Objective Explanation:**
Minimize the total travel time of all trains (arrival at destination minus departure from origin).

### 5. Constraints

#### 5.1 Running Time Constraints

$$
D_{i,s} + t_{s,s+1}^{v_i} = A_{i,s+1}, \quad \forall i \in \mathcal{T}, \forall s \in \{1, ..., |\mathcal{S}|-1\}
$$

**Interpretation:**

- Departure time from current station plus running time equals arrival time at next station
- Running time depends on train's speed class

#### 5.2 Dwell Time Constraints

**Departure after Arrival:**

$$
A_{is} \leq D_{is}, \quad \forall i \in \mathcal{T}, \forall s \in \mathcal{S}
$$

**Intermediate Stations:**
For $s \in \mathcal{S}_{int}$:

- **If train stops** ($s_{is} = 1$):
  $$
  d_{min} \leq D_{is} - A_{is} \leq d_{max}
  $$
- **If train passes through** ($s_{is} = 0$):
  $$
  D_{is} - A_{is} = 0
  $$

**Origin and Destination:**

$$
D_{i,s_1} - A_{i,s_1} = 0, \quad \forall i \in \mathcal{T}
$$

$$
D_{i,s_{|\mathcal{S}|}} - A_{i,s_{|\mathcal{S}|}} = 0, \quad \forall i \in \mathcal{T}
$$

#### 5.3 Headway and No-Overtaking Constraints

For all train pairs $(i,j)$ where $i \neq j$ and for all stations $s \in \mathcal{S}$:

**If train $i$ precedes train $j$ ($x_{ij} = 1$):**

$$
A_{js} - A_{is} \geq H - M(1 - x_{ij})
$$

$$
D_{js} - D_{is} \geq H - M(1 - x_{ij})
$$

**If train $j$ precedes train $i$ ($x_{ij} = 0$):**

$$
A_{is} - A_{js} \geq H - M x_{ij}
$$

$$
D_{is} - D_{js} \geq H - M x_{ij}
$$

**Interpretation:**

- Ensures minimum headway interval between any two trains at any station
- Prevents overtaking by enforcing temporal separation
- Big-M formulation links binary ordering variables with time constraints

#### 5.4 Time Window Constraints

$$
0 \leq A_{is} \leq T, \quad \forall i \in \mathcal{T}, \forall s \in \mathcal{S}
$$

$$
0 \leq D_{is} \leq T, \quad \forall i \in \mathcal{T}, \forall s \in \mathcal{S}
$$

#### 5.5 Variable Type Constraints

$$
A_{is} \geq 0, \quad D_{is} \geq 0, \quad \forall i \in \mathcal{T}, \forall s \in \mathcal{S}
$$

$$
x_{ij} \in \{0, 1\}, \quad \forall i, j \in \mathcal{T}, i \neq j
$$

### 6. Complete Mathematical Model

$$
\begin{align}
\text{minimize} \quad & Z = \sum_{i \in \mathcal{T}} (A_{i,s_{|\mathcal{S}|}} - D_{i,s_1}) \\
\text{subject to} \quad & D_{i,s} + t_{s,s+1}^{v_i} = A_{i,s+1}, & \forall i \in \mathcal{T}, s \in \{1, ..., |\mathcal{S}|-1\} \\
& A_{is} \leq D_{is}, & \forall i \in \mathcal{T}, \forall s \in \mathcal{S} \\
& d_{min} \leq D_{is} - A_{is} \leq d_{max}, & \forall i \in \mathcal{T}, \forall s \in \mathcal{S}_{int}: s_{is} = 1 \\
& D_{is} - A_{is} = 0, & \forall i \in \mathcal{T}, \forall s \in \mathcal{S}_{int}: s_{is} = 0 \\
& D_{i,s_1} - A_{i,s_1} = 0, & \forall i \in \mathcal{T} \\
& D_{i,s_{|\mathcal{S}|}} - A_{i,s_{|\mathcal{S}|}} = 0, & \forall i \in \mathcal{T} \\
& A_{js} - A_{is} \geq H - M(1 - x_{ij}), & \forall i,j \in \mathcal{T}, i \neq j, \forall s \in \mathcal{S} \\
& D_{js} - D_{is} \geq H - M(1 - x_{ij}), & \forall i,j \in \mathcal{T}, i \neq j, \forall s \in \mathcal{S} \\
& A_{is} - A_{js} \geq H - M x_{ij}, & \forall i,j \in \mathcal{T}, i \neq j, \forall s \in \mathcal{S} \\
& D_{is} - D_{js} \geq H - M x_{ij}, & \forall i,j \in \mathcal{T}, i \neq j, \forall s \in \mathcal{S} \\
& 0 \leq A_{is} \leq T, 0 \leq D_{is} \leq T, & \forall i \in \mathcal{T}, \forall s \in \mathcal{S} \\
& x_{ij} \in \{0, 1\}, & \forall i,j \in \mathcal{T}, i \neq j
\end{align}
$$

## Model Extensions

### 1. Train Priority Weights

$$
\text{minimize} \quad Z = \sum_{i \in \mathcal{T}} w_i (A_{i,s_{|\mathcal{S}|}} - D_{i,s_1})
$$

where $w_i > 0$ is the priority weight for train $i$.

### 2. Station Capacity Constraints

$$
\sum_{i \in \mathcal{T}} y_{is} \leq C_s, \quad \forall s \in \mathcal{S}
$$

where $y_{is} \in \{0,1\}$ indicates if train $i$ occupies station $s$ and $C_s$ is station capacity.

### 3. Maintenance Window Constraints

$$
A_{is} \geq m_s^{start} \text{ or } D_{is} \leq m_s^{end}, \quad \forall i \in \mathcal{T}, \forall s \in \mathcal{S}_{maint}
$$

where $\mathcal{S}_{maint}$ is the set of stations with maintenance windows.

### 4. Passenger Transfer Constraints

$$
D_{is} + \tau_{transfer} \leq A_{js}, \quad \forall (i,j,s) \in \mathcal{Transfer}
$$

where $\mathcal{Transfer}$ is the set of passenger transfer connections and $\tau_{transfer}$ is minimum transfer time.
