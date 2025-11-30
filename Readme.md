# 🌐 Optimal Network Topology Designer

## Minimum Spanning Tree (MST) Visualizer using Prim’s & Kruskal’s Algorithms
This application provides an interactive way to design, analyze, and optimize network layouts using **Minimum Spanning** Tree algorithms.

It is especially useful for academic demonstrations in **Discrete Mathematics, Graph Theory, Computer Networks, and Algorithm Design**.

---
# 📌 Project Overview
## The tool simulates real-world network design problems:

- Campus backbone cabling (LAN / fiber)
- Regional electrical power grid
- Rural water distribution pipelines
- Circuit board (PCB) wiring

## Users can:

- Choose predefined scenarios or generate custom graphs
- Select between Prim’s or Kruskal’s algorithm
- Watch MST construction step-by-step
- Calculate total network cost based on unit cable price
- Compare algorithm performance across different graph sizes
- View network roles: Core, Distribution, Access

---

# 🧠 Key Concepts
| Concept                     | Meaning |
|-----------------------------|--------|
| Spanning Tree               |   A connected acyclic subgraph containing all nodes     |
| MST (Minimal Spanning Tree) |A spanning tree with minimum total edge weight|
|Prim’s Algorithm|Node-based greedy selection|
|Kruskal’s Algorithm|Edge-based greedy selection|

## Algorithm Complexity
|Algorithm| Complexity | Recommended Use |
|-----------------------------|------------|-----------------|
|Prim’s| O(E log V) | Dense networks  |
|Kruskal’s| O(E log E) | Sparse networks                |

---
# 🚀 Features
- Real-time MST visualization
- Adjustable animation step slider
- Custom adjacency matrix editor
- Path highlight mode (shortest MST path between nodes)
- Cluster mode (remove K–1 highest cost edges)
- Automatic role classification based on degree
- Algorithm benchmark simulator
- Built-in theory tab for quick revision
---
# 📂 Installation

### 1. Clone / download the project
```shell
git clone <repository-url>
cd <project-folder>
```

### 2. Install dependencies
```shell
pip install -r requirements.txt
```

### 3. Run the application
```shell
streamlit run main.py
```

---
# 🏗 Project Architecture (Simplified)

```
main.py
 ├─ MST Algorithms
 │   ├─ run_prims()
 │   └─ run_kruskals()
 ├─ UnionFind class (cycle detection for Kruskal)
 ├─ Scenario dataset generator
 ├─ Custom topology graph generator
 ├─ Benchmark engine (timing 50 iterations)
 └─ Streamlit UI (tabs: Design • Benchmark • Theory)
```

---
