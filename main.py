import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq
import numpy as np
import pandas as pd
import io
import time
from matplotlib.lines import Line2D
from typing import Any, Literal, Union

# PART 1: ALGORITHMS (CORE LOGIC)

class UnionFind:
    """Helper for Kruskal's Algorithm."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1
            return True
        return False

def run_kruskals(num_nodes, adj_df) -> tuple[list, Any | Literal[0], list, list, float]:
    """Kruskal's Algorithm: Edge-based greedy approach."""
    edges = []
    for r in range(num_nodes):
        for c in range(r + 1, num_nodes):
            weight = adj_df.iloc[r, c]
            if weight > 0:
                edges.append((r, c, weight))
    
    mst_edges = []
    total_cost = 0
    full_log = []
    step_log = []
    
    start_time = time.perf_counter()
    sorted_edges = sorted(edges, key=lambda x: x[2])
    uf = UnionFind(num_nodes)
    
    full_log.append(f"Analyzing {len(edges)} potential cable routes...")
    
    count = 0
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_cost += w
            msg = f"Global Cheapest Link found: {u} <-> {v} (Cost: {w})"
            full_log.append(f"Step {count+1}: ✅ {msg}")
            step_log.append(msg)
            count += 1
        else:
            full_log.append(f"Skipped {u}-{v} (Cost: {w}) - Loop detected in Set {uf.find(u)}")
            
        if count == num_nodes - 1:
            break
            
    exec_time = (time.perf_counter() - start_time) * 1000 # ms
    return mst_edges, total_cost, full_log, step_log, exec_time

def run_prims(num_nodes, adj_df) -> tuple[list, Any | Literal[0], list, list, float]:
    """Prim's Algorithm: Node-based greedy approach."""
    mst_edges = []
    total_cost = 0
    full_log = []
    step_log = []
    
    start_time = time.perf_counter()
    visited = {0}
    min_heap = []
    
    # Start Prim's from Node 0
    for neighbor in range(num_nodes):
        weight = adj_df.iloc[0, neighbor]
        if weight > 0:
            heapq.heappush(min_heap, (weight, 0, neighbor))
            
    full_log.append("Starting Network Design at Central Node 0...")
    
    step_count = 1
    while len(visited) < num_nodes and min_heap:
        weight, u, v = heapq.heappop(min_heap)
        
        if v in visited:
            continue
            
        visited.add(v)
        mst_edges.append((u, v, weight))
        total_cost += weight
        
        msg = f"Extended Network: Connected {v} to {u} (Cost: {weight})"
        full_log.append(f"Step {step_count}: ✅ {msg}")
        step_log.append(msg)
        step_count += 1
        
        for next_node in range(num_nodes):
            new_weight = adj_df.iloc[v, next_node]
            if next_node not in visited and new_weight > 0:
                heapq.heappush(min_heap, (new_weight, v, next_node))
                
    exec_time = (time.perf_counter() - start_time) * 1000 # ms
    return mst_edges, total_cost, full_log, step_log, exec_time

# PART 2: UI (STREAMLIT)

st.set_page_config(layout="wide", page_title="Network Design Tool", page_icon="🌐")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Optimal Network Topology Designer")
st.markdown("**Discrete Math Project: Minimum Spanning Trees (MST)**")

# SCENARIO DATASETS
def get_scenario_data(scenario):
    """Returns (adj_df, num_nodes, unit_cost, label, layout_type, algo_pref)"""
    
    if scenario == "Problem 1: University Backbone":
        # TRUE STAR TOPOLOGY
        n = 6
        cols = ["Server Room", "Dept CS", "Dept Eng", "Library", "Admin", "Dorms"]
        matrix = np.zeros((n, n))
        connections = [
            (0,1,5),  (0,2,8),  (0,3,12), (0,4,6),  (0,5,15), # Star connections
            (1,2,20), (2,3,20), (4,5,25) # Expensive cross-links
        ]
        for u,v,w in connections:
            matrix[u][v] = matrix[v][u] = w
            
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 2000, "m (Fiber)", "Star Network", "Prim's Algorithm"

    elif scenario == "Problem 2: Regional Power Grid":
        # TRUE RING TOPOLOGY
        n = 8
        cols = [f"City {i+1}" for i in range(n)]
        cols[0] = "Power Plant" 
        matrix = np.zeros((n, n))
        for i in range(n):
            u, v = i, (i+1)%n
            w = 50 + (i*5) # varying costs
            matrix[u][v] = matrix[v][u] = w
        
        matrix[0][4] = matrix[4][0] = 120
        matrix[2][6] = matrix[6][2] = 150
            
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 1000, "km (HV Line)", "Ring Network", "Kruskal's Algorithm"

    elif scenario == "Problem 3: Rural Water Supply":
        # TRUE BRANCHED/TREE TOPOLOGY (Corrected)
        n = 8 # Reservoir + 7 Villages
        cols = ["Reservoir"] + [f"Village {i}" for i in range(1, n)]
        matrix = np.zeros((n, n))
        
        # Logical Water Flow: Reservoir -> Main A/B -> Sub-villages
        connections = [
            (0,1,10), (0,2,12),  # Reservoir to 2 Main Villages
            (1,3,15), (1,4,18),  # V1 feeds V3, V4
            (2,5,20), (2,6,22),  # V2 feeds V5, V6
            (4,7,25),            # V4 feeds V7 (Remote)
            (3,5,40), (6,7,50)   # Backup loops (Expensive)
        ]
        for u,v,w in connections:
            matrix[u][v] = matrix[v][u] = w
            
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        # Changed layout from "Ring Network" to "Random Mesh" (Force-directed) for organic tree look
        return df, n, 15000, "km (Pipe)", "Random Mesh", "Kruskal's Algorithm"

    elif scenario == "Problem 4: Circuit Board Wiring":
        # TRUE GRID TOPOLOGY
        n = 9 
        cols = [f"Pin {i+1}" for i in range(n)]
        matrix = np.zeros((n, n))
        side = 3
        for i in range(n):
            if (i + 1) % side != 0 and (i + 1) < n:
                w = 1 + (i % 2) 
                matrix[i][i+1] = matrix[i+1][i] = w
            if (i + side) < n:
                w = 1 + (i % 2)
                matrix[i][i+side] = matrix[i+side][i] = w
            if i < n-2 and random.random() > 0.8:
                matrix[i][i+2] = matrix[i+2][i] = 10 
        
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 0.5, "mm (Trace)", "Grid Lattice", "Prim's Algorithm"
    
    else:
        return None, 0, 0, "", "", ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. SCENARIO SELECTOR
    scenario_mode = st.selectbox(
        "Select Scenario", 
        [
            "Custom Playground", 
            "Problem 1: University Backbone", 
            "Problem 2: Regional Power Grid",
            "Problem 3: Rural Water Supply",
            "Problem 4: Circuit Board Wiring"
        ],
        help="Choose 'Custom' to generate random graphs, or select a Textbook Problem."
    )
    
    if scenario_mode == "Custom Playground":
        st.subheader("1. Topology Settings")
        topo_type = st.selectbox("Graph Topology", ["Random Mesh", "Ring Network", "Star Network", "Grid Lattice"], key="topo_type_input")
        num_nodes = st.slider("Number of Nodes", 3, 20, 8, key="num_nodes_input")
        seed = st.number_input("Random Seed", value=42, key="seed_input")
        
        st.subheader("2. Project Settings")
        media_type = st.selectbox("Cabling Standard", 
            ["Custom Cost", "Cat6a Ethernet (LAN - $5/m)", "Single-Mode Fiber (WAN - $2000/km)", 
             "Submarine Cable (Global - $50,000/km)", "High Voltage Power (Grid - $1000/unit)"]
        )
        if "Cat6a" in media_type: unit_cost, unit_label = 5, "m"
        elif "Fiber" in media_type: unit_cost, unit_label = 2000, "km"
        elif "Submarine" in media_type: unit_cost, unit_label = 50000, "km"
        elif "Power" in media_type: unit_cost, unit_label = 1000, "km"
        else: unit_cost, unit_label = st.number_input("Cost per Unit ($)", value=100), "units"
        
        algo = st.radio("Algorithm", ["Prim's Algorithm", "Kruskal's Algorithm"])
        
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    else:
        # Load Pre-defined Scenario Data
        st.info(f"🔒 **{scenario_mode}** active.")
        df_fixed, n_fixed, cost_fixed, label_fixed, layout_fixed, algo_fixed = get_scenario_data(scenario_mode)
        
        # Lock controls but show values
        st.caption(f"Nodes: {n_fixed} | Layout: {layout_fixed}")
        st.caption(f"Standard: {label_fixed} | Cost: ${cost_fixed}/unit")
        
        # Allow Algo Switching but suggest default
        algo = st.radio("Algorithm", ["Prim's Algorithm", "Kruskal's Algorithm"], index=0 if "Prim" in algo_fixed else 1)
        if algo != algo_fixed:
            st.warning(f"Note: The textbook recommends **{algo_fixed}** for this problem.")
            
        # Set variables for main logic
        topo_type = layout_fixed
        num_nodes = n_fixed
        unit_cost = cost_fixed
        unit_label = label_fixed
        seed = 42 # Fixed seed for scenarios

    # Big O Display Side (Optional - also in main)
    if algo == "Prim's Algorithm":
        st.info("⏱️ **Complexity:** $O(E \log V)$\n\n*Best for dense networks.*")
    else:
        st.info("⏱️ **Complexity:** $O(E \log E)$\n\n*Best for sparse networks.*")

# Graph Generator Logic (For Custom Mode)
def get_graph_matrix(n, type, seed) -> pd.DataFrame:
    random.seed(seed)
    matrix = np.zeros((n, n))
    cols = [f"N{i}" for i in range(n)]
    
    if type == "Random Mesh":
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() > 0.3: 
                    w = random.randint(5, 50)
                    matrix[i][j] = matrix[j][i] = w
    elif type == "Ring Network":
        nodes = list(range(n))
        random.shuffle(nodes)
        for k in range(n):
            u, v = nodes[k], nodes[(k+1)%n]
            w = random.randint(5, 50)
            matrix[u][v] = matrix[v][u] = w
            if random.random() > 0.8:
                matrix[u][nodes[(k+2)%n]] = matrix[nodes[(k+2)%n]][u] = random.randint(10, 60)
    elif type == "Star Network":
        center = 0
        for i in range(1, n):
            w = random.randint(5, 50)
            matrix[center][i] = matrix[i][center] = w
            if random.random() > 0.8:
                j = random.randint(1, n-1)
                if i != j:
                    matrix[i][j] = matrix[j][i] = random.randint(20, 80)
    elif type == "Grid Lattice":
        side = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            if (i + 1) % side != 0 and (i + 1) < n:
                w = random.randint(5, 50)
                matrix[i][i+1] = matrix[i+1][i] = w
            if (i + side) < n:
                w = random.randint(5, 50)
                matrix[i][i+side] = matrix[i+side][i] = w

    return pd.DataFrame(matrix, columns=cols, index=cols)

# Automatic State Managemen
if scenario_mode == "Custom Playground":
    # Check if params changed
    if 'graph_params' not in st.session_state:
        st.session_state['graph_params'] = {}
    
    current_params = {'n': num_nodes, 'type': topo_type, 'seed': seed, 'mode': 'custom'}
    
    if current_params != st.session_state['graph_params']:
        st.session_state['adj_df'] = get_graph_matrix(num_nodes, topo_type, seed)
        st.session_state['graph_params'] = current_params
else:
    # Always force load the fixed scenario data
    st.session_state['adj_df'] = df_fixed
    st.session_state['graph_params'] = {'mode': scenario_mode} # Track that we are in a fixed mode

# Layout
tab_design, tab_perf, tab_theory = st.tabs(["🛠️ Design Studio", "🚀 Algorithm Benchmark", "📚 Theory"])

with tab_design:
    # 0. Problem Statement (If Scenario Active)
    if scenario_mode == "Problem 1: University Backbone":
        st.markdown("""
        ### 🏫 Problem 1: Campus Fiber Backbone
        **Objective:** Connect all 6 department buildings to the Server Room (Center) using fiber optic cables. 
        **Constraint:** Minimizing total cable length reduces excavation costs.
        """)
    elif scenario_mode == "Problem 2: Regional Power Grid":
        st.markdown("""
        ### ⚡ Problem 2: Regional Power Distribution
        **Objective:** Connect 8 cities to the main Power Plant using High Voltage lines.
        **Constraint:** Use the minimum copper possible. The network is sparse due to geographic obstacles.
        """)
    elif scenario_mode == "Problem 3: Rural Water Supply":
        st.markdown("""
        ### 💧 Problem 3: Rural Water Supply Network
        **Objective:** Connect remote villages to a central Reservoir.
        **Constraint:** Excavation and piping are expensive per km. Network topology is naturally sparse.
        """)
    elif scenario_mode == "Problem 4: Circuit Board Wiring":
        st.markdown("""
        ### 🔌 Problem 4: Circuit Board (PCB) Wiring
        **Objective:** Connect 9 pins on a square component with minimum trace length.
        **Constraint:** Copper traces are cheap, but signal latency must be minimized. Components are densely packed.
        """)

    # 1. Run Algorithm & Benchmark Loop
    df = st.session_state['adj_df']
    
    # Run loop 50 times to get stable average and total time
    perf_times = []
    complexity_label = "$O(E \log V)$" if algo == "Prim's Algorithm" else "$O(E \log E)$"
    
    # Run once for data
    if algo == "Prim's Algorithm":
        mst_edges, total_cost, full_log, step_log, _ = run_prims(num_nodes, df)
    else:
        mst_edges, total_cost, full_log, step_log, _ = run_kruskals(num_nodes, df)
        
    # Run 50 times for timing
    for _ in range(50):
        if algo == "Prim's Algorithm":
            _, _, _, _, t = run_prims(num_nodes, df)
        else:
            _, _, _, _, t = run_kruskals(num_nodes, df)
        perf_times.append(t)
        
    avg_exec = sum(perf_times) / len(perf_times)
    total_exec = sum(perf_times)

    # 2. Metrics Display
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Total Length", f"{int(total_cost)} {unit_label}")
    
    # Parse cable info for the label
    if scenario_mode == "Custom Playground":
        cable_type = media_type.split('(')[0].strip()
    else:
        cable_type = unit_label.split(' ')[1] if ' ' in unit_label else unit_label
        
    m2.metric("Project Cost", f"${total_cost * unit_cost:,}", f"{cable_type} @ ${unit_cost}/{unit_label}", delta_color="off")
    
    # Execution Metrics
    m3.metric("Avg Time (50 runs)", f"{avg_exec:.3f} ms")
    m4.metric("Total Benchmark", f"{total_exec:.2f} ms")
    m5.metric("Complexity", complexity_label)

    # Status check
    is_connected = len(mst_edges) >= num_nodes - 1
    if not is_connected:
        st.error("⚠️ **Status: Partitioned / Disconnected** - Some nodes are unreachable!")

    # 3. Main Interface
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("Connection Matrix")
        st.caption(f"**Adjacency Matrix**: Values = {unit_label} cost. 0 = No Link.")
        
        # Read-only if scenario, editable if custom
        is_disabled = scenario_mode != "Custom Playground"
        edited_df = st.data_editor(
            df, 
            height=300, 
            key="editor",
            disabled=is_disabled,
            column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=100, format="%d") for c in df.columns}
        )
        
        st.divider()
        st.subheader("Analysis Tools")
        tool = st.radio("Mode", ["View Only", "Highlight Path", "Cluster Regions"], horizontal=True)
        
        path_edges = []
        cluster_colors = {}
        
        # Get Node Names list for dropdowns
        node_names = list(df.columns)
        
        if tool == "Highlight Path":
            c1, c2 = st.columns(2)
            n1_name = c1.selectbox("Start", node_names, index=0)
            n2_name = c2.selectbox("End", node_names, index=1)
            
            # Map names to indices for nx
            n1 = node_names.index(n1_name)
            n2 = node_names.index(n2_name)
            
            temp_G = nx.Graph()
            temp_G.add_weighted_edges_from(mst_edges)
            try:
                path = nx.shortest_path(temp_G, n1, n2)
                path_edges = list(zip(path, path[1:]))
                # Convert path indices back to names for display
                path_named = [node_names[i] for i in path]
                st.info(f"Path: {' → '.join(path_named)}")
            except:
                st.error("No path found.")
        
        elif tool == "Cluster Regions":
            k = st.slider("Cuts (Clusters)", 1, 5, 1)
            sorted_mst = sorted(mst_edges, key=lambda x: x[2], reverse=True)
            active_edges = [e for e in mst_edges if e not in sorted_mst[:k-1]]
            temp_G = nx.Graph()
            temp_G.add_nodes_from(range(num_nodes))
            temp_G.add_weighted_edges_from(active_edges)
            comps = list(nx.connected_components(temp_G))
            colors = ['#FF4B4B', '#1E88E5', '#FFD700', '#2ECC71', '#9B59B6']
            for idx, comp in enumerate(comps):
                for n in comp:
                    cluster_colors[n] = colors[idx % len(colors)]
            mst_edges = active_edges

    with col_viz:
        st.subheader("Topology Visualization")
        
        # ANIMATION CONTROL
        step = st.slider("Construction Step", 0, len(mst_edges), len(mst_edges), key="step_slider")
        
        # LOGIC DISPLAY PER STEP
        if step > 0:
            # Map step logic roughly to step log
            log_idx = min(step-1, len(step_log)-1)
            st.info(f"**Step {step}:** {step_log[log_idx]}")
        else:
            st.write("Adjust slider to start visualization.")

        visible_edges = mst_edges[:step]
        
        G = nx.from_numpy_array(edited_df.values)
        labels = {i: name for i, name in enumerate(node_names)}
        
        # FIXED LAYOUT LOGIC
        # Ensure we use the detected topo_type, not just fallback
        if topo_type == "Ring Network": 
            pos = nx.circular_layout(G)
        elif topo_type == "Star Network":
            # Star Layout: Node 0 center, others in circle
            pos = nx.shell_layout(G, nlist=[[0], list(range(1, len(G.nodes())))])
        elif topo_type == "Grid Lattice":
            # Grid Layout: Square grid calculation
            side = int(np.ceil(np.sqrt(len(G.nodes()))))
            pos = {}
            for i, node in enumerate(G.nodes()):
                r = i // side
                c = i % side
                pos[node] = np.array([c, -r])
        else:
            # Fallback for Random Mesh or undefined types (Problem 3 uses this now)
            pos = nx.spring_layout(G, seed=seed)
        
        mst_G = nx.Graph()
        mst_G.add_nodes_from(G.nodes())
        mst_G.add_weighted_edges_from(visible_edges)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Background
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', style='dashed', width=1, ax=ax)
        
        # MST
        if visible_edges:
            nx.draw_networkx_edges(mst_G, pos, edge_color='#2E86C1', width=3, ax=ax)
            nx.draw_networkx_edge_labels(mst_G, pos, edge_labels={(u,v): w for u,v,w in visible_edges}, font_size=8, ax=ax)
        
        if path_edges and step == len(mst_edges):
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#FF4B4B', width=5, ax=ax)
            
        # SMART NODE CLASSIFICATION
        full_mst = nx.Graph()
        full_mst.add_weighted_edges_from(mst_edges)
        deg = dict(full_mst.degree())
        max_degree = max(deg.values()) if deg else 0
        
        node_sizes = []
        node_cols = []
        
        role_counts = {"Core": [], "Dist": [], "Access": []}
        
        for n in G.nodes():
            d = deg.get(n, 0)
            is_force_core = (num_nodes <= 8 and d == max_degree and d > 1)
            
            if cluster_colors and step == len(mst_edges):
                node_cols.append(cluster_colors.get(n, 'white'))
                node_sizes.append(600)
            else:
                if d >= 3 or is_force_core:
                    node_cols.append('#FF4B4B') # Red for Core
                    node_sizes.append(900)
                    role_counts["Core"].append(node_names[n])
                elif d == 2:
                    node_cols.append('#F1C40F') # Yellow for Dist
                    node_sizes.append(600)
                    role_counts["Dist"].append(node_names[n])
                else:
                    node_cols.append('#2ECC71') # Green for Access
                    node_sizes.append(400)
                    role_counts["Access"].append(node_names[n])
            
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_cols, edgecolors='#2C3E50', ax=ax)
        # Use Custom Labels (Names instead of 0,1,2)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
        
        legend_elements = [
            Line2D([0], [0], color='#2E86C1', lw=3, label='Active Backbone'),
            Line2D([0], [0], marker='o', color='w', label='Core Router/Hub', markerfacecolor='#FF4B4B', markersize=10, markeredgecolor='#2C3E50'),
            Line2D([0], [0], marker='o', color='w', label='Dist. Switch', markerfacecolor='#F1C40F', markersize=8, markeredgecolor='#2C3E50'),
            Line2D([0], [0], marker='o', color='w', label='Access Point', markerfacecolor='#2ECC71', markersize=6, markeredgecolor='#2C3E50')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        ax.set_title(f"Construction Progress: {int((step/max(1,len(mst_edges)))*100)}%", fontsize=10, color="#2E86C1")
        ax.axis('off')
        st.pyplot(fig)
        
        with st.expander("ℹ️ Node Roles & Network Hierarchy"):
            st.write(f"**Core Routers (Red):** {', '.join(role_counts['Core'])}")
            st.write(f"**Distribution (Yellow):** {', '.join(role_counts['Dist'])}")
            st.write(f"**Access Points (Green):** {', '.join(role_counts['Access'])}")

with tab_perf:
    st.header("Algorithm Performance Analysis")
    st.write("Compare how Prim's and Kruskal's scale as the network size (N) grows.")
    
    if st.button("🚀 Run Benchmark Simulation"):
        results = {'Nodes': [], 'Prim (ms)': [], 'Kruskal (ms)': []}
        progress = st.progress(0)
        
        test_sizes = [10, 20, 30, 40, 50, 75, 100]
        for i, n in enumerate(test_sizes):
            df_test = get_graph_matrix(n, "Random Mesh", 42)
            _, _, _, _, _, t_prim = run_prims(n, df_test)
            _, _, _, _, _, t_kruskal = run_kruskals(n, df_test)
            
            results['Nodes'].append(n)
            results['Prim (ms)'].append(t_prim)
            results['Kruskal (ms)'].append(t_kruskal)
            progress.progress((i+1)/len(test_sizes))
            
        res_df = pd.DataFrame(results).set_index('Nodes')
        st.line_chart(res_df)
        st.dataframe(res_df)
        
        avg_prim = np.mean(results['Prim (ms)'])
        avg_kruskal = np.mean(results['Kruskal (ms)'])
        
        st.divider()
        st.subheader("🏆 Benchmark Conclusion")
        if avg_prim < avg_kruskal:
            st.success(f"**Winner: Prim's Algorithm** (Faster on average)")
            st.markdown("**Why?** This simulation generates **dense graphs** (many connections), where Prim's approach usually outperforms Kruskal's sorting overhead.")
        else:
            st.success(f"**Winner: Kruskal's Algorithm** (Faster on average)")
            st.markdown("**Why?** This simulation generated **sparse graphs**, where Kruskal's simple edge sorting often beats Prim's priority queue management.")

with tab_theory:
    st.header("Graph Theory & Algorithms")
    
    st.markdown("""
    ### 1. Mathematical Definition
    A **Spanning Tree** $T$ of an undirected graph $G = (V, E)$ is a subgraph that includes all of the vertices of $G$ with the minimum possible number of edges.
    
    * **Properties:**
        * Contains $|V|$ vertices and $|V|-1$ edges.
        * Connected (no isolated nodes).
        * Acyclic (no loops).
        
    A **Minimum Spanning Tree (MST)** is a spanning tree where the sum of edge weights $w(T) = \sum_{e \in T} w(e)$ is minimized.
    """)
    
    st.markdown("""
    ### 2. Fundamental Theorems
    
    #### The Cut Property (Basis for Prim's Algorithm)
    For any cut (a partition of the vertices into two disjoint sets) in the graph, if an edge has the minimum weight among all edges crossing the cut, then this edge belongs to the MST.
    
    #### The Cycle Property (Basis for Kruskal's Algorithm)
    If an edge is the heaviest edge in any cycle in the graph, then it cannot belong to the MST.
    """)

    st.markdown(r"""
    ### 3. Complexity Analysis
    | Algorithm | Time Complexity | Best For |
    | :--- | :--- | :--- |
    | **Prim's** | $O(E + V \log V)$ | **Dense Graphs** (Lots of cables/roads) |
    | **Kruskal's** | $O(E \log E)$ | **Sparse Graphs** (Few connections) |
    """)
    
    st.markdown("""
    ### 4. Real World Applications
    * **Network Design:** Telecommunications, electrical grids, water supply networks, and computer networks (LANs).
    * **Approximation Algorithms:** Used as a key step in solving the Traveling Salesperson Problem (TSP) and Steiner Tree problem.
    * **Cluster Analysis:** MSTs can be used for clustering data points (e.g., Single-linkage clustering) by dropping the $k-1$ most expensive edges.
    * **Image Segmentation:** Grouping pixels based on color similarity to identify objects in computer vision.
    * **Maze Generation:** A random MST on a grid graph produces a "perfect" maze with no loops and a unique path between any two cells.
    """)