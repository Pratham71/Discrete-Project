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

#ALGORITHMS (CORE LOGIC)
class UnionFind:
    """Helper for Kruskal's Algorithm."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
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

def bubble_sort_edges(edges):
    """Manual Bubble Sort implementation for educational demonstration."""
    # edges is a list of tuples (u, v, w)
    n = len(edges)
    # We sort by weight, which is index 2
    for i in range(n):
        for j in range(0, n-i-1):
            if edges[j][2] > edges[j+1][2]:
                edges[j], edges[j+1] = edges[j+1], edges[j]
    return edges

def run_kruskals(num_nodes, adj_df, use_bubble_sort=False):
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
    
    if use_bubble_sort:
        # educational slow sort
        sorted_edges = bubble_sort_edges(edges[:]) # copy list
        full_log.append("Sorting edges using Bubble Sort (O(E^2))...")
    else:
        # standard python Timsort
        sorted_edges = sorted(edges, key=lambda x: x[2])
        full_log.append("Sorting edges using Timsort (O(E log E))...")

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

def run_prims(num_nodes, adj_df):
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

#UI (STREAMLIT)
st.set_page_config(layout="wide", page_title="Network Design Tool", page_icon="🌐")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Optimal Network Topology Designer")
st.markdown("**Discrete Math Project: Minimum Spanning Trees (MST)**")

#SCENARIO DATASETS
def get_scenario_data(scenario):
    """Returns (adj_df, num_nodes, unit_cost, label, layout_type, algo_pref)"""
    if scenario == "Problem 1: University Backbone":
        n = 6
        cols = ["Server Room", "Dept CS", "Dept Eng", "Library", "Admin", "Dorms"]
        matrix = np.zeros((n, n))
        connections = [
            (0,1,10), (0,2,15), (0,4,20), 
            (1,2,12), (1,3,15), 
            (2,4,10), (2,5,30),
            (3,5,10), (4,5,12)
        ]
        for u,v,w in connections:
            matrix[u][v] = matrix[v][u] = w
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 2000, "m (Fiber)", "Star Network", "Prim's Algorithm"

    elif scenario == "Problem 2: Regional Power Grid":
        n = 8
        cols = [f"City {i+1}" for i in range(n)]
        cols[0] = "Power Plant" 
        matrix = np.zeros((n, n))
        connections = [
            (0,1,50), (0,2,80), 
            (1,3,40), (1,2,60),
            (2,4,70), (3,5,30),
            (4,5,50), (4,6,90),
            (5,7,20), (6,7,40)
        ]
        for u,v,w in connections:
            matrix[u][v] = matrix[v][u] = w
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 1000, "km (HV Line)", "Ring Network", "Kruskal's Algorithm"

    elif scenario == "Problem 3: Rural Water Supply":
        n = 8 
        cols = ["Reservoir"] + [f"Village {i}" for i in range(1, n)]
        matrix = np.zeros((n, n))
        connections = [
            (0,1,10), (0,2,12), 
            (1,3,15), (1,4,18), 
            (2,5,20), (2,6,22), 
            (4,7,25),            
            (3,5,40), (6,7,50)   
        ]
        for u,v,w in connections:
            matrix[u][v] = matrix[v][u] = w
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 15000, "km (Pipe)", "Random Mesh", "Kruskal's Algorithm"

    elif scenario == "Problem 4: Circuit Board Wiring":
        n = 10
        cols = [f"Pin {i+1}" for i in range(n)]
        matrix = np.zeros((n, n))
        random.seed(101) 
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() > 0.4: 
                    w = random.randint(1, 20)
                    matrix[i][j] = matrix[j][i] = w
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df, n, 0.5, "mm (Trace)", "Grid Lattice", "Prim's Algorithm"
    
    else:
        return None, 0, 0, "", "", ""

#Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    scenario_mode = st.selectbox(
        "Select Scenario", 
        ["Custom Playground", "Problem 1: University Backbone", "Problem 2: Regional Power Grid", 
         "Problem 3: Rural Water Supply"]
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
        
        # BUBBLE SORT TOGGLE (Only for Kruskal's in Custom Mode)
        use_bubble = False
        if algo == "Kruskal's Algorithm":
            use_bubble = st.checkbox("Use Bubble Sort (Education Mode)", help="Simulate slower O(N^2) sorting to benchmark performance impact.")
        
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    else:
        st.info(f"🔒 **{scenario_mode}** active.")
        df_fixed, n_fixed, cost_fixed, label_fixed, layout_fixed, algo_fixed = get_scenario_data(scenario_mode)
        st.caption(f"Nodes: {n_fixed} | Layout: {layout_fixed}")
        st.caption(f"Standard: {label_fixed} | Cost: ${cost_fixed}/unit")
        algo = st.radio("Algorithm", ["Prim's Algorithm", "Kruskal's Algorithm"], index=0 if "Prim" in algo_fixed else 1)
        topo_type = layout_fixed
        num_nodes = n_fixed
        unit_cost = cost_fixed
        unit_label = label_fixed
        seed = 42
        use_bubble = False # Disable bubble for fixed scenarios to keep them snappy

    # Complexity Display
    if algo == "Prim's Algorithm":
        st.info("⏱️ **Complexity:** $O(E \log V)$\n\n*Best for dense networks.*")
    else:
        base_complexity = "$O(E \log E)$" if not use_bubble else "$O(E^2)$ (Bubble Sort)"
        st.info(f"⏱️ **Complexity:** {base_complexity}\n\n*Standard Kruskal's is best for sparse networks.*")

#Graph Generator Logic
def get_graph_matrix(n, type, seed, density=0.7):
    random.seed(seed)
    matrix = np.zeros((n, n))
    cols = [f"N{i}" for i in range(n)]
    
    if type == "Random Mesh":
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density: 
                    w = random.randint(5, 50)
                    matrix[i][j] = matrix[j][i] = w
    elif type == "Ring Network":
        nodes = list(range(n))
        random.shuffle(nodes)
        for k in range(n):
            u, v = k, (k+1)%n
            w = random.randint(5, 50)
            matrix[u][v] = matrix[v][u] = w
            if random.random() > 0.8:
                matrix[u][(u+2)%n] = matrix[(u+2)%n][u] = random.randint(10, 60)
    elif type == "Star Network":
        center = 0
        for i in range(1, n):
            w = random.randint(5, 50)
            matrix[center][i] = matrix[i][center] = w
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

#State Management
if scenario_mode == "Custom Playground":
    if 'graph_params' not in st.session_state:
        st.session_state['graph_params'] = {}
    current_params = {'n': num_nodes, 'type': topo_type, 'seed': seed, 'mode': 'custom'}
    if current_params != st.session_state['graph_params']:
        st.session_state['adj_df'] = get_graph_matrix(num_nodes, topo_type, seed)
        st.session_state['graph_params'] = current_params
else:
    st.session_state['adj_df'] = df_fixed
    st.session_state['graph_params'] = {'mode': scenario_mode}

#Layout
tab_design, tab_perf, tab_theory = st.tabs(["🛠️ Design Studio", "🚀 Algorithm Benchmark", "📚 Theory"])

with tab_design:
    if scenario_mode == "Problem 1: University Backbone":
        st.markdown("### 🏫 Problem 1: Campus Fiber Backbone")
    elif scenario_mode == "Problem 2: Regional Power Grid":
        st.markdown("### ⚡ Problem 2: Regional Power Distribution")
    elif scenario_mode == "Problem 3: Rural Water Supply":
        st.markdown("### 💧 Problem 3: Rural Water Supply Network") 

    df = st.session_state['adj_df']
    
    # Run loop 50 times for timing
    perf_times = []
    if algo == "Prim's Algorithm":
        mst_edges, total_cost, full_log, step_log, _ = run_prims(num_nodes, df)
    else:
        mst_edges, total_cost, full_log, step_log, _ = run_kruskals(num_nodes, df, use_bubble)
        
    for _ in range(50):
        if algo == "Prim's Algorithm":
            _, _, _, _, t = run_prims(num_nodes, df)
        else:
            _, _, _, _, t = run_kruskals(num_nodes, df, use_bubble)
        perf_times.append(t)
        
    avg_exec = sum(perf_times) / len(perf_times)
    total_exec = sum(perf_times)

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Length", f"{int(total_cost)} {unit_label}")
    
    if scenario_mode == "Custom Playground":
        cable_type = media_type.split('(')[0].strip()
    else:
        cable_type = unit_label.split(' ')[1] if ' ' in unit_label else unit_label
        
    m2.metric("Project Cost", f"${total_cost * unit_cost:,}", f"{cable_type} @ ${unit_cost}/{unit_label}", delta_color="off")
    m3.metric("Avg Time", f"{avg_exec:.3f} ms")
    m4.metric("Total Benchmark", f"{total_exec:.2f} ms")
    
    is_connected = len(mst_edges) >= num_nodes - 1
    if not is_connected:
        st.error("⚠️ **Status: Partitioned / Disconnected** - Some nodes are unreachable!")

    # Visualization
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("Connection Matrix")
        is_disabled = scenario_mode != "Custom Playground"
        edited_df = st.data_editor(df, height=300, key="editor", disabled=is_disabled, column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=100, format="%d") for c in df.columns})
        
        with st.expander("Visualize Matrix as Heatmap"):
            fig_matrix, ax_matrix = plt.subplots(figsize=(6, 4))
            cax = ax_matrix.matshow(df.values, cmap="Blues")
            fig_matrix.colorbar(cax)
            st.pyplot(fig_matrix)

    with col_viz:
        st.subheader("Topology Visualization")
        step = st.slider("Construction Step", 0, len(mst_edges), len(mst_edges), key="step_slider")
        if step > 0:
            log_idx = min(step-1, len(step_log)-1)
            st.info(f"**Step {step}:** {step_log[log_idx]}")
        else:
            st.write("Adjust slider to start visualization.")

        visible_edges = mst_edges[:step]
        G = nx.from_numpy_array(edited_df.values)
        node_names = list(df.columns)
        labels = {i: name for i, name in enumerate(node_names)}
        
        if topo_type == "Ring Network": pos = nx.circular_layout(G)
        elif topo_type == "Star Network": pos = nx.shell_layout(G, nlist=[[0], list(range(1, len(G.nodes())))])
        elif topo_type == "Grid Lattice":
            side = int(np.ceil(np.sqrt(len(G.nodes()))))
            pos = {}
            for i, node in enumerate(G.nodes()):
                pos[node] = np.array([i % side, -(i // side)])
        else: pos = nx.spring_layout(G, seed=seed)
        
        mst_G = nx.Graph()
        mst_G.add_nodes_from(G.nodes())
        mst_G.add_weighted_edges_from(visible_edges)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', style='dashed', width=1, ax=ax)
        if visible_edges:
            nx.draw_networkx_edges(mst_G, pos, edge_color='#2E86C1', width=3, ax=ax)
            nx.draw_networkx_edge_labels(mst_G, pos, edge_labels={(u,v): w for u,v,w in visible_edges}, font_size=8, ax=ax)
        
        full_mst = nx.Graph()
        full_mst.add_weighted_edges_from(mst_edges)
        deg = dict(full_mst.degree())
        max_degree = max(deg.values()) if deg else 0
        node_cols = []
        node_sizes = []
        for n in G.nodes():
            d = deg.get(n, 0)
            is_force_core = (num_nodes <= 8 and d == max_degree and d > 1)
            if d >= 3 or is_force_core: 
                node_cols.append('#FF4B4B')
                node_sizes.append(900)
            elif d == 2: 
                node_cols.append('#F1C40F')
                node_sizes.append(600)
            else: 
                node_cols.append('#2ECC71')
                node_sizes.append(400)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_cols, edgecolors='#2C3E50', ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
        
        # Legend below
        legend_elements = [
            Line2D([0], [0], color='#2E86C1', lw=3, label='Active Backbone'),
            Line2D([0], [0], marker='o', color='w', label='Core Router', markerfacecolor='#FF4B4B', markersize=10, markeredgecolor='#2C3E50'),
            Line2D([0], [0], marker='o', color='w', label='Dist. Switch', markerfacecolor='#F1C40F', markersize=8, markeredgecolor='#2C3E50'),
            Line2D([0], [0], marker='o', color='w', label='Access Point', markerfacecolor='#2ECC71', markersize=6, markeredgecolor='#2C3E50')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize='small')
        ax.margins(0.2)
        ax.axis('off')
        st.pyplot(fig)

with tab_perf:
    st.header("Algorithm Performance Analysis")
    st.write("Compare how Prim's and Kruskal's scale as the network size (N) grows.")
    
    benchmark_density = st.slider("Graph Density (Connectivity)", 0.1, 1.0, 0.7, 0.1, key="bench_density")
    use_bubble_bench = st.checkbox("Include Bubble Sort Kruskal's in Benchmark", key="bubble_bench")
    
    if st.button("🚀 Run Benchmark Simulation"):
        results = {'Nodes': [], 'Prim (ms)': [], 'Kruskal (Timsort) (ms)': []}
        if use_bubble_bench:
            results['Kruskal (Bubble) (ms)'] = []
            
        progress = st.progress(0)
        test_sizes = [10, 20, 30, 40, 50, 75, 100]
        
        for i, n in enumerate(test_sizes):
            df_test = get_graph_matrix(n, "Random Mesh", 42, density=benchmark_density)
            
            _, _, _, _, t_prim = run_prims(n, df_test)
            _, _, _, _, t_kruskal = run_kruskals(n, df_test, use_bubble_sort=False)
            
            results['Nodes'].append(n)
            results['Prim (ms)'].append(t_prim)
            results['Kruskal (Timsort) (ms)'].append(t_kruskal)
            
            if use_bubble_bench:
                _, _, _, _, t_bubble = run_kruskals(n, df_test, use_bubble_sort=True)
                results['Kruskal (Bubble) (ms)'].append(t_bubble)
                
            progress.progress((i+1)/len(test_sizes))
            
        res_df = pd.DataFrame(results).set_index('Nodes')
        st.line_chart(res_df)
        st.dataframe(res_df)
        
        if use_bubble_bench:
            st.warning("Notice how **Kruskal (Bubble)** shoots up exponentially ($O(E^2)$) compared to the standard sort ($O(E \log E)$).")

with tab_theory:
    st.header("Graph Theory & Algorithms")
    st.markdown(r"""
    ### 1. Sorting Algorithms in MST
    * **Timsort (Python Standard):** $O(E \log E)$. Highly optimized C implementation. Fast for most real-world data.
    * **Bubble Sort:** $O(E^2)$. Simple but inefficient. Useful for educational comparison to show why optimization matters.
    
    ### 2. Complexity Analysis
    | Algorithm | Time Complexity | Best For |
    | :--- | :--- | :--- |
    | **Prim's** | $O(E + V \log V)$ | **Dense Graphs** (Lots of cables/roads) |
    | **Kruskal's** | $O(E \log E)$ | **Sparse Graphs** (Few connections) |
    
    ### 3. Real World Application
    * **Spanning Tree Protocol (STP):** Used in Ethernet switches (IEEE 802.1D) to disable redundant links and prevent broadcast radiation loops.
    """)