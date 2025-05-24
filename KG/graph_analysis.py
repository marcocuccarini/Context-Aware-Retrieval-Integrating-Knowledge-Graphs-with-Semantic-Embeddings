# List of JSON filenames

import json
import os
import networkx as nx

def analyze_graph_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()

    for edge in data:
        from_id = edge['from_id']
        to_id = edge['to_id']
        G.add_edge(from_id, to_id)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    density = nx.density(G)
    num_weakly_cc = nx.number_weakly_connected_components(G)
    num_strongly_cc = nx.number_strongly_connected_components(G)

    print(f"📄 Analysis for {os.path.basename(file_path)}:")
    print(f"  ➤ Nodes: {num_nodes}")
    print(f"  ➤ Edges: {num_edges}")
    print(f"  ➤ Average Degree: {avg_degree:.2f}")
    print(f"  ➤ Graph Density: {density:.4f}")
    print(f"  ➤ Weakly Connected Components: {num_weakly_cc}")
    print(f"  ➤ Strongly Connected Components: {num_strongly_cc}")
    print("-" * 50)

# 📝 List your JSON file paths here
json_files = ['Florence_Ps_KG.json', 'Venice_Ps_KG.json', 'Rome_Ps_KG.json']  # <-- Add your file names here


# 🔍 Run analysis for each file
for file in json_files:
    analyze_graph_from_file(file)
