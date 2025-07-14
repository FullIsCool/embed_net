import numpy as np
import networkx as nx
from collections import deque

def load_swc(swc_file):
    nodes = []
    with open(swc_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            node = {
                'id': int(parts[0]),
                'type': int(parts[1]),
                'x': float(parts[2]),
                'y': float(parts[3]),
                'z': float(parts[4]),
                'radius': float(parts[5]),
                'parent': int(parts[6])
            }
            nodes.append(node)
    return nodes


def prune_small_branch(nodes, min_l=2):
    node_dict = {node["id"]: node for node in nodes}
    valid_ids = set(node_dict.keys())
    children = {}
    for node in node_dict.values():
        pid = node["parent"]
        if pid in valid_ids:
            children.setdefault(pid, []).append(node["id"])
    
    leaves = [node for nid, node in node_dict.items() if nid not in children]
    
    removal_set = set()
    for leaf in leaves:
        branch_nodes = [leaf["id"]]  
        current = leaf
        branch_point = None  
        while True:
            pid = current["parent"]
            if pid == -1:  
                branch_point = current
                break
            parent = node_dict.get(pid, None)
            if parent is None:
                break
            parent_children = children.get(pid, [])
            if len(parent_children) > 1 or parent["parent"] == -1:
                branch_point = parent
                break
            branch_nodes.append(parent["id"])
            current = parent
        if branch_point is not None:
            dx = leaf["x"] - branch_point["x"]
            dy = leaf["y"] - branch_point["y"]
            dz = leaf["z"] - branch_point["z"]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            if distance < min_l:
                removal_set.update(branch_nodes)
    
    pruned_dict = {nid: node for nid, node in node_dict.items() if nid not in removal_set}
    pruned_nodes = list(pruned_dict.values())
    return pruned_nodes



def connect_close_leaves(nodes, threshold):
    node_dict = {node["id"]: node for node in nodes}

    G = nx.Graph()
    for node in nodes:
        nid = node["id"]
        pos = (node["x"], node["y"], node["z"])
        G.add_node(nid, pos=pos)
        pid = node["parent"]
        if pid != -1:
            G.add_edge(nid, pid)
            
    leaves = [n for n, d in G.degree() if d == 1]
    print(f"Found {len(leaves)} leaves in the graph.")
    for i in range(len(leaves)):
        pos_i = np.array(G.nodes[leaves[i]]["pos"])
        for j in range(i+1, len(leaves)):
            pos_j = np.array(G.nodes[leaves[j]]["pos"])
            dist = np.linalg.norm(pos_i - pos_j)
            if dist < threshold:
                G.add_edge(leaves[i], leaves[j])
                
    components = list(nx.connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components]
    print(f"Found {len(subgraphs)} connected components in the graph.")
    trees = []
    for i, sg in enumerate(subgraphs):
        max_degree_node = max(sg.degree, key=lambda x: x[1])[0]
        max_degree = sg.degree[max_degree_node]
            
        T = nx.bfs_tree(sg, source=max_degree_node)
        index_map = {node: idx + 1 for idx, node in enumerate(T.nodes())}
        tree_nodes = []
        for node in T.nodes():
            z, y, x = node_dict[node]["z"], node_dict[node]["y"], node_dict[node]["x"]
            parent = -1
            if node != max_degree_node:
                parent_node = list(T.predecessors(node))[0]
                parent = index_map[parent_node]
            tree_nodes.append({
                "id": index_map[node],
                "type": 1 if node == max_degree_node else 3,
                "x": x,
                "y": y,
                "z": z,
                "radius": 1.0,
                "parent": parent
            })
        trees.append(tree_nodes)
        
    swc_data = []
    node_offset = 0

    for tree in trees:
        for node in tree:
            swc_data.append({
                "id": node["id"] + node_offset,
                "type": node["type"],
                "x": node["x"],
                "y": node["y"],
                "z": node["z"],
                "radius": node["radius"],
                "parent": node["parent"] + node_offset if node["parent"] != -1 else -1
            })
        node_offset += len(tree)

    return swc_data


def sleek(nodes, n):
    node_dict = {node["id"]: node for node in nodes}
    valid_ids = set(node_dict.keys())
    children = {}
    for node in node_dict.values():
        pid = node["parent"]
        if pid in valid_ids:
            children.setdefault(pid, []).append(node["id"])
    
    branches = {nid for nid, node in node_dict.items() if len(children.get(nid, [])) > 1}
    leaves = {nid for nid, node in node_dict.items() if nid not in children}
    starts = {nid for nid, node in node_dict.items() if node['parent'] == -1}
    
    def select(node):
        return (node['type'] == 1) or (node['parent'] == -1) or (node['id'] in branches) or (node['id'] in leaves)
    
    base_nodes = {nid for nid, node in node_dict.items() if select(node)}
    
    selected_nodes = base_nodes.copy()
    counter = 0
    stack = deque(starts)

    while stack:

        current = stack.pop()
        if current in base_nodes:
            counter = 0
        if counter % n == 0 :
            selected_nodes.add(current)

        counter += 1
        for cid in children.get(current, []):
            stack.append(cid)
            
    print("len(selected_nodes):", len(selected_nodes))
    #print(sorted(list(base_nodes))[:100])
    #print(sorted(list(selected_nodes))[:100])
    new_nodes = []
    ddd=0
    for node in nodes:
        if (node['id'] in selected_nodes):
            ddd+=1
            new_node = node.copy()
            pid = node['parent']
            record = [node['id'], pid]
            c = 0
            while(pid not in selected_nodes):
                if pid == -1:
                    break
                parent_node = node_dict[pid]
                pid = parent_node['parent']
                record.append(pid)
                c+=1
                if c > n+2:
                    print(record)
                    raise ValueError("Error")

            new_node['parent'] = pid
            new_nodes.append(new_node)
        


    return new_nodes

def save_swc(nodes, output_file):
    if len(nodes) == 0:
        with open(output_file, 'w') as f:
            f.write("# Generated SWC file\n")
            f.write("# No nodes to save.\n")
        return
    if isinstance(nodes[0], list):
        with open(output_file, 'w') as f:
            f.write("# Generated SWC file\n")
            for node in nodes:
                line = f"{node[0]} {node[1]} {node[2]} {node[3]} {node[4]} {node[5]} {node[6]}\n"
                f.write(line)
        return 
    if isinstance(nodes[0], dict):
        with open(output_file, 'w') as f:
            f.write("# Generated SWC file\n")
            for node in nodes:
                line = f"{node['id']} {node['type']} {node['x']} {node['y']} {node['z']} {node['radius']} {node['parent']}\n"
                f.write(line)
        return
    
    
import os
import glob
def main1():
    input_dir = "./dataset_etv133/block_128/swc_pred"
    output_dir = "./dataset_etv133/block_128/swc_pruned"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating directory: {output_dir}")


    for i, swc_file in enumerate(sorted(glob.glob(os.path.join(input_dir, "*.swc")))):
        save_file = os.path.join(output_dir, os.path.basename(swc_file))
        print(f"Processing {swc_file}...")
        nodes = load_swc(swc_file)
        #print("len(nodes):", len(nodes))
        nodes = prune_small_branch(nodes, min_l=5)
        nodes = connect_close_leaves(nodes, threshold=5)
        nodes = sleek(nodes, n=5)
        #print("len(nodes):", len(nodes))
        save_swc(nodes, save_file)

if __name__ == "__main__":
    main1()
