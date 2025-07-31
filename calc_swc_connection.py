from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist
import numpy as np
from scipy.spatial import KDTree


def load_swc(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                _, node_type, x, y, z, radius, _ = map(float, parts)
                node_id, _, _, _, _, _, parent = parts
                node_id = int(node_id)
                data[node_id] = {
                    'type': int(node_type),
                    'x': x,
                    'y': y,
                    'z': z,
                    'radius': radius,
                    'parent': int(parent)
                }
    return data


def extract_swc_block(swc_file, output_file, bz, by, bx, length, scale=0.35):
    nodes = load_swc(swc_file)

    children_map = {}
    for node_id, props in nodes.items():
        parent_id = props['parent']
        if parent_id != -1:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node_id)
    
    inside_nodes = {}
    outside_nodes = {}
    
    for node_id, props in nodes.items():
        x = props['x']/scale 
        y = props['y']/scale
        z = props['z']
        local_props = props.copy()
        local_props['z'] = z - bz
        local_props['y'] = y - by
        local_props['x'] = x - bx
        
        is_inside = (0 <= local_props['z'] <= length and 
                     0 <= local_props['y'] <= length and 
                     0 <= local_props['x'] <= length)
        
        if is_inside:
            inside_nodes[node_id] = local_props
        else:
            outside_nodes[node_id] = local_props
    
    boundary_nodes = {}  
    next_id = max(nodes.keys()) + 1 if nodes else 1
    
    for inside_id, inside_props in inside_nodes.items():
        parent_id = inside_props['parent']
        if parent_id != -1 and parent_id in outside_nodes:
            outside_props = nodes[parent_id]
            
            x1, y1, z1 = inside_props['x'], inside_props['y'], inside_props['z']
            x2 = outside_props['x']/scale - bx
            y2 = outside_props['y']/scale - by 
            z2 = outside_props['z'] - bz
            
            intersections = find_intersection(x1, y1, z1, x2, y2, z2, length)
            
            if intersections:
                intersections.sort(key=lambda p: p[3])
                ix, iy, iz, _ = intersections[0]
                boundary_node = {
                    'type': inside_props['type'],
                    'x': ix,
                    'y': iy,
                    'z': iz,
                    'radius': inside_props['radius'],
                    'parent': -1  
                }
                boundary_nodes[next_id] = boundary_node
                inside_props['parent'] = next_id
                next_id += 1
            else:
                inside_props['parent'] = -1
    
    for inside_id, inside_props in inside_nodes.items():
        if inside_id in children_map:
            for child_id in children_map[inside_id]:
                if child_id in outside_nodes:
                    outside_props = nodes[child_id]
                    x1, y1, z1 = inside_props['x'], inside_props['y'], inside_props['z']
                    x2 = outside_props['x']/scale - bx
                    y2 = outside_props['y']/scale - by
                    z2 = outside_props['z'] - bz
                    intersections = find_intersection(x1, y1, z1, x2, y2, z2, length)
                    
                    if intersections:
                        intersections.sort(key=lambda p: p[3])
                        ix, iy, iz, _ = intersections[0]
                        boundary_node = {
                            'type': outside_props['type'],
                            'x': ix,
                            'y': iy,
                            'z': iz,
                            'radius': outside_props['radius'],
                            'parent': inside_id  
                        }
                        boundary_nodes[next_id] = boundary_node
                        next_id += 1
    
    local_swc = {}
    local_swc.update(inside_nodes)
    local_swc.update(boundary_nodes)
    
    children = {}
    roots = []
    for node_id, props in local_swc.items():
        parent_id = props['parent']
        if parent_id == -1:
            roots.append(node_id)
        elif parent_id in local_swc:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(node_id)
    
    ordered_nodes = []
    visited = set()
    
    def dfs(node_id):
        visited.add(node_id)
        ordered_nodes.append(node_id)
        for child_id in children.get(node_id, []):
            if child_id not in visited:
                dfs(child_id)
    
    for root in roots:
        if root not in visited:
            dfs(root)
    
    for node_id in local_swc.keys():
        if node_id not in visited:
            dfs(node_id)
    
    id_map = {}
    new_id = 1
    for old_id in ordered_nodes:
        id_map[old_id] = new_id
        new_id += 1
    
    lines = ["# Generated SWC file"]
    for old_id in ordered_nodes:
        props = local_swc[old_id]
        new_node_id = id_map[old_id]
        parent_id = props['parent']
        if parent_id != -1:
            new_parent_id = id_map.get(parent_id, -1)
        else:
            new_parent_id = -1
        
        line = f"{new_node_id} {props['type']} {props['x']} {props['y']} {props['z']} {props['radius']} {new_parent_id}"
        lines.append(line)
    
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    return len(local_swc)


def find_intersection(x1, y1, z1, x2, y2, z2, length):
    intersections = []
    if (z1 >= 0 and z2 < 0) or (z1 <= 0 and z2 > 0):
        t = z1 / (z1 - z2)
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        if 0 <= ix <= length and 0 <= iy <= length:
            intersections.append((ix, iy, 0, t))

    if (z1 >= length and z2 < length) or (z1 <= length and z2 > length):
        t = (z1 - length) / (z1 - z2)
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        if 0 <= ix <= length and 0 <= iy <= length:
            intersections.append((ix, iy, length, t))

    if (y1 >= 0 and y2 < 0) or (y1 <= 0 and y2 > 0):
        t = y1 / (y1 - y2)
        ix = x1 + t * (x2 - x1)
        iz = z1 + t * (z2 - z1)
        if 0 <= ix <= length and 0 <= iz <= length:
            intersections.append((ix, 0, iz, t))

    if (y1 >= length and y2 < length) or (y1 <= length and y2 > length):
        t = (y1 - length) / (y1 - y2)
        ix = x1 + t * (x2 - x1)
        iz = z1 + t * (z2 - z1)
        if 0 <= ix <= length and 0 <= iz <= length:
            intersections.append((ix, length, iz, t))

    if (x1 >= 0 and x2 < 0) or (x1 <= 0 and x2 > 0):
        t = x1 / (x1 - x2)
        iy = y1 + t * (y2 - y1)
        iz = z1 + t * (z2 - z1)
        if 0 <= iy <= length and 0 <= iz <= length:
            intersections.append((0, iy, iz, t))

    if (x1 >= length and x2 < length) or (x1 <= length and x2 > length):
        t = (x1 - length) / (x1 - x2)
        iy = y1 + t * (y2 - y1)
        iz = z1 + t * (z2 - z1)
        if 0 <= iy <= length and 0 <= iz <= length:
            intersections.append((length, iy, iz, t))
    
    return intersections



def evaluate_surface_connectivity(gt_swc, pred_swc, length=128, tolerance=1):
    gt_nodes = load_swc(gt_swc)
    pred_nodes = load_swc(pred_swc)
    
    if not gt_nodes or not pred_nodes:
        return 0, 0, 0 if not gt_nodes else len(gt_nodes)
    
    surface_nodes = {}
    for node_id, props in gt_nodes.items():
        x, y, z = props['x'], props['y'], props['z']
        if (abs(x) <= tolerance or abs(x - length) <= tolerance or
            abs(y) <= tolerance or abs(y - length) <= tolerance or
            abs(z) <= tolerance or abs(z - length) <= tolerance):
            surface_nodes[node_id] = props
    
    gt_connections = {}
    for node_id, props in gt_nodes.items():
        parent_id = props['parent']
        if parent_id != -1:
            if node_id not in gt_connections:
                gt_connections[node_id] = set()
            if parent_id not in gt_connections:
                gt_connections[parent_id] = set()
            gt_connections[node_id].add(parent_id)
            gt_connections[parent_id].add(node_id)
    
    gt_surface_groups = []
    visited = set()
    for start_node in surface_nodes:
        if start_node in visited:
            continue
        current_group = []
        queue = [start_node]
        local_visited = {start_node}
        while queue:
            node = queue.pop(0)
            current_group.append(node)
            visited.add(node)
            if node in gt_connections:
                for neighbor in gt_connections[node]:
                    if neighbor in surface_nodes and neighbor not in local_visited:
                        queue.append(neighbor)
                        local_visited.add(neighbor)
        if current_group:
            gt_surface_groups.append(current_group)
    
    pred_connections = {}
    for node_id, props in pred_nodes.items():
        parent_id = props['parent']
        if parent_id != -1:
            if node_id not in pred_connections:
                pred_connections[node_id] = set()
            if parent_id not in pred_connections:
                pred_connections[parent_id] = set()
            pred_connections[node_id].add(parent_id)
            pred_connections[parent_id].add(node_id)
    
    pred_coords = []
    pred_ids = []
    for node_id, props in pred_nodes.items():
        pred_coords.append([props['x'], props['y'], props['z']])
        pred_ids.append(node_id)
    
    if not pred_coords:
        return 0, len(gt_surface_groups), 0
    
    pred_kd_tree = KDTree(pred_coords)
    
    correct_count = 0
    error_count = 0
    cross_group_error_count = 0
    
    group_to_pred_points = []
    for group in gt_surface_groups:
        matched_pred_ids = []
        for node_id in group:
            node = gt_nodes[node_id]
            query_point = [node['x'], node['y'], node['z']]
            distance, index = pred_kd_tree.query(query_point)
            matched_pred_id = pred_ids[index]
            matched_pred_ids.append(matched_pred_id)
        
        group_to_pred_points.append(matched_pred_ids)
        
        if len(matched_pred_ids) <= 1:
            correct_count += 1
            continue
        
        start_node = matched_pred_ids[0]
        visited_pred = set()
        queue = [start_node]
        visited_pred.add(start_node)
        while queue:
            node = queue.pop(0)
            if node in pred_connections:
                for neighbor in pred_connections[node]:
                    if neighbor not in visited_pred:
                        queue.append(neighbor)
                        visited_pred.add(neighbor)
        
        all_connected = all(pred_id in visited_pred for pred_id in matched_pred_ids)
        if all_connected:
            correct_count += 1
        else:
            error_count += 1
    
    for i, group_pred_points in enumerate(group_to_pred_points):
        found = False
        for j, other_group_pred_points in enumerate(group_to_pred_points):
            if i == j:
                continue
            for pred_point in group_pred_points:
                for other_pred_point in other_group_pred_points:
                    if pred_point in pred_connections and other_pred_point in pred_connections[pred_point]:
                        found = True
                        cross_group_error_count += 1
                        break
                if found:
                    break
            if found:
                break
    
    return correct_count, error_count, cross_group_error_count

import glob
import os

def count_swc_part(filepath):
    data = {}
    count = 0
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                _, node_type, x, y, z, radius, _ = map(float, parts)
                node_id, _, _, _, _, _, parent = parts
                node_id = int(node_id)
                data[node_id] = {
                    'type': int(node_type),
                    'x': x,
                    'y': y,
                    'z': z,
                    'radius': radius,
                    'parent': int(parent)
                }
                if int(parent) == -1:
                    count += 1
    return count

  
if __name__ == "__main__":
    p_dir = "pridict_swc/"
    g_dir = "gt_swc/"

    output_file = "test_emb.txt"
    with open(output_file, "w") as f:
        f.write("filename Pred_to_GT GT_to_Pred\n")
    total_x1 = np.array([0, 0,0], dtype=np.float32)
    gt_files = sorted(glob.glob(os.path.join(g_dir, "block*.swc")))
    for i, gt_file in enumerate(gt_files):
        pred_file = os.path.join(p_dir, os.path.basename(gt_file))
        x1 = np.array(evaluate_surface_connectivity(gt_file, pred_file, length=128, tolerance=1))
        total_x1 += x1
        with open(output_file, "a") as f:
            f.write(f"{os.path.basename(gt_file)} {x1[0]} {x1[1]} {x1[2]}\n")
            
        if (i>100000):
            break
    print(p_dir)
    print(f"Pred to GT:  {total_x1[0]} {total_x1[1]} {total_x1[2]}")