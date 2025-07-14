from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist
from treelib import Tree
import numpy as np
from scipy.spatial import KDTree

class TreePlus:
    def __init__(self, swc_file, block_size, margin, node_margin):
        self.block_size = block_size
        self.margin = margin
        self.node_margin = node_margin

        self.tree = Tree()
        self.tree.create_node('0', 0)
        self.tree_coor = [np.ones(3) * -1]
        self.n = 1

        self.terminal_id = []

        self.from_swc(swc_file)

        self.dfn = []
        self.depth = []
        self.pos = [False] * self.n

        self.lg = None
        self.bit = int(np.ceil(np.log2(self.n)) + 1)
        self.st = np.zeros([self.bit, ((self.n << 1) + 3)])
        self.rev = np.zeros([self.bit, ((self.n << 1) + 3)])

        self.dfs(0, 0)
        self.init_st()

    def from_swc(self, swc_file):
        swc = open(swc_file, "r")

        line = swc.readline()
        while line:
            if line[0] == '#' or line[0] == '\n':
                line = swc.readline()
                continue
            line_data = line.split('\n')[0].split(' ')
            self.tree.create_node(line_data[1], int(line_data[0]), parent=max(0, int(line_data[-1])))
            self.tree_coor.append(np.array(line_data[2:5], dtype='float32'))
            line = swc.readline()
            self.n = self.n + 1
        self.tree_coor = np.array(self.tree_coor)

        subroot_idx = self.tree.is_branch(0)
        for n in self.tree.all_nodes_itr():
            if not n.is_root():
                if (n.identifier in subroot_idx) and (len(self.tree.is_branch(n.identifier)) == 1):
                    if ((min(self.tree_coor[n.identifier]) <= self.margin) or
                            ((self.block_size - max(self.tree_coor[n.identifier])) <= self.margin)):
                        term_set = [n.identifier]
                        valid = True
                        for i in range(self.node_margin - 1):
                            if not self.tree.is_branch(term_set[-1]):
                                valid = False
                                break
                            term_set.append(self.tree.is_branch(term_set[-1])[0])
                        if valid:
                            self.terminal_id.append(term_set)
                elif len(self.tree.is_branch(n.identifier)) == 0:
                    if ((min(self.tree_coor[n.identifier]) <= self.margin) or
                            ((self.block_size - max(self.tree_coor[n.identifier])) <= self.margin)):
                        term_set = [n.identifier]
                        valid = True
                        for i in range(self.node_margin - 1):
                            if self.tree.parent(term_set[-1]) is None:
                                valid = False
                                break
                            term_set.append(self.tree.parent(term_set[-1]).identifier)
                        if valid:
                            self.terminal_id.append(term_set)
        return

    def dfs(self, cur, dep):
        self.dfn.append(cur)
        self.depth.append(dep)
        self.pos[cur] = True
        child = self.tree.is_branch(cur)
        for i in range(len(child)):
            v = child[i]
            if not self.pos[v]:
                self.dfs(v, dep + 1)
                self.dfn.append(cur)
                self.depth.append(dep)
        return

    def init_st(self):
        self.lg = [0] * (len(self.dfn) + 1)
        for i in range(2, len(self.dfn) + 1):
            self.lg[i] = self.lg[i >> 1] + 1
        for i in range(1, len(self.dfn)):
            self.st[0][i] = self.depth[i]
            self.rev[0][i] = self.dfn[i]

        for i in range(1, self.lg[len(self.dfn) - 1] + 1):
            for j in range(1, len(self.dfn) - (1 << i) + 1):
                if self.st[i - 1][j] > self.st[i - 1][j + (1 << i - 1)]:
                    self.st[i][j] = self.st[i - 1][j + (1 << i - 1)]
                    self.rev[i][j] = self.rev[i - 1][j + (1 << i - 1)]
                else:
                    self.st[i][j] = self.st[i - 1][j]
                    self.rev[i][j] = self.rev[i - 1][j]
        return

    def query(self, r, l):
        if not r > l:
            t = r
            l = t
            r = l
        k = self.lg[r - l + 1]
        if self.st[k][l] < self.st[k][r + 1 - (1 << k)]:
            return self.rev[k][l]
        else:
            return self.rev[k][r + 1 - (1 << k)]

    def to_swc(self, path):
        with open(path, 'w') as swc_file:
            for n in self.tree.all_nodes_itr():
                if n.is_root():
                    continue
                p = self.tree.parent(n.identifier).identifier
                coor = self.tree_coor[n.identifier]
                if p == 0:
                    p = -1
                line = [str(n.identifier), '1', str(coor[0]), str(coor[1]), str(coor[2]),
                        '1.0', str(p)]
                print(' '.join(line) + '\n', file=swc_file)


class ConnAccuracy:
    def __init__(self, block_size, margin, node_margin):
        self.block_size = block_size
        self.margin = margin
        self.node_margin = node_margin

        self.input_reader = None
        self.target_reader = None

    def calculate(self, input, target):
        self.input_reader = TreePlus(input, self.block_size, self.margin, self.node_margin)
        self.target_reader = TreePlus(target, self.block_size, self.margin, self.node_margin)

        target_coor = self.target_reader.tree_coor[np.reshape(self.target_reader.terminal_id, [-1])]
        target_kdtree = KDTree(target_coor)
        ext_id = len(self.target_reader.terminal_id)
        match = []
        for term_set in self.input_reader.terminal_id:
            dis, idx = target_kdtree.query(self.input_reader.tree_coor[term_set])
            if np.min(dis) < self.margin:
                match.append(idx[np.argmin(dis)][0] // self.node_margin)
            else:
                match.append(ext_id)
                ext_id = ext_id + 1
        match = np.array(match)

        input_connection = self.find_connection(self.input_reader.terminal_id, self.input_reader)
        target_connection = self.find_connection(self.target_reader.terminal_id, self.target_reader)

        match_connection = []
        for c in input_connection:
            match_connection.append(tuple(set(match[list(c)].tolist())))
        match_connection = list(set(match_connection))
        print(match_connection)
        print(target_connection)
        tp, tn, fp, f1 = self.connection_diff(match_connection, target_connection)
        _tp, _tn, _fp, _f1 = self.count_diff(match_connection, target_connection)
        return [_tp, _tn, _fp, _f1]

    @staticmethod
    def find_connection(lst, reader):
        connection = []
        for i in range(len(lst)):
            new = True
            for j in range(len(connection)):
                if reader.query(lst[connection[j][0]][0], lst[i][0]):
                    connection[j].append(i)
                    new = False
                    break
            if new:
                connection.append([i])
        tupled_conn = []
        for c in connection:
            tupled_conn.append(tuple(c))
        return tupled_conn

    @staticmethod
    def connection_diff(input_conn, target_conn):
        tp = 0
        for t in target_conn:
            for i in input_conn:
                if set(t) == set(i):
                    tp = tp + 1
        if tp == 0:
            return [0, len(input_conn), len(target_conn), 0]

        tn = len(input_conn) - tp
        fp = len(target_conn) - tp
        r = tp / (tp + tn)
        p = tp / (tp + fp)
        f1 = 2 * r * p / (r + p)
        return [tp, tn, fp, f1]

    @staticmethod
    def count_diff(input_conn, target_conn):
        input_len = [len(c) for c in input_conn]
        target_len = [len(c) for c in target_conn]
        tp = 0
        for i in range(min(len(input_len), len(target_len))):
            if input_len[i] == target_len[i]:
                tp = tp + 1
        if tp == 0:
            return [0, len(input_conn), len(target_conn), 0]

        tn = len(input_conn) - tp
        fp = len(target_conn) - tp
        r = tp / (tp + tn)
        p = tp / (tp + fp)
        f1 = 2 * r * p / (r + p)
        return [tp, tn, fp, f1]



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

def _evaluate_template(gt_swc, pred_swc, length=128, tolerance=1):
    gt_nodes = load_swc(gt_swc)
    pred_nodes = load_swc(pred_swc)
    
    if not gt_nodes or not pred_nodes:
        return 0, 0 if not gt_nodes else len(gt_nodes)
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

    from scipy.spatial import KDTree
    pred_coords = []
    pred_ids = []
    for node_id, props in pred_nodes.items():
        pred_coords.append([props['x'], props['y'], props['z']])
        pred_ids.append(node_id)
    
    if not pred_coords:
        return 0, len(gt_surface_groups)
    
    pred_kd_tree = KDTree(pred_coords)
    correct_count = 0
    error_count = 0
    
    for group in gt_surface_groups:
        matched_pred_ids = []
        for node_id in group:
            node = gt_nodes[node_id]
            query_point = [node['x'], node['y'], node['z']]
            distance, index = pred_kd_tree.query(query_point)
            matched_pred_id = pred_ids[index]
            matched_pred_ids.append(matched_pred_id)

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
    
    return correct_count, error_count

def connectivity_score(gt_swc, pred_swc, length=128, tolerance=1):
    c, t1 = _evaluate_template(gt_swc, pred_swc, length, tolerance)
    c_, t2 = _evaluate_template(pred_swc, gt_swc, length, tolerance)
    t2*=2
    return c-t1-t2, t1, t2


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