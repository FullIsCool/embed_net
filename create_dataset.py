import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import os
import tifffile as tiff
from skimage.draw import line
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull, Delaunay

COUNTER_CONFLICT = 0

def parse_swc(file_path):
    nodes = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'): 
                continue
            parts = line.strip().split()
            if len(parts) == 7:
                n, T, x, y, z, R, P = map(float, parts)
                nodes[int(n)]={
                    'id': int(n),
                    'type': int(T),
                    'x': x,
                    'y': y,
                    'z': z,
                    'radius': R,
                    'parent': int(P)
                }
    return nodes

def add_f(nodes):
    for id in sorted(nodes.keys()):
        node = nodes[id]
        pid = node['parent']
        if pid == -1:
            node['f'] = id + 100
        if pid != -1:
            node['f'] = nodes[pid]['f'] + 1

def create_soma(nodes, shape=(300, 300, 300), scale=0.35):
    soma_id_list=[]
    for i, node in enumerate(nodes):
        if node['type'] == 1: 
            soma_id_list.append(node['id'])

    mask = np.zeros(shape, dtype=int)    
    for soma_id in soma_id_list:
        soma = nodes[soma_id]
        soma_center = (soma['x'], soma['y'], soma['z'])
        direction = None
        surface_points = []
        for i, node in enumerate(nodes):
            if node['parent'] == soma_id:
                direction = np.array([node['x'], node['y'], node['z']]) - np.array(soma_center)
                direction = direction / np.linalg.norm(direction)
            elif direction is not None:
                n = np.array([nodes[i + 1]['x'], nodes[i + 1]['y'], nodes[i + 1]['z']]) - \
                    np.array([node['x'], node['y'], node['z']])
                n = n / np.linalg.norm(n)
                if np.linalg.norm(n - direction) > 1e-2:
                    surface_points.append((node['x'] / scale, node['y'] / scale, node['z']))
                    direction = None
        
        for point in surface_points:
            x, y, z = map(int, point)
            mask[z, y, x] = soma_id
            
        sp = np.array(surface_points).astype(int)
        x_min, x_max = np.min(sp[:, 0]), np.max(sp[:, 0])
        y_min, y_max = np.min(sp[:, 1]), np.max(sp[:, 1])
        z_min, z_max = np.min(sp[:, 2]), np.max(sp[:, 2])

        if len(surface_points) > 3:  
            hull = ConvexHull(surface_points)
            delaunay = Delaunay(surface_points)
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    for z in range(z_min, z_max + 1):
                        point = np.array([x, y, z])
                        if delaunay.find_simplex(point) >= 0:
                            mask[z, y, x] = soma_id 

    return mask

def generate_prism(image, point1, point2, content=1, width=5, count=None):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    diff_id = 20
    if dz >= dx and dz >= dy:
        axis_range = range(min(z1, z2), max(z1, z2) + 1)
        for z in axis_range:
            t = (z - z1) / (z2 - z1) if z2 != z1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            x_min, x_max = max(0, x - width // 2), min(image.shape[0], x + width // 2 + 1)
            y_min, y_max = max(0, y - width // 2), min(image.shape[1], y + width // 2 + 1)
            diff = np.abs(image[z, y_min:y_max, x_min:x_max] - content) 
            coords = np.argwhere((diff > diff_id) & (image[z, y_min:y_max, x_min:x_max] != 0))
            temp = count[z, y_min:y_max, x_min:x_max][coords[:, 0], coords[:, 1]]
            
            image[z, y_min:y_max, x_min:x_max] = content
            count[z, y_min:y_max, x_min:x_max] = 1
            if temp.size !=0:
                count[z, y_min:y_max, x_min:x_max][coords[:, 0], coords[:, 1]] = (temp+1)

    elif dx >= dy:
        axis_range = range(min(x1, x2), max(x1, x2) + 1)
        for x in axis_range:
            t = (x - x1) / (x2 - x1) if x2 != x1 else 0
            y = int(y1 + t * (y2 - y1))
            z = int(z1 + t * (z2 - z1))
            y_min, y_max = max(0, y - width // 2), min(image.shape[1], y + width // 2 + 1)
            z_min, z_max = max(0, z - width // 2), min(image.shape[2], z + width // 2 + 1)
            
            diff = np.abs(image[z_min:z_max, y_min:y_max, x] - content) 
            coords = np.argwhere((diff > diff_id) & (image[z_min:z_max, y_min:y_max, x] != 0))
            temp = count[z_min:z_max, y_min:y_max, x][coords[:, 0], coords[:, 1]]
            image[z_min:z_max, y_min:y_max, x] = content
            count[z_min:z_max, y_min:y_max, x] = 1
            if temp.size !=0:
                count[z_min:z_max, y_min:y_max, x][coords[:, 0], coords[:, 1]] = (temp+1)

    else:
        axis_range = range(min(y1, y2), max(y1, y2) + 1)
        for y in axis_range:
            t = (y - y1) / (y2 - y1) if y2 != y1 else 0
            x = int(x1 + t * (x2 - x1))
            z = int(z1 + t * (z2 - z1))
            x_min, x_max = max(0, x - width // 2), min(image.shape[0], x + width // 2 + 1)
            z_min, z_max = max(0, z - width // 2), min(image.shape[2], z + width // 2 + 1)
            
            diff = np.abs(image[z_min:z_max, y, x_min:x_max] - content) 
            coords = np.argwhere((diff > diff_id) & (image[z_min:z_max, y, x_min:x_max] != 0))
            temp = count[z_min:z_max, y, x_min:x_max][coords[:, 0], coords[:, 1]]
            image[z_min:z_max, y, x_min:x_max] = content
            count[z_min:z_max, y, x_min:x_max] = 1
            if temp.size != 0:
                count[z_min:z_max, y, x_min:x_max][coords[:, 0], coords[:, 1]] = (temp+1)



def create_fiber(swc_data, w, shape=(300, 300, 300), scale=0.35):
    fiber = np.zeros(shape, dtype=int)
    count = np.zeros(shape, dtype=int)
    point1, point2 = None, None  
    for i, node in enumerate(swc_data.values()):
        x, y, z, parent, id, f = node['x'], node['y'], node['z'], node['parent'], node['id'], node['f']
        x, y, z = int(x / scale), int(y / scale), int(z)
        point1 = (x, y, z)
        if parent == -1:
            point2 = (x, y, z)
            continue
        parent_node = swc_data[int(parent)]
        parent_point = (int(parent_node['x'] / scale), int(parent_node['y'] / scale), int(parent_node['z']))
        
        # 画正方体
        x_min, x_max = max(0, int(x - w//2)), min(shape[0], int(x + w//2 + 1))
        y_min, y_max = max(0, int(y - w//2)), min(shape[1], int(y + w//2 + 1))
        z_min, z_max = max(0, int(z - w//2)), min(shape[2], int(z + w//2 + 1))
        cube = fiber[z_min:z_max, y_min:y_max, x_min:x_max]
        
        diff = np.abs(cube - f) 
        coords = np.argwhere((diff > 20) & (cube != 0))
        temp = count[z_min:z_max, y_min:y_max, x_min:x_max][coords[:, 0], coords[:, 1], coords[:, 2]]
        fiber[z_min:z_max, y_min:y_max, x_min:x_max] = f
        count[z_min:z_max, y_min:y_max, x_min:x_max] = 1
        if temp.size != 0:
            count[z_min:z_max, y_min:y_max, x_min:x_max][coords[:, 0], coords[:, 1], coords[:, 2]] = (temp+1)
            pass
        
        if point2 is not None:
            generate_prism(fiber, point1, parent_point, content=f, width=w, count=count)

        point2 = point1
    return fiber, count

def mark_branch(image, swc_data, scale=0.35):
    children_count = {node_id: 0 for node_id in swc_data.keys()}
    for node in swc_data.values():
        parent = node['parent']
        if parent != -1:
            children_count[parent] += 1
    for node in swc_data.values():
        x, y, z = node['x'], node['y'], node['z']
        x, y, z = round(x/scale), round(y/scale), round(z)
        if children_count[node['id']] > 1: 
            if x >= 0 and x < image.shape[2] and y >= 0 and y < image.shape[1] and z >= 0 and z < image.shape[0]:
                image[z, y, x] = -1


def make_mask(input_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/origin",
              output_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/mask_with_id"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    for filename in os.listdir(input_dir):

        if filename.endswith('.swc'):
            filepath = os.path.join(input_dir, filename)
            swc_data = parse_swc(filepath)
            add_f(swc_data)
            fiber = create_fiber(swc_data, w=3)
            soma = create_soma(swc_data)
            generated_image = np.maximum(fiber, soma)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_MaskID.npy")

            np.save(output_path, generated_image)
            print(f"Processed {filename} -> {output_path}")
            counter += 1


def split_128():
    input_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/mask_with_id"
    output_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/block_128/mask_with_id"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    index = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_dir, filename)
            image = np.load(file_path)
            print(filename)
            for x in range(5):
                for y in range(5):
                    for z in range(5):
                        block = image[40*z:40*z+128, 40*y:40*y+128, 40*x:40*x+128]
                        save_path = os.path.join(output_dir, f"block_idx_{index}.npy")
                        np.save(save_path, block)
                        index += 1


if __name__ == "__main__":
    #make_mask()
    #split_128()
    input_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/origin"
    output_dir1 = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/count_overlap"
    output_dir2 = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/mask_with_id"
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    flag =0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.swc'):
            print('-'*20, flush=True)
            filepath = os.path.join(input_dir, filename)
            swc_data = parse_swc(filepath)
            add_f(swc_data)
            fiber, count = create_fiber(swc_data, w=5)
            mark_branch(count, swc_data)
            #soma = create_soma(swc_data)
            #generated_image = np.maximum(fiber, soma)

            base_name = os.path.splitext(filename)[0]
            count_path = os.path.join(output_dir1, f"{base_name}_count.npy")
            id_path = os.path.join(output_dir2, f"{base_name}_ID.tif")
            print(np.max(count), np.min(count), flush=True)
            #visualize_count = (count/count.max() *65535).astype(np.uint16)
            fiber = fiber.astype(np.uint16)
            np.save(count_path, count)
            tiff.imwrite(id_path, fiber, imagej=True)
            print(f"Processed {filename} -> {count_path}", flush=True)
            flag+=1

