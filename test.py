import os
from  model import *
import tifffile as tiff
from loss import *
import numpy as np
import torch
from skimage.morphology import skeletonize_3d
from skimage.filters import threshold_otsu
from model import UNet3D 
import networkx as nx
import pandas as pd


class TIF3DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, length=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
        self.layer_num = 300 // 2
        if length is None:
            self.image_num = len(self.image_paths)
        else:
            self.image_num = length

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image = tiff.imread(self.image_paths[idx]).astype(np.float32)
        mask = tiff.imread(self.mask_paths[idx]).astype(np.float32)
        image = image / image.max() 
        mask = mask / 65535
        image = torch.tensor(image).unsqueeze(0)  
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


def visualize_featmap(feats, mask=None):
    C, X, Y, Z = feats.shape
    N = X * Y * Z
    data = feats.reshape(C, N)

    G = np.random.randn(C, 3)        
    Q, R = np.linalg.qr(G)       

    proj3 = Q.T.dot(data)             


    rgb = np.zeros_like(proj3)
    for i in range(3):
        v = proj3[i]
        v_min, v_max = v.min(), v.max()
        rgb[i] = (v - v_min) / (v_max - v_min + 1e-12)

    rgb_uint8 = (rgb * 255).astype(np.uint8)  
    rgb_uint8 = rgb_uint8.T.reshape(X, Y, Z, 3)
    
    if mask is not None:
        mask = mask.reshape(X, Y, Z, 1)
        mask = (mask > 0.5).astype(np.uint8)
        rgb_uint8 = rgb_uint8 * mask
        
    return rgb_uint8

    
@torch.no_grad()
def test_visualize_featmap():
    test_save_dir = r"./test_save"
    os.makedirs(test_save_dir, exist_ok=True)
    image_dir = r"./dataset_etv133/origin"
    mask_dir = r"./dataset_etv133/mask_with_id"
    count_dir = r"./dataset_etv133/count_overlap"
    model = UNet3D_embed(1, 4)
    sem_model = UNet3D(1, 1)
    model.load_state_dict(torch.load(r'./embed_net.pth',
                                     map_location=torch.device('cpu')))
    sem_model.load_state_dict(torch.load(r'./unet3d.pth',
                                         map_location=torch.device('cpu')))
    print("model loaded")
    dataset = EmbedNetDataset(image_dir, mask_dir, count_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                                 num_workers=4, pin_memory=True)
    i_list = [23, 519, 5017, 204]
    for i in i_list:
        image, label, count = dataset[i]
        image = image.unsqueeze(0)
        emb, sem = model(image)
        sem = sem_model(image)
        sem = sem.squeeze(0).detach().cpu().numpy()
        emb = emb.squeeze(0).detach().cpu().numpy()
        #sem = (sem > threshold_otsu(sem))
        sem = sem > 0.3
        visualize_emb = visualize_featmap(emb, sem)
        image = image.squeeze(0).detach().cpu().numpy()
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_image.tif"), (image*65535).astype(np.uint16), imagej=True)
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_label.tif"), label.squeeze(0).detach().cpu().numpy(), imagej=True)
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_visualized_prediction_.tif"), visualize_emb, imagej=True, photometric='rgb')
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_sem.tif"), (sem*65535).astype(np.uint16), imagej=True)
        print(f"save {i}") 


def disconnect(mask, emb, threshold):
    feat_dim, D, W, H  = emb.shape
    emb_pad = np.pad(emb, ((0,0), (1,1), (1,1), (1,1)), mode='constant')
    emb_neib = np.lib.stride_tricks.sliding_window_view(emb_pad, window_shape=(3,3,3), axis=(1,2,3)) 
    emb_neib = emb_neib.reshape(feat_dim, D, W, H, 27) 
    mask_pad = np.pad(mask, ((1,1), (1,1), (1,1)), mode='constant')
    mask_neib = np.lib.stride_tricks.sliding_window_view(mask_pad, window_shape=(3,3,3), axis=(0,1,2)) 
    mask_neib = mask_neib.reshape(D, W, H, 27) 
    center = emb.reshape(feat_dim, D, W, H, 1)
    distances = np.linalg.norm(emb_neib - center, axis=0) 
    distances = distances * mask_neib 
    max_distance = np.max(distances, axis=-1)  
    print(f"max_distance shape: {max_distance.shape}")
    new_mask = mask.copy()
    new_mask[max_distance > threshold] = 0
    return new_mask

def graph2swctrees(G): 
    SOMA_DEGREE = 4
    MIN_NODES = 10
    scale = 1
    components = list(nx.connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components if len(c) >= MIN_NODES]
    print(f"Found {len(subgraphs)} connected components in the graph.")
    trees = []
    for i, sg in enumerate(subgraphs):
        max_degree_node = max(sg.degree, key=lambda x: x[1])[0]
        max_degree = sg.degree[max_degree_node]
        if max_degree >= SOMA_DEGREE:
            node_type = 1
        else:
            node_type = 3
            
        T = nx.bfs_tree(sg, source=max_degree_node)
        index_map = {node: idx + 1 for idx, node in enumerate(T.nodes())}
        tree_nodes = []
        for node in T.nodes():
            z, y, x = node
            parent = -1
            if node != max_degree_node:
                parent_node = list(T.predecessors(node))[0]
                parent = index_map[parent_node]
            tree_nodes.append({
                "id": index_map[node],
                "type": 1 if node == max_degree_node else 3,
                "x": round(x * scale, 2),
                "y": round(y * scale, 2),
                "z": z,
                "radius": 1.0,
                "parent": parent
            })
        trees.append(tree_nodes)
    return trees



def save_trees_to_swc(trees, output_file):
    swc_data = []
    node_offset = 0
    
    for tree in trees:
        for node in tree:
            swc_data.append([
                node["id"] + node_offset,
                node["type"],
                node["x"],
                node["y"],
                node["z"],
                node["radius"],
                node["parent"] + node_offset if node["parent"] != -1 else -1
            ])
        node_offset += len(tree)
    
    swc_df = pd.DataFrame(swc_data, columns=["#id", "type", "x", "y", "z", "radius", "parent"])
    swc_df.to_csv(output_file, sep=" ", index=False, header=True)


@torch.no_grad()
def test_connect():
    test_save_dir = r"/home/fit/guozengc/WORK/fhy/codes/embed/test_save"
    os.makedirs(test_save_dir, exist_ok=True)
    image_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/origin"
    mask_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/mask_with_id"
    count_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/count_overlap"
    model = UNet3D_embed(1, 8)
    sem_model = UNet3D(1, 1)
    model.load_state_dict(torch.load(r'/home/fit/guozengc/WORK/fhy/codes/embed/model/v428_discrimitive_loss/embed_net.pth',
                                     map_location=torch.device('cpu')))
    sem_model.load_state_dict(torch.load(r'/home/fit/guozengc/WORK/fhy/codes/unet3d/model/unet3d_128.pth',
                                         map_location=torch.device('cpu')))
    print("model loaded")
    dataset = EmbedNetDataset(image_dir, mask_dir, count_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                                 num_workers=4, pin_memory=True)
    #i_list = list(range(200, 220)) + list(range(500, 520)) + list(range(5000, 5020))
    i_list = [23, 519, 5017,204]
    for i in i_list:
        image,label, count = dataset[i]
        if count.max() <= 1:
            continue
        image = image.unsqueeze(0)
        emb, sem = model(image)
        sem = sem_model(image)
        sem = sem.squeeze(0).detach().cpu().numpy()
        emb = emb.squeeze(0).detach().cpu().numpy()
        if sem.shape[0] == 1:
            sem = sem.squeeze(0)
        #sem = (sem > threshold_otsu(sem))
        sem = sem > 0.3
        # 断开
        EMB_DISCONNECT_THRESHOLD = 4.0
        sem = disconnect(sem, emb, threshold=EMB_DISCONNECT_THRESHOLD)
        skeleton = skeletonize_3d(sem)
        
        G = nx.Graph()
        region_points = np.argwhere(skeleton == 1)
        for point in region_points:
            z, y, x = point 
            G.add_node((z, y, x))
        for point in region_points:
            z, y, x = point
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue 
                        neighbor = (z + dz, y + dy, x + dx)
                        if neighbor in G.nodes:
                            G.add_edge((z, y, x), neighbor)

        MIN_NODES = 10
        EMB_CON_THRESHOLD = 2
        R = 16
        leafnode_map = np.zeros_like(skeleton)
        components = list(nx.connected_components(G))
        subgraphs = [G.subgraph(c).copy() for c in components if len(c) >= MIN_NODES]
        for sg in subgraphs:
            max_degree_node = max(sg.degree, key=lambda x: x[1])[0]
            tree = nx.bfs_tree(sg, source=max_degree_node)
            leaf_nodes = [node for node in tree.nodes() if 
                          (tree.out_degree(node) == 0) or (tree.in_degree(node) == 0)]
            for node in leaf_nodes:
                leafnode_map[node] = 1
        for z, y, x in np.argwhere(leafnode_map == 1):
            z_min, z_max = np.clip([z-R, z+R+1], 0, skeleton.shape[0])
            y_min, y_max = np.clip([y-R, y+R+1], 0, skeleton.shape[1])
            x_min, x_max = np.clip([x-R, x+R+1], 0, skeleton.shape[2])
            nei = leafnode_map[z_min:z_max, y_min:y_max, x_min:x_max]
            for nei_z, nei_y, nei_x in np.argwhere(nei == 1):
                if np.linalg.norm(emb[:, z, y, x] - emb[:, nei_z + z_min, nei_y + y_min, nei_x + x_min]) < EMB_CON_THRESHOLD:
                    G.add_edge((z, y, x), (nei_z + z_min, nei_y + y_min, nei_x + x_min))

        trees = graph2swctrees(G)
        output_file = os.path.join(test_save_dir, f"id_{i}_tree_522.swc")
        save_trees_to_swc(trees, output_file)
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_label.tif"), label.squeeze(0).detach().cpu().numpy(), imagej=True)
        count = count.detach().cpu().numpy()
        count = (count*65535/count.max()).astype(np.uint16)
        tiff.imwrite(os.path.join(test_save_dir, f"id_{i}_count_.tif"), count, imagej=True)
        print(f"save {i} to {output_file}")


if __name__ == "__main__":
    test_visualize_featmap()
    test_connect()