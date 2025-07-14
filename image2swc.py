import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_neighbor_offsets_3d():
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                offsets.append((di, dj, dk))
    return np.array(offsets)

NEIGHBOR_OFFSETS = get_neighbor_offsets_3d()

def extract_patch(mask, vec, x, y, z):
    pad_width = 1
    mask_p = np.pad(mask, pad_width, mode='constant', constant_values=False)
    vec_p = np.pad(vec, ((pad_width, pad_width),
                         (pad_width, pad_width),
                         (pad_width, pad_width),
                         (0, 0)), mode='edge')
    x_p, y_p, z_p = x + pad_width, y + pad_width, z + pad_width
    patch_mask = mask_p[x_p - 1: x_p + 2, y_p - 1: y_p + 2, z_p - 1: z_p + 2]
    patch_vec = vec_p[x_p - 1: x_p + 2, y_p - 1: y_p + 2, z_p - 1: z_p + 2, :]
    return patch_mask.copy(), patch_vec.copy()

def count_components_in_patch(patch_mask, patch_vec, thresh):
    shape = patch_mask.shape
    coords = np.indices(shape).reshape(3, -1).T  
    valid = patch_mask.flatten()  
    if np.sum(valid) == 0:
        return 0
    n_nodes = coords.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    comp_count = 0
    patch_vec_flat = patch_vec.reshape(-1, patch_vec.shape[-1])
    coords_dict = {tuple(coords[i]): i for i in range(n_nodes)}
    adj_list = [[] for _ in range(n_nodes)]
    for idx in range(n_nodes):
        if not valid[idx]:
            continue
        ci, cj, ck = coords[idx]
        for offset in NEIGHBOR_OFFSETS:
            ni, nj, nk = ci + offset[0], cj + offset[1], ck + offset[2]
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                neighbor_idx = coords_dict[(ni, nj, nk)]
                if not valid[neighbor_idx]:
                    continue
                diff = patch_vec_flat[idx] - patch_vec_flat[neighbor_idx]
                if np.linalg.norm(diff) < thresh:
                    adj_list[idx].append(neighbor_idx)
                    
    for idx in range(n_nodes):
        if valid[idx] and not visited[idx]:
            comp_count += 1
            queue = [idx]
            visited[idx] = True
            while queue:
                current = queue.pop(0)
                for nbr in adj_list[current]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(nbr)
    return comp_count

def is_simple_point_tensor(x, y, z, mask, vec, thresh):
    patch_mask, patch_vec = extract_patch(mask, vec, x, y, z)
    comp_before = count_components_in_patch(patch_mask, patch_vec, thresh)
    patch_mask_removed = patch_mask.copy()
    patch_mask_removed[1, 1, 1] = False
    comp_after = count_components_in_patch(patch_mask_removed, patch_vec, thresh)
    return comp_before == comp_after

def compute_border_candidates(mask):
    structure = np.ones((3, 3, 3), dtype=int)
    conv_res = convolve(mask.astype(int), structure, mode='constant', cval=0)
    border_candidates = mask & (conv_res < structure.sum())
    return border_candidates

def vector_skeletonize_3d_tensor(image_vector, image_mask, thresh, max_iter=100):
    mask = image_mask.copy()
    vec = image_vector 
    iteration = 0

    while iteration < max_iter:
        iteration += 1
        border_candidates = compute_border_candidates(mask)
        coords = np.argwhere(border_candidates)
        to_remove = []
        for x, y, z in coords:
            if mask[x, y, z] and is_simple_point_tensor(x, y, z, mask, vec, thresh):
                to_remove.append((x, y, z))
        if not to_remove:
            print(f"Iteration {iteration}: no points removed")
            break
        else:
            for x, y, z in to_remove:
                mask[x, y, z] = False
            print(f"Iteration {iteration}: removed {len(to_remove)} points")
    return mask

def visualize_3d_mask(mask, title="3D Mask Visualization"):

    coords = np.argwhere(mask)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue', marker='o', s=5, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    X, Y, Z, D = 20, 20, 20, 3
    image_vector = np.random.rand(X, Y, Z, D)
    image_mask = np.ones((X, Y, Z), dtype=bool)
    for i in range(min(X, Y, Z)):
        image_vector[i, i, i] = np.array([0.5, 0.5, 0.5]) + 0.01 * i
        
    thresh = 0.05
    skel_mask = vector_skeletonize_3d_tensor(image_vector, image_mask, thresh, max_iter=50)
    print("nodes", np.sum(skel_mask))
    visualize_3d_mask(image_mask, title="Initial Image Mask")
    visualize_3d_mask(skel_mask, title="Skeletonized Mask")