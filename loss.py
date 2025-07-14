import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time 

def group_consecutive_ids(instance_ids, interval=1):
    instance_ids = instance_ids.sort().values.tolist()  
    groups = []
    group = [instance_ids[0]] 

    for i in range(1, len(instance_ids)):
        if instance_ids[i] <= instance_ids[i - 1] + interval:
            group.append(instance_ids[i]) 
        else:
            groups.append(group)  
            group = [instance_ids[i]]

    groups.append(group)  
    return groups

def discriminative_loss(embeddings, labels, delta_v=0.5, delta_d=1.5,
                        alpha=1.0, beta=1.0, gamma=0.001, device=None):
    N, D, _, _, _ = embeddings.shape
    total_loss = embeddings.new_tensor(0.0)
    total_L_each = embeddings.new_tensor([0.0, 0.0, 0.0])
    if device is None:
        device = labels.device
        
    for b in range(N):
        emb = embeddings[b].reshape(D, -1).transpose(0,1)  
        lbl = labels[b].reshape(-1)                      
        all_ids = lbl.unique()
        all_ids = all_ids[all_ids != 0]  
        
        if len(all_ids)==0:
            continue
        
        id_groups = group_consecutive_ids(all_ids)
        mu_list = []
        for id_group in id_groups:
            mask = torch.isin(lbl, torch.tensor(id_group, device=device))
            if mask.sum()==0: 
                mu_list.append(torch.zeros(D, device=device))
                raise ValueError(f"mask.sum()=0")
            else:
                mu = emb[mask].mean(dim=0)
                mu_list.append(mu)
        mus = torch.stack(mu_list, dim=0)  
        
        L_var = emb.new_tensor(0.0)
        for k, id_group in enumerate(id_groups):
            mask = torch.isin(lbl, torch.tensor(id_group, device=device))
            if mask.sum() > 0:
                dist = (emb[mask] - mus[k].unsqueeze(0)).norm(p=2, dim=1)
                L_var += torch.mean(F.relu(dist - delta_v)**2)
        L_var = L_var / len(id_groups)
        
        K = len(id_groups)
        if K > 1:
            mu_a = mus.unsqueeze(0).expand(K, K, D)
            mu_b = mus.unsqueeze(1).expand(K, K, D)
            diff = mu_a - mu_b  
            dist_mat = torch.norm(diff, dim=2) 
            margin = F.relu(2*delta_d - dist_mat)
            diag_mask = torch.ones_like(dist_mat)
            diag_mask.fill_diagonal_(0)
            margin = margin * diag_mask
            L_dist = torch.sum(margin**2) / (K*(K-1))
        else:
            L_dist = emb.new_tensor(0.0)
        
        L_reg = torch.mean(mus.norm(p=2, dim=1))
        total_loss += alpha * L_var + beta * L_dist + gamma * L_reg
        total_L_each += torch.tensor([L_var, L_dist, L_reg], device=device)

    return total_loss / N, total_L_each / N


def continuous_loss(embeds, labels, count, patch_size=10, delta_v=0.5, delta_d=1.5, id_interval=1, device=None):
    if device is None:
        device = embeds.device
    B, feat_dim, D, H, W = embeds.shape
    labels_pad = F.pad(labels, (1, 1, 1, 1, 1, 1), mode='constant', value=0) 
    embeds_pad = F.pad(embeds, (1, 1, 1, 1, 1, 1), mode='constant', value=0)  
    label_n = torch.zeros((B, 27, D, H, W), dtype=int, device=device)
    embed_n = torch.zeros((B, 27, feat_dim, D, H, W), device=device)
    for i in range(3):
        for j in range(3):
                for k in range(3):
                    index = i * 9 + j * 3 + k
                    label_n[:, index] = labels_pad[:, i:D+i, j:H+j, k:W+k]
                    embed_n[:, index] = embeds_pad[:, :, i:D+i, j:H+j, k:W+k]
    mask1 = (torch.abs(label_n - labels.unsqueeze(1)) < 2) & (label_n != 0) & (labels.unsqueeze(1) != 0)
    norm = torch.norm(embed_n - embeds.unsqueeze(1), p=2, dim=2)  
    loss_var = torch.mean(F.relu(norm[mask1]-delta_v) ** 2)
    
    mask2 = (torch.abs(label_n - labels.unsqueeze(1)) > 1) & (label_n != 0) & (labels.unsqueeze(1) != 0)
    
    half = patch_size // 2
    mask_count = count>1
    nonzero_indices = mask_count.nonzero(as_tuple=False)
    nonzero_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))]
    
    loss_dist = []
    num = 0
    for idx in nonzero_indices:
        b, d, h, w = idx.tolist()
        if mask_count[b, d, h, w].item() == 0:
            continue

        D_dim, H_dim, W_dim = mask_count.shape[1], mask_count.shape[2], mask_count.shape[3]

        d0, d1 = max(d - half, 0), min(d + half, D_dim)
        h0, h1 = max(h - half, 0), min(h + half, H_dim)
        w0, w1 = max(w - half, 0), min(w + half, W_dim)
        patch_embed = embeds[b, :, d0:d1, h0:h1, w0:w1]
        patch_label = labels[b, d0:d1, h0:h1, w0:w1]
        all_ids = patch_label.unique()
        all_ids = all_ids[all_ids != 0]     
        if len(all_ids)==0:
            continue
        id_groups = group_consecutive_ids(all_ids, id_interval)
        mu_list = []
        for id_group in id_groups:
            mask = torch.isin(patch_label, torch.tensor(id_group, device=device))
            if mask.sum()==0: 
                mu_list.append(torch.zeros(feat_dim, device=device))
                raise ValueError(f"mask.sum()=0")
            else:
                mu = patch_embed[:,mask].mean(dim=1)  
                mu_list.append(mu)
        mus = torch.stack(mu_list, dim=0)  
        K = len(id_groups)
        if K > 1:
            mu_a = mus.unsqueeze(0).expand(K, K, feat_dim)
            mu_b = mus.unsqueeze(1).expand(K, K, feat_dim)
            diff = mu_a - mu_b  
            dist_mat = torch.norm(diff, dim=2)  
            margin = F.relu(2*delta_d - dist_mat)
            diag_mask = torch.ones_like(dist_mat)
            diag_mask.fill_diagonal_(0)
            margin = margin * diag_mask
            L_dist = torch.sum(margin**2) / (K*(K-1))
            loss_dist.append(L_dist)

    loss_dist = torch.stack(loss_dist).mean() if loss_dist else torch.tensor(0.0, device=device)
    loss_reg = torch.mean(embeds.norm(p=2, dim=1))  
    return loss_var, loss_dist, loss_reg


if __name__ == '__main__':
    labels = torch.zeros((2, 2, 5, 5), dtype=torch.int)
    counts = torch.zeros((2, 2, 5, 5), dtype=torch.int)
    counts[:, 0, 3, 3] = 2
    labels[:, 0, 1:4, 1:4] = 1
    labels[:, 0, 3:4, 3:4] = 4
    embed = torch.zeros((2, 3, 2, 5, 5))
    embed[:, :, 0, 1:4, 1:4] = 1
    embed[:, :, 0, 3:4, 3:4] = 3
    print(continuous_loss(embed, labels, counts))
    