import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, DistributedSampler
from tensorboardX import SummaryWriter
from model import *
from loss import *

def compute_loss_sample(embed, segment, label, count, args):
    mask = (label > 0).float()
    bce_criterion = nn.BCEWithLogitsLoss()
    loss_bce = bce_criterion(segment, mask)
    
    patch_size = args.patch_size
    half = patch_size // 2
    patch_loss_total = 0.0
    patch_loss_total_each = torch.tensor([0., 0, 0], device=embed.device)
    patch_num = 0
    mask_count = count>1
    nonzero_indices = mask_count.nonzero(as_tuple=False)
    nonzero_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))]
    
    num = 0
    for idx in nonzero_indices:
        b, d, h, w = idx.tolist()
        if mask_count[b, d, h, w].item() == 0:
            continue
        
        num+=1
        D_dim, H_dim, W_dim = mask_count.shape[1], mask_count.shape[2], mask_count.shape[3]

        d0, d1 = max(d - half, 0), min(d + half, D_dim)
        h0, h1 = max(h - half, 0), min(h + half, H_dim)
        w0, w1 = max(w - half, 0), min(w + half, W_dim)
        patch_emb = embed[b, :, d0:d1, h0:h1, w0:w1].unsqueeze(0)  
        patch_label = label[b, d0:d1, h0:h1, w0:w1].unsqueeze(0) 
        patch_loss, patch_loss_each = discriminative_loss(patch_emb, patch_label, delta_v=args.delta_v, delta_d=args.delta_d, 
                                         alpha=args.var_weight, beta=args.dist_weight, gamma=args.reg_weight)
        patch_loss_total += patch_loss
        patch_loss_total_each += patch_loss_each
        patch_num += 1
        mask_count[b, d0:d1, h0:h1, w0:w1] = 0
        
    loss_dsc = patch_loss_total / patch_num if patch_num > 0 else torch.tensor(0.0,device=embed.device)
    loss_dsc_detect = {"overlap": patch_loss_total_each/patch_num if patch_num > 0 else torch.tensor([0.0, 0, 0],device=embed.device)}
    branch_count = (count ==-1)
    nonzero_indices = branch_count.nonzero(as_tuple=False)
    nonzero_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))]
    patch_loss_total = 0.0
    patch_loss_total_each = torch.tensor([0., 0, 0], device=embed.device)
    patch_num = 0
    for limit, idx in enumerate(nonzero_indices):
        if limit > num + 10:
            break
        b, d, h, w = idx.tolist()
        if branch_count[b, d, h, w].item() == 0:
            continue

        D_dim, H_dim, W_dim = branch_count.shape[1], branch_count.shape[2], branch_count.shape[3]

        d0, d1 = max(d - half, 0), min(d + half, D_dim)
        h0, h1 = max(h - half, 0), min(h + half, H_dim)
        w0, w1 = max(w - half, 0), min(w + half, W_dim)
        patch_emb = embed[b, :, d0:d1, h0:h1, w0:w1].unsqueeze(0)
        patch_label = label[b, d0:d1, h0:h1, w0:w1].unsqueeze(0)
        patch_loss, patch_loss_each = discriminative_loss(patch_emb, patch_label, delta_v=args.delta_v, delta_d=args.delta_d, 
                                         alpha=args.var_weight, beta=args.dist_weight, gamma=args.reg_weight)
        patch_loss_total += patch_loss
        patch_loss_total_each += patch_loss_each
        patch_num += 1
    
    branch_loss = patch_loss_total / patch_num if patch_num > 0 else torch.tensor(0.0,device=embed.device)
    loss_dsc += branch_loss
    loss_dsc_detect["branch"] = patch_loss_total_each/patch_num if patch_num > 0 else torch.tensor([0.0, 0, 0],device=embed.device)
    straight_count = (count == 1)
    nonzero_indices = straight_count.nonzero(as_tuple=False)
    nonzero_indices = nonzero_indices[torch.randperm(nonzero_indices.size(0))]
    patch_loss_total = 0.0
    patch_loss_total_each = torch.tensor([0., 0, 0], device=embed.device)
    patch_num = 0
    for limit, idx in enumerate(nonzero_indices):
        if limit > num + 10:
            break
        b, d, h, w = idx.tolist()
        if straight_count[b, d, h, w].item() == 0:
            continue

        D_dim, H_dim, W_dim = straight_count.shape[1], straight_count.shape[2], straight_count.shape[3]

        d0, d1 = max(d - half, 0), min(d + half, D_dim)
        h0, h1 = max(h - half, 0), min(h + half, H_dim)
        w0, w1 = max(w - half, 0), min(w + half, W_dim)
        patch_emb = embed[b, :, d0:d1, h0:h1, w0:w1].unsqueeze(0)
        patch_label = label[b, d0:d1, h0:h1, w0:w1].unsqueeze(0)
        patch_loss, patch_loss_each = discriminative_loss(patch_emb, patch_label, delta_v=args.delta_v, delta_d=args.delta_d, 
                                         alpha=args.var_weight, beta=args.dist_weight, gamma=args.reg_weight)
        patch_loss_total += patch_loss
        patch_loss_total_each += patch_loss_each
        patch_num += 1
        straight_count[b, d0:d1, h0:h1, w0:w1] = 0
    
    loss_dsc += (patch_loss_total / patch_num if patch_num > 0 else torch.tensor(0.0,device=embed.device) )
    loss_dsc_detect["straight"] = patch_loss_total_each/patch_num if patch_num > 0 else torch.tensor([0.0, 0, 0],device=embed.device)
    total_loss = loss_dsc + args.bce_weight * loss_bce
    return total_loss, loss_dsc, loss_bce, loss_dsc_detect

def compute_loss_grid(embed, segment, label, count, args):
    mask = (label > 0).float()

    bce_criterion = nn.BCEWithLogitsLoss()
    loss_bce = bce_criterion(segment, mask)
    
    patch_size = args.patch_size

    patch_loss_total = 0.0
    patch_loss_total_each = torch.tensor([0., 0, 0], device=embed.device)
    patch_num = 0
    D_dim, H_dim, W_dim = count.shape[1], count.shape[2], count.shape[3]
    stride = patch_size // 2
    for i in range((D_dim-patch_size)//stride + 1):
        for j in range((H_dim-patch_size)//patch_size + 1):
            for k in range((W_dim-patch_size)//patch_size + 1):
                d0, d1 = i * stride, i * stride + patch_size
                h0, h1 = j * stride, j * stride + patch_size
                w0, w1 = k * stride, j * stride + patch_size
                patch_emb = embed[:, :, d0:d1, h0:h1, w0:w1]
                patch_label = label[:, d0:d1, h0:h1, w0:w1]
                patch_loss, patch_loss_each = discriminative_loss(patch_emb, patch_label, delta_v=args.delta_v, delta_d=args.delta_d, 
                                         alpha=args.var_weight, beta=args.dist_weight, gamma=args.reg_weight)
                patch_loss_total += patch_loss
                patch_loss_total_each += patch_loss_each
                patch_num += 1
    loss_dsc = patch_loss_total / patch_num if patch_num > 0 else torch.tensor(0.0,device=embed.device)
    loss_dsc_detect = {"grid": patch_loss_total_each / patch_num if patch_num > 0 else torch.tensor([0.0, 0, 0],device=embed.device)}
    total_loss = loss_dsc + args.bce_weight * loss_bce
    return total_loss, loss_dsc, loss_bce, loss_dsc_detect
    
def compute_loss_continuous(embeds, segment, labels, counts, args):
    loss_var, loss_dist, loss_reg = continuous_loss(embeds, labels, counts, 
                                                    args.patch_size, args.delta_v, args.delta_d, args.id_interval)
    masks = (labels > 0).float() 
    loss_bce = F.binary_cross_entropy_with_logits(segment, masks, reduction='mean')
    total_loss = args.var_weight*loss_var + args.dist_weight*loss_dist + args.reg_weight*loss_reg + args.bce_weight*loss_bce
    loss_detect = {'var': loss_var, 'dist': loss_dist, 'reg': loss_reg, 'bce': loss_bce}
    return total_loss, loss_detect
    
def train_epoch(model, loss_func, dataloader, optimizer, device, args):
    model.train()
    epoch_loss, epoch_loss_dsc, epoch_loss_bce = 0.0, 0.0, 0.0
    epoch_loss_dsc_detect = {'overlap': torch.zeros(3, device=device),
                             'branch': torch.zeros(3, device=device),
                             'straight': torch.zeros(3, device=device)}
    epoch_loss_dsc_detect = None
    total_samples = 0

    for batch in dataloader:
        patches = batch[0].to(device)         
        targets = batch[1].to(device)         
        counts = batch[2].to(device)         
        optimizer.zero_grad()
        embed, segment = model(patches) 
        loss, loss_dsc, loss_bce, loss_dsc_detect = loss_func(embed, segment, targets, counts, args)
        loss.backward()
        optimizer.step()

        batch_size = patches.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_loss_dsc += loss_dsc.item() * batch_size
        epoch_loss_bce += loss_bce.item() * batch_size
        
        for key in loss_dsc_detect.keys():
            if epoch_loss_dsc_detect is None:
                epoch_loss_dsc_detect = {key: torch.zeros(3, device=device) for key in loss_dsc_detect.keys()}
            epoch_loss_dsc_detect[key] += loss_dsc_detect[key] * batch_size
        total_samples += batch_size

    avg_loss = epoch_loss / total_samples
    avg_loss_tp = epoch_loss_dsc / total_samples
    avg_loss_bce = epoch_loss_bce / total_samples
    avg_loss_dsc_detect = {key: value / total_samples for key, value in epoch_loss_dsc_detect.items()}
    return avg_loss, avg_loss_tp, avg_loss_bce, avg_loss_dsc_detect


@torch.no_grad()
def valid_epoch(model, loss_func, dataloader, device, args):
    model.eval()
    val_loss, val_loss_dsc, val_loss_bce = 0.0, 0.0, 0.0
    val_loss_dsc_detect = {'overlap': torch.zeros(3, device=device),
                            'branch': torch.zeros(3, device=device),
                            'straight': torch.zeros(3, device=device)}
    val_loss_dsc_detect = None
    total_samples = 0

    for batch in dataloader:
        patches = batch[0].to(device)        
        targets = batch[1].to(device)          
        counts = batch[2].to(device)         
        embed, segment = model(patches)
        loss, loss_dsc, loss_bce, loss_dsc_detect = loss_func(embed, segment, targets, counts, args)
        batch_size = patches.size(0)
        val_loss += loss.item() * batch_size
        val_loss_dsc += loss_dsc.item() * batch_size
        val_loss_bce += loss_bce.item() * batch_size
        for key in loss_dsc_detect.keys():
            if val_loss_dsc_detect is None:
                val_loss_dsc_detect = {key: torch.zeros(3, device=device) for key in loss_dsc_detect.keys()}
            val_loss_dsc_detect[key] += loss_dsc_detect[key] * batch_size
        total_samples += batch_size

    avg_loss = val_loss / total_samples
    avg_loss_tp = val_loss_dsc / total_samples
    avg_loss_bce = val_loss_bce / total_samples
    avg_loss_dsc_detect = {key: value / total_samples for key, value in val_loss_dsc_detect.items()}
    return avg_loss, avg_loss_tp, avg_loss_bce, avg_loss_dsc_detect


def adjust_learning_rate(optimizer, current_loss, previous_losses):
    if previous_losses and current_loss > max(previous_losses):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9995


def main(args):
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    logger = SummaryWriter(logdir=args.train_dir)

    dataset = EmbedNetDataset(image_dir=args.image_dir,
                                  mask_dir=args.mask_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = UNet3D_embed(in_channels=1, out_channels=8)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_epoch = -1
    previous_losses = [1e18] * 3

    if args.is_train:
        for epoch in range(1, args.num_epochs + 1):
            start_time = time.time()
            train_loss, train_loss_tp, train_loss_bce = train_epoch(model, train_loader, optimizer, device, args.bce_weight)
            val_loss, val_loss_tp, val_loss_bce = valid_epoch(model, val_loader, device, args.bce_weight)
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch}/{args.num_epochs}")
            print(f"  Training   Loss: {train_loss:.4f}, TP Loss: {train_loss_tp:.4f}, BCE Loss: {train_loss_bce:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}, TP Loss: {val_loss_tp:.4f}, BCE Loss: {val_loss_bce:.4f}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")

            logger.add_scalar("training/loss", train_loss, epoch)
            logger.add_scalar("training/loss_tp", train_loss_tp, epoch)
            logger.add_scalar("training/loss_bce", train_loss_bce, epoch)
            logger.add_scalar("validation/loss", val_loss, epoch)
            logger.add_scalar("validation/loss_tp", val_loss_tp, epoch)
            logger.add_scalar("validation/loss_bce", val_loss_bce, epoch)
            logger.add_scalar("epoch_time", epoch_time, epoch)
            logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

            for name, param in model.named_parameters():
                logger.add_histogram(name, param.data.cpu().numpy(), epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                ckpt_path = os.path.join(args.train_dir, f"embed_net.pth")
                print(f"Saving checkpoint at epoch {epoch} to {ckpt_path}")
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)

            adjust_learning_rate(optimizer, train_loss, previous_losses)
            previous_losses = previous_losses[1:] + [train_loss]

        print(f"Training complete. Best validation loss {best_val_loss:.4f} at epoch {best_epoch}.")
        logger.close()
        
    else:
        ckpt_path = os.path.join(args.train_dir, f"checkpoint_{args.checkpoint_ver}.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded model from {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} does not exist. Exiting...")
            return

        test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        test_loss= valid_epoch(model, test_loader, device, args.bce_weight)
        print(f"Test Loss: {test_loss:.4f}")



def main_distributed(args):
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training")

    is_main_process = True
    if dist.is_initialized():
        is_main_process = (dist.get_rank() == 0)

    if is_main_process:
        print("Arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print('-'*20)
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        logger = SummaryWriter(logdir=args.train_dir)
    else:
        logger = None
    
    dataset = EmbedNetDataset(image_dir=args.image_dir, 
                              label_dir=args.label_dir,
                              count_dir=args.count_dir,
                              random=args.random_sample_dataset,
                              num_blocks_per_file=args.num_blocks_per_file,
                              length=args.patch_length,
                              transform=args.transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, sampler=val_sampler)

    model = UNet3D_embed(in_channels=1, embed_dim=args.embed_dim)
    model = model.to(device)
    
    loss_func = compute_loss_sample
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.from_checkpoint:
        ckpt_path = os.path.join(args.train_dir, f"embed_net.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded model from {ckpt_path}")
            
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process:
            print(f"Using DistributedDataParallel with {dist.get_world_size()} GPUs", flush=True)
            print("Model name:", model.module.__class__.__name__)
            print("Loss name:", loss_func.__name__)
    else:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs", flush=True)
            print("Model name:", model.module.__class__.__name__)
            print("Loss name:", loss_func.__name__)
        else:
            print("Using a single GPU", flush=True)
            print("Model name:", model.__class__.__name__)
            print("Loss name:", loss_func.__name__)

    best_val_loss = float('inf')
    best_epoch = -1
    previous_losses = [1e18] * 3

    if args.is_train:
        for epoch in range(1, args.num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, train_loss_tp, train_loss_bce, train_loss_dsc_detect = train_epoch(model, loss_func, train_loader, optimizer, device, args)
            val_loss,   val_loss_tp,   val_loss_bce, val_loss_dsc_detect = valid_epoch(model, loss_func, val_loader, device, args)
            for key, _ in train_loss_dsc_detect.items():
                train_loss_dsc_detect[key] = torch.round(train_loss_dsc_detect[key] * 100) / 100
                val_loss_dsc_detect[key] = torch.round(val_loss_dsc_detect[key] * 100) / 100
            
            epoch_time = time.time() - start_time

            if dist.is_initialized():
                train_loss_tensor = torch.tensor(train_loss, device=device)
                val_loss_tensor = torch.tensor(val_loss, device=device)
                train_loss_tp_tensor = torch.tensor(train_loss_tp, device=device)
                train_loss_bce_tensor = torch.tensor(train_loss_bce, device=device)
                val_loss_tp_tensor = torch.tensor(val_loss_tp, device=device)
                val_loss_bce_tensor = torch.tensor(val_loss_bce, device=device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_loss_tp_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_tp_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_loss_bce_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_loss_bce_tensor, op=dist.ReduceOp.SUM)
                train_loss = train_loss_tensor.item() / dist.get_world_size()
                val_loss = val_loss_tensor.item() / dist.get_world_size()
                train_loss_tp = train_loss_tp_tensor.item() / dist.get_world_size()
                val_loss_tp = val_loss_tp_tensor.item() / dist.get_world_size()
                train_loss_bce = train_loss_bce_tensor.item() / dist.get_world_size()
                val_loss_bce = val_loss_bce_tensor.item() / dist.get_world_size()
            if is_main_process:
                print(f"Epoch {epoch}/{args.num_epochs}", flush=True)
                print(f"  Training   Total Loss: {train_loss:.4f} = TP({train_loss_tp:.4f}) + {args.bce_weight:.1f} * BCE({train_loss-train_loss_tp:.4f})", flush=True)
                print(f"    each part of training loss: {train_loss_dsc_detect}", flush=True)
                print(f"  Validation Total Loss: {  val_loss:.4f} = TP({  val_loss_tp:.4f}) + {args.bce_weight:.1f} * BCE({    val_loss-val_loss_tp:.4f})", flush=True)
                print(f"    each part of validation loss: {val_loss_dsc_detect}", flush=True)
                print(f"  Epoch time: {epoch_time:.2f}s", flush=True)
                print(f"  Learning rate: {optimizer.param_groups[0]['lr']}", flush=True)


                logger.add_scalar("training/loss", train_loss, epoch)
                logger.add_scalar("validation/loss", val_loss, epoch)
                logger.add_scalar("training/loss_tp", train_loss_tp, epoch)
                logger.add_scalar("validation/loss_tp", val_loss_tp, epoch)
                logger.add_scalar("training/loss_bce", train_loss_bce, epoch)
                logger.add_scalar("validation/loss_bce", val_loss_bce, epoch)
                logger.add_scalar("epoch_time", epoch_time, epoch)
                logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

                for name, param in model.named_parameters():
                    logger.add_histogram(name, param.data.cpu().numpy(), epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    ckpt_path = os.path.join(args.train_dir, f"embed_net.pth")
                    print(f"Saving checkpoint at epoch {epoch} to {ckpt_path}", flush=True)
                    if args.save_model:
                        torch.save(model.module.state_dict(), ckpt_path)

            adjust_learning_rate(optimizer, train_loss, previous_losses)
            previous_losses = previous_losses[1:] + [train_loss]

        if is_main_process:
            print(f"Training complete. Best validation loss {best_val_loss:.4f} at epoch {best_epoch}.")
            logger.close()

    else:
        ckpt_path = os.path.join(args.train_dir, f"checkpoint_{args.checkpoint_ver}.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded model from {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} does not exist. Exiting...")
            return

        test_dataset = val_dataset
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, sampler=test_sampler)
        test_loss = valid_epoch(model, test_loader, device, args)

        if dist.is_initialized():
            test_loss_tensor = torch.tensor(test_loss, device=device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            test_loss = test_loss_tensor.item()

        print(f"Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Test EmbedNetWithProb Model")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--from_checkpoint", type=bool, default=False)
    #---------- important -----------
    parser.add_argument("--train_dir", type=str)
    # -------------------------------
    parser.add_argument("--image_dir", type=str, default="~/dataset_etv133/origin")
    parser.add_argument("--label_dir", type=str, default="~/dataset_etv133/mask_with_id")
    parser.add_argument("--count_dir", type=str, default="~/dataset_etv133/count_overlap_w5")

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--bce_weight", type=float, default=0.01, help="weight of bce term")
    parser.add_argument("--embed_dim", type=int, default=4)
    parser.add_argument("--delta_v", type=float, default=1.0)
    parser.add_argument("--delta_d", type=float, default=4.0)
    parser.add_argument("--var_weight", type=float, default=1.0, help="weight of variance term")
    parser.add_argument("--dist_weight", type=float, default=1.0, help="weight of distance term")
    parser.add_argument("--reg_weight", type=float, default=0.001, help="weight of regularization term")
    parser.add_argument("--patch_size", type=int, default=32, help="discriminative loss patch size")

    parser.add_argument("--patch_length", type=int, default=128, help="dataset patch length")
    parser.add_argument("--random_sample_dataset", type=bool, default=False)
    parser.add_argument("--num_blocks_per_file", type=int, default=30)
    parser.add_argument("--transform", type=bool, default=True)
    parser.add_argument('--model_introduce:', type=str)
    args = parser.parse_args()
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    main_distributed(args)