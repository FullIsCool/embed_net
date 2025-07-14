import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import os
import numpy as np
import torch.optim as optim
import tifffile as tiff


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_channels, out_channels, kernel_size):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size= kernel_size,
                                      stride=(2, 2, 2))

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512, 2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256, 2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128, 2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64, 2)
        self.decoder1 = conv_block(128, 64)

        self.conv_final = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_final(dec1))


class EmbedNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, count_dir, length=128,
                 random=False, num_blocks_per_file=125, transform=False):
        self.length = length
        self.transform = transform
        self.random = random
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.count_dir = count_dir       
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.tif')))
        self.count_paths = sorted(glob.glob(os.path.join(count_dir, '*.npy')))
        if not random:
            self.num_blocks_per_file = 5 * 5 * 5  
        else:
            self.num_blocks_per_file = num_blocks_per_file
        if ((len(self.image_paths) != len(self.label_paths)) or
            (len(self.image_paths) != len(self.count_paths))):
            raise ValueError()

    def __len__(self):
        return len(self.image_paths) * self.num_blocks_per_file

    def __getitem__(self, idx):
        length = self.length
        if not self.random:
            file_idx = idx // self.num_blocks_per_file
            block_idx = idx % self.num_blocks_per_file
            x_coord = block_idx // 25           
            y_coord = (block_idx % 25) // 5       
            z_coord = block_idx % 5               
            image = tiff.imread(self.image_paths[file_idx]).astype(np.float32)
            label = tiff.imread(self.label_paths[file_idx]).astype(np.float32)
            count = np.load(self.count_paths[file_idx])
            start_z = 40 * z_coord
            start_y = 40 * y_coord
            start_x = 40 * x_coord
            end_z = start_z + length
            end_y = start_y + length
            end_x = start_x + length

        else:
            file_idx = idx // self.num_blocks_per_file
            image = tiff.imread(self.image_paths[file_idx]).astype(np.float32)
            label = tiff.imread(self.label_paths[file_idx]).astype(np.float32)
            count = np.load(self.count_paths[file_idx])
            start_z = np.random.randint(0, 300-length)
            start_y = np.random.randint(0, 300-length)
            start_x = np.random.randint(0, 300-length)
            end_z = start_z + length
            end_y = start_y + length
            end_x = start_x + length
            
        image_block = image[start_z:end_z, start_y:end_y, start_x:end_x]    
        label_block = label[start_z:end_z, start_y:end_y, start_x:end_x]
        count_block = count[start_z:end_z, start_y:end_y, start_x:end_x]

        if self.transform:
            if np.random.rand() > 0.5:
                image_block = np.flip(image_block, axis=0)  
                label_block = np.flip(label_block, axis=0)
                count_block = np.flip(count_block, axis=0)
            if np.random.rand() > 0.5:
                image_block = np.flip(image_block, axis=1) 
                label_block = np.flip(label_block, axis=1)
                count_block = np.flip(count_block, axis=1)
            if np.random.rand() > 0.5:
                image_block = np.flip(image_block, axis=2)  
                label_block = np.flip(label_block, axis=2)
                count_block = np.flip(count_block, axis=2)

            num_rotations = np.random.choice([0, 1, 2, 3])  
            axe_rotation = [(0, 1), (1, 2), (0, 2)][np.random.choice([0, 1, 2])]
            image_block = np.rot90(image_block, k=num_rotations, axes=axe_rotation) 
            label_block = np.rot90(label_block, k=num_rotations, axes=axe_rotation)
            count_block = np.rot90(count_block, k=num_rotations, axes=axe_rotation)
            
        if image_block.max() != 0:
            image_block = image_block / image_block.max() 
        image_block = torch.tensor(image_block.copy(), dtype=torch.float32).unsqueeze(0)  
        label_block = torch.tensor(label_block.copy(), dtype=torch.float32)  
        count_block = torch.tensor(count_block.copy())
        return image_block, label_block, count_block


class UNet3D_embed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=4, enable_bce=True):
        super(UNet3D_embed, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)  
            )
            return block

        def up_conv(in_channels, out_channels, kernel_size):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size= kernel_size,
                                      stride=(2, 2, 2))
        self.enable_bce = enable_bce
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512, 2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256, 2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128, 2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64, 2)
        self.decoder1 = conv_block(128, 64)

        self.embed_head = nn.Conv3d(64, embed_dim, kernel_size=1)
        if self.enable_bce:
            self.sem_head = nn.Conv3d(64, 1, kernel_size=1)
        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        emb = self.embed_head(dec1) 
        if self.enable_bce:
            sem = self.sem_head(dec1) 
            sem = sem.squeeze(1) 
            emb = torch.sigmoid(emb)  
            return emb, sem
        else:
            emb = torch.sigmoid(emb)
            return emb


class MultiDilationConv3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, 
                 dilations=(1, 2, 3, 4), kernel_size=3, padding=1):
        super().__init__()
        self.groups = len(dilations)
        group_channels = out_channels // self.groups
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = d + padding -1
            conv = nn.Conv3d(in_channels,
                             group_channels,
                             kernel_size=kernel_size,
                             padding=pad,
                             dilation=d)
            self.convs.append(conv)

    def forward(self, x):
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        return torch.cat(outs, dim=1)

class UNet3D_embed_dilate(nn.Module):
    def __init__(self, in_channels=1, embed_dim=4, enable_bce=True):
        super(UNet3D_embed_dilate, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                MultiDilationConv3d(in_channels, out_channels, dilations=(1, 2, 3, 4), kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                MultiDilationConv3d(out_channels, out_channels, dilations=(1, 2, 3, 4), kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)  
            )
            return block

        def up_conv(in_channels, out_channels, kernel_size):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size= kernel_size,
                                      stride=(2, 2, 2))
        self.enable_bce = enable_bce
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512, 2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256, 2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128, 2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64, 2)
        self.decoder1 = conv_block(128, 64)

        self.embed_head = nn.Conv3d(64, embed_dim, kernel_size=1)
        if self.enable_bce:
            self.sem_head = nn.Conv3d(64, 1, kernel_size=1)
        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        emb = self.embed_head(dec1) 
        if self.enable_bce:
            sem = self.sem_head(dec1) 
            sem = sem.squeeze(1) 
            return emb, sem
        else:
            return emb


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        N = D * H * W           
        d_k = C                
        scale = np.sqrt(d_k)  

        query = self.query(x).view(batch_size, C, -1) 
        key = self.key(x).view(batch_size, C, -1).permute(0, 2, 1)  
        value = self.value(x).view(batch_size, C, -1)  

        attention_map = torch.bmm(query, key)  
        attention_map = attention_map / scale  
        attention_map = self.softmax(attention_map)  

        out = torch.bmm(attention_map, value)  
        out = out.view(batch_size, C, D, H, W)  

        return out + x  

class AttentionUNet3D_embed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=4, enable_bce=True):
        super(AttentionUNet3D_embed, self).__init__()
        self.enable_bce = enable_bce
        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_channels, out_channels, kernel_size):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=(2, 2, 2))

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bottleneck = conv_block(512, 1024)
        
        self.att4 = AttentionBlock(512)
        self.att3 = AttentionBlock(256)
        self.att2 = AttentionBlock(128)
        self.upconv4 = up_conv(1024, 512, 2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256, 2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128, 2)
        self.decoder2 = conv_block(256, 128) 
        
        self.upconv1 = up_conv(128, 64, 2)
        self.decoder1 = conv_block(128, 64)
        self.embed_head = nn.Conv3d(64, embed_dim, kernel_size=1)
        if self.enable_bce:
            self.sem_head = nn.Conv3d(64, 1, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool(enc1))  
        enc3 = self.encoder3(self.pool(enc2)) 
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        enc4 = self.att4(enc4)  
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        enc3 = self.att3(enc3)  
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        enc2 = self.att2(enc2)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        emb = self.embed_head(dec1) 
        if self.enable_bce:
            sem = self.sem_head(dec1)
            sem = sem.squeeze(1) 
            return emb, sem
        else:
            return emb   


if __name__ == "__main__":
    from loss import *
    
    model = UNet3D_embed(in_channels=1, embed_dim=8, enable_bce=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("/home/fit/guozengc/WORK/fhy/codes/embed/model/v428_discrimitive_loss/embed_net.pth"))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    image_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/origin"
    mask_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/mask_with_id"
    count_dir = r"/home/fit/guozengc/WORK/fhy/dataset_etv133/count_overlap"
    dataset = EmbedNetDataset(image_dir, mask_dir, count_dir,
                              random=True, num_blocks_per_file=10, transform=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=4, pin_memory=True)
    counter = 0
    s = time.time()
    for i, (image, label, count) in enumerate(dataset):
        print("image:", image.shape)
        print("label:", label.shape)
        print("count:", count.shape)
        out= model(image.unsqueeze(0).to(device))
        if i>10:
            break
    end= time.time()
    print("time:", (end-s)/(i+1))
    