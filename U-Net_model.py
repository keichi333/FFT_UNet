import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import pandas as pd

# 随机种子
torch.manual_seed(42)
np.random.seed(42)

###########################
#------自定义数据集类------#
###########################

class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        
        # 获取图像和掩码文件列表
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        
        # 确保图像和掩码数量匹配
        assert len(self.image_names) == len(self.mask_names), "图像和掩码数量不匹配"
        
        # 定义CamVid数据集中的类别和对应的颜色
        self.class_names = [
            'Sky', 'Building', 'Pole', 'Road', 'Pavement', 
            'Tree', 'SignSymbol', 'Fence', 'Car', 
            'Pedestrian', 'Bicyclist', 'Unlabelled'
        ]
        
        # 定义类别到颜色的映射（BGR格式）
        self.id_to_color = {
            0: (128, 128, 128),  # Sky
            1: (128, 0, 0),      # Building
            2: (192, 192, 128),  # Pole
            3: (128, 64, 128),   # Road
            4: (0, 0, 192),      # Pavement
            5: (128, 128, 0),    # Tree
            6: (192, 128, 128),  # SignSymbol
            7: (64, 64, 128),    # Fence
            8: (64, 0, 128),     # Car
            9: (64, 64, 0),      # Pedestrian
            10: (0, 128, 192),   # Bicyclist
            11: (0, 0, 0)        # Unlabelled
        }
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 加载掩码
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        if len(mask.shape) == 3:
            mask = self.rgb_to_mask(mask)
        
        if self.transform:
            image = self.transform(image)
        
        # 转换掩码为张量
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def rgb_to_mask(self, rgb_mask):
        h, w, _ = rgb_mask.shape
        mask = np.zeros((h, w), dtype=np.int64)
        for id, color in self.id_to_color.items():
            class_mask = np.all(rgb_mask == color, axis=-1)
            mask[class_mask] = id
        return mask
    
    def visualize_sample(self, idx):
        image, mask = self[idx]
        image_np = image.permute(1, 2, 0).numpy()
        
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for id, color in self.id_to_color.items():
            color_mask[mask == id] = color[::-1]  # 转换为RGB
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image_np)
        ax1.set_title('Image')
        ax1.axis('off')
        ax2.imshow(color_mask)
        ax2.set_title('Mask')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

###########################
#----基于FFT的卷积实现----#
###########################

# 二维快速傅里叶变换（使用PyTorch内置函数）
def fft2(x):
    return torch.fft.fft2(x, norm='ortho')

# 二维快速傅里叶逆变换（使用PyTorch内置函数）
def ifft2(x):
    return torch.fft.ifft2(x, norm='ortho')

# FFT卷积实现（推理阶段使用）
class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化卷积核
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.register_buffer('weight_fft', torch.zeros(0))  # 预分配FFT权重
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        
        # 计算FFT尺寸（确保能容纳卷积结果）
        fft_H = height + self.kernel_size - 1
        fft_W = width + self.kernel_size - 1
        fft_H = 2 ** int(np.ceil(np.log2(fft_H)))
        fft_W = 2 ** int(np.ceil(np.log2(fft_W)))
        
        # 输入填充
        padded_x = F.pad(x, (0, fft_W - width, 0, fft_H - height))
        
        # 对输入进行二维FFT
        fft_x = torch.zeros(batch_size, in_channels, fft_H, fft_W, 
                           dtype=torch.cfloat, device=x.device)
        for b in range(batch_size):
            for c in range(in_channels):
                fft_x[b, c] = fft2(padded_x[b, c].to(torch.cfloat))
        
        # 使用预计算的FFT权重
        weight_fft = self.weight_fft
        if weight_fft.numel() == 0:
            # 首次推理时计算FFT权重
            weight_fft = torch.zeros(
                self.out_channels, self.in_channels, fft_H, fft_W,
                dtype=torch.cfloat, device=x.device
            )
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    # 卷积核填充
                    padded_weight = torch.zeros(fft_H, fft_W, device=x.device)
                    kernel = self.weight[oc, ic].to(torch.float32)
                    padded_weight[:self.kernel_size, :self.kernel_size] = kernel
                    # 计算FFT
                    weight_fft[oc, ic] = fft2(padded_weight.to(torch.cfloat))
            self.weight_fft = weight_fft
        
        # 频域乘法
        fft_output = torch.zeros(
            batch_size, self.out_channels, fft_H, fft_W,
            dtype=torch.cfloat, device=x.device
        )
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    fft_output[b, oc] += fft_x[b, ic] * weight_fft[oc, ic]
        
        # 逆FFT
        spatial_output = ifft2(fft_output).real
        
        # 裁剪结果
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        start_h = (fft_H - out_height) // 2
        start_w = (fft_W - out_width) // 2
        output = spatial_output[:, :, start_h:start_h+out_height, start_w:start_w+out_width]
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output

###########################
#----普通卷积实现----#
###########################

# 普通卷积模块（训练阶段使用）
class NormalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)

###########################
#----可切换卷积模块----#
###########################

# 卷积模块（根据模式切换普通卷积或FFT卷积）
class SwitchableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_fft_inference=True):
        super().__init__()
        self.use_fft_inference = use_fft_inference
        
        # 初始化两种卷积实现
        self.normal_conv = NormalConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.fft_conv = FFTConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        
        # 复制权重（确保两种卷积共享参数）
        self.fft_conv.weight.data = self.normal_conv.conv.weight.data.clone()
        if bias:
            self.fft_conv.bias.data = self.normal_conv.conv.bias.data.clone()
    
    def forward(self, x):
        if self.training:
            # 训练阶段使用普通卷积
            return self.normal_conv(x)
        else:
            # 推理阶段使用FFT卷积（如果启用）
            if self.use_fft_inference:
                return self.fft_conv(x)
            else:
                return self.normal_conv(x)

# 单个卷积操作（卷积->BN->ReLU）
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft_inference=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            SwitchableConv2d(in_channels, out_channels, kernel_size=3, padding=1, use_fft_inference=use_fft_inference),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SwitchableConv2d(out_channels, out_channels, kernel_size=3, padding=1, use_fft_inference=use_fft_inference),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# 下采样模块（编码器）
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft_inference=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_fft_inference)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

# 上采样模块（解码器）
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_fft_inference=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_fft_inference)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_fft_inference)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft_inference=True):
        super().__init__()
        self.conv = SwitchableConv2d(in_channels, out_channels, kernel_size=1, use_fft_inference=use_fft_inference)
    
    def forward(self, x):
        return self.conv(x)

################################
#------可切换FFT的U-Net模型------#
################################

class SwitchableFFTUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_fft_inference=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_fft_inference = use_fft_inference  # 控制推理阶段是否使用FFT
        
        self.inc = DoubleConv(n_channels, 64, use_fft_inference)
        self.down1 = Down(64, 128, use_fft_inference)
        self.down2 = Down(128, 256, use_fft_inference)
        self.down3 = Down(256, 512, use_fft_inference)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_fft_inference)
        self.up1 = Up(1024, 512 // factor, bilinear, use_fft_inference)
        self.up2 = Up(512, 256 // factor, bilinear, use_fft_inference)
        self.up3 = Up(256, 128 // factor, bilinear, use_fft_inference)
        self.up4 = Up(128, 64, bilinear, use_fft_inference)
        self.outc = OutConv(64, n_classes, use_fft_inference)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

###########################
#----定义训练和验证函数----#
###########################

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, scheduler=None, save_path='switchable_fft_unet_model.pth'):
    model.to(device)
    scaler = GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段（自动使用普通卷积，因为model.training=True）
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        # 验证阶段（使用普通卷积，与训练保持一致）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if scheduler:
            scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at {save_path}')
    
    return model

def test_model(model, test_loader, device, class_names, use_fft_inference=True):
    """测试模型，可选择是否在推理时使用FFT加速"""
    # 确保模型处于推理模式
    model.eval()
    # 设置推理模式是否使用FFT
    if hasattr(model, 'use_fft_inference'):
        model.use_fft_inference = use_fft_inference
    
    model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())
    
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    ious = []
    for i in range(len(class_names)):
        if i in np.unique(all_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            ious.append(iou)
        else:
            ious.append(0)
    
    print(f'Overall Accuracy: {acc:.4f}')
    print(f'Weighted F1-Score: {f1:.4f}')
    print('\nClass-wise IoU:')
    for i, (name, iou) in enumerate(zip(class_names, ious)):
        print(f'{name}: {iou:.4f}')
    
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, ious)
    plt.xlabel('Classes')
    plt.ylabel('IoU')
    plt.title('Class-wise IoU')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return acc, ious

def visualize_predictions(model, test_loader, device, dataset, num_samples=5, use_fft_inference=True):
    """可视化预测结果，可选择是否在推理时使用FFT加速"""
    # 确保模型处于推理模式
    model.eval()
    # 设置推理模式是否使用FFT
    if hasattr(model, 'use_fft_inference'):
        model.use_fft_inference = use_fft_inference
    
    model.to(device)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            image = images[0].unsqueeze(0).to(device)
            true_mask = masks[0].cpu().numpy()
            
            output = model(image)
            if output.shape[2:] != true_mask.shape:
                output = F.interpolate(output, size=true_mask.shape, mode='bilinear', align_corners=False)
            pred_mask = output.argmax(dim=1)[0].cpu().numpy()
            
            color_true_mask = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
            color_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            
            for id, color in dataset.id_to_color.items():
                color_true_mask[true_mask == id] = color[::-1]
                color_pred_mask[pred_mask == id] = color[::-1]
            
            axes[i, 0].imshow(images[0].permute(1, 2, 0).numpy())
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(color_true_mask)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(color_pred_mask)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

###########################
#----------主程序----------#
###########################

def main():
    data_dir = './UNET_process/data/camvid'
    n_classes = 12
    num_epochs = 30
    batch_size = 2
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CamVidDataset(
        image_dir=os.path.join(data_dir, 'train'),
        mask_dir=os.path.join(data_dir, 'trainannot'),
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = CamVidDataset(
        image_dir=os.path.join(data_dir, 'val'),
        mask_dir=os.path.join(data_dir, 'valannot'),
        transform=test_transform,
        mode='val'
    )
    
    test_dataset = CamVidDataset(
        image_dir=os.path.join(data_dir, 'test'),
        mask_dir=os.path.join(data_dir, 'testannot'),
        transform=test_transform,
        mode='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 可视化训练集样本
    train_dataset.visualize_sample(0)
    
    # 计算类别权重（处理类别不平衡）
    class_weights = torch.ones(n_classes, dtype=torch.float32).to(device)
    class_weights[11] = 0.5  # Unlabelled类别权重降低
    
    # 初始化模型（训练阶段使用普通卷积）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = SwitchableFFTUNet(n_channels=3, n_classes=n_classes, bilinear=True, use_fft_inference=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("开始训练模型...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        save_path='switchable_fft_camvid_unet_model.pth'
    )
    
    # 加载最佳模型
    model = SwitchableFFTUNet(n_channels=3, n_classes=n_classes, bilinear=True, use_fft_inference=True)
    model.load_state_dict(torch.load('switchable_fft_camvid_unet_model.pth', map_location=device))
    
    print("\n使用普通卷积进行测试...")
    test_model(model, test_loader, device, test_dataset.class_names, use_fft_inference=False)
    
    print("\n使用FFT加速卷积进行测试...")
    test_model(model, test_loader, device, test_dataset.class_names, use_fft_inference=True)
    
    print("\n使用普通卷积可视化预测结果...")
    visualize_predictions(model, test_loader, device, test_dataset, use_fft_inference=False)
    
    print("\n使用FFT加速卷积可视化预测结果...")
    visualize_predictions(model, test_loader, device, test_dataset, use_fft_inference=True)

if __name__ == '__main__':
    main()