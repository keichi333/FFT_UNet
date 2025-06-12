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
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib后端为Agg，避免GUI依赖
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import pandas as pd
import math
import time
from typing import Optional, Tuple, Union

# 随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建输出目录
os.makedirs('output_plots', exist_ok=True)

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
        plt.savefig(f'output_plots/sample_{idx}_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"样本 {idx} 可视化图已保存至 output_plots/sample_{idx}_visualization.png")

###########################
#----优化的FFT工具函数----#
###########################

def get_optimal_fft_size(size: int) -> int:
    """获取最优FFT尺寸（接近2的幂次）"""
    return 2 ** int(np.ceil(np.log2(size)))

def batch_fft2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """批量化二维FFT，优化GPU利用率"""
    return torch.fft.fft2(x, norm=norm)

def batch_ifft2d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """批量化二维逆FFT"""
    return torch.fft.ifft2(x, norm=norm)

def compute_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """计算卷积输出尺寸"""
    return (input_size + 2 * padding - kernel_size) // stride + 1

###########################
#----频域特征提取器----#
###########################

class FrequencyFeatureExtractor(nn.Module):
    """频域特征提取器，利用FFT的频域特性"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 频域压缩网络
        self.freq_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),  # *2因为复数有实部虚部
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, freq_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_features: 频域特征 (B, C, H, W) 复数张量
        Returns:
            attention_weights: 注意力权重 (B, C, 1, 1)
        """
        # 将复数特征转换为实数特征 [实部, 虚部]
        real_part = freq_features.real
        imag_part = freq_features.imag

        # 连接实部和虚部
        freq_cat = torch.cat([real_part, imag_part], dim=1)

        # 生成注意力权重
        attention = self.freq_compress(freq_cat)

        return attention

###########################
#----优化的FFT卷积实现----#
###########################

class OptimizedFFTConv2d(nn.Module):
    """优化的FFT卷积实现"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 use_frequency_attention: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_frequency_attention = use_frequency_attention

        # 初始化卷积核
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

        # FFT权重缓存
        self.register_buffer('weight_fft_cache', torch.zeros(0, dtype=torch.cfloat))
        self.register_buffer('cached_fft_size', torch.tensor([0, 0], dtype=torch.long))

        # 频域特征提取器
        if use_frequency_attention:
            self.freq_extractor = FrequencyFeatureExtractor(in_channels)

        # 性能统计
        self.register_buffer('inference_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_time', torch.tensor(0.0))

    def _get_fft_size(self, height: int, width: int) -> Tuple[int, int]:
        """计算优化的FFT尺寸"""
        fft_H = height + self.kernel_size - 1
        fft_W = width + self.kernel_size - 1
        return get_optimal_fft_size(fft_H), get_optimal_fft_size(fft_W)

    def _prepare_weight_fft(self, fft_H: int, fft_W: int, device: torch.device) -> torch.Tensor:
        """预处理权重的FFT，使用缓存机制"""
        current_size = torch.tensor([fft_H, fft_W], device=device)

        # 检查缓存是否有效
        if (self.weight_fft_cache.numel() > 0 and
            torch.equal(self.cached_fft_size, current_size)):
            return self.weight_fft_cache

        # 计算新的FFT权重
        weight_fft = torch.zeros(
            self.out_channels, self.in_channels, fft_H, fft_W,
            dtype=torch.cfloat, device=device
        )

        # 批量处理所有卷积核
        for oc in range(self.out_channels):
            # 创建填充后的权重张量
            padded_weights = torch.zeros(
                self.in_channels, fft_H, fft_W,
                dtype=torch.float32, device=device
            )

            # 复制权重到填充张量的左上角
            padded_weights[:, :self.kernel_size, :self.kernel_size] = self.weight[oc]

            # 批量FFT
            weight_fft[oc] = batch_fft2d(padded_weights.to(torch.cfloat))

        # 更新缓存
        self.weight_fft_cache = weight_fft
        self.cached_fft_size = current_size

        return weight_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start_time = time.time()

        batch_size, in_channels, height, width = x.size()

        # 计算FFT尺寸
        fft_H, fft_W = self._get_fft_size(height, width)

        # 输入填充 - 使用更高效的填充方式
        pad_h = fft_H - height
        pad_w = fft_W - width
        padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # 批量化FFT输入
        fft_x = batch_fft2d(padded_x.to(torch.cfloat))

        # 获取权重FFT
        weight_fft = self._prepare_weight_fft(fft_H, fft_W, x.device)

        # 频域注意力机制
        if self.use_frequency_attention and hasattr(self, 'freq_extractor'):
            freq_attention = self.freq_extractor(fft_x)
            fft_x = fft_x * freq_attention

        # 优化的频域卷积：使用Einstein求和约定进行批量矩阵乘法
        # fft_x: (B, IC, H, W), weight_fft: (OC, IC, H, W)
        # 输出: (B, OC, H, W)
        fft_output = torch.einsum('bihw,oihw->bohw', fft_x, weight_fft)

        # 批量化逆FFT
        spatial_output = batch_ifft2d(fft_output).real

        # 计算输出尺寸并裁剪
        out_height = compute_conv_output_size(height, self.kernel_size, self.stride, self.padding)
        out_width = compute_conv_output_size(width, self.kernel_size, self.stride, self.padding)

        # 智能裁剪：避免不必要的计算
        if self.padding == 0:
            # 标准裁剪
            crop_h = (fft_H - out_height) // 2
            crop_w = (fft_W - out_width) // 2
            output = spatial_output[:, :, crop_h:crop_h+out_height, crop_w:crop_w+out_width]
        else:
            # 考虑padding的裁剪
            start_h = self.kernel_size - 1 - self.padding
            start_w = self.kernel_size - 1 - self.padding
            output = spatial_output[:, :, start_h:start_h+out_height, start_w:start_w+out_width]

        # 应用步长（如果需要）
        if self.stride > 1:
            output = output[:, :, ::self.stride, ::self.stride]

        # 添加偏置
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        # 更新性能统计
        if not self.training:
            self.inference_count += 1
            self.total_time += time.time() - start_time

        return output

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        if self.inference_count > 0:
            avg_time = self.total_time / self.inference_count
            return {
                'inference_count': self.inference_count.item(),
                'total_time': self.total_time.item(),
                'average_time_per_inference': avg_time.item()
            }
        return {'inference_count': 0, 'total_time': 0.0, 'average_time_per_inference': 0.0}

###########################
#----混合精度优化器----#
###########################

class MixedPrecisionFFTConv2d(OptimizedFFTConv2d):
    """支持混合精度的FFT卷积"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在混合精度下执行FFT卷积
        return super().forward(x)

###########################
#----普通卷积实现----#
###########################

class NormalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)

###########################
#----智能卷积选择器----#
###########################

class AdaptiveConv2d(nn.Module):
    """智能卷积选择器：根据输入尺寸和卷积核大小自动选择最优实现"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 fft_threshold: int = 7, use_mixed_precision: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.fft_threshold = fft_threshold
        self.use_mixed_precision = use_mixed_precision

        # 初始化两种卷积实现
        self.normal_conv = NormalConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

        if use_mixed_precision:
            self.fft_conv = MixedPrecisionFFTConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        else:
            self.fft_conv = OptimizedFFTConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)

        # 权重同步
        self._sync_weights()

    def _sync_weights(self):
        """同步两种实现的权重"""
        with torch.no_grad():
            self.fft_conv.weight.data = self.normal_conv.conv.weight.data.clone()
            if self.normal_conv.conv.bias is not None and self.fft_conv.bias is not None:
                self.fft_conv.bias.data = self.normal_conv.conv.bias.data.clone()

    def _should_use_fft(self, x: torch.Tensor) -> bool:
        """判断是否应该使用FFT卷积"""
        if self.training:
            return False  # 训练时使用普通卷积

        # 根据输入尺寸和卷积核大小决定
        _, _, h, w = x.shape
        min_size = min(h, w)

        # FFT对于大卷积核和大图像更有效
        return (self.kernel_size >= self.fft_threshold and
                min_size >= 32)  # 避免在小图像上使用FFT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_use_fft(x):
            # 确保权重同步
            if not self.training:
                self._sync_weights()
            return self.fft_conv(x)
        else:
            return self.normal_conv(x)

    def get_performance_stats(self) -> dict:
        """获取FFT卷积的性能统计"""
        if hasattr(self.fft_conv, 'get_performance_stats'):
            return self.fft_conv.get_performance_stats()
        return {}

###########################
#----优化的U-Net组件----#
###########################

class OptimizedDoubleConv(nn.Module):
    """优化的双卷积模块"""
    def __init__(self, in_channels: int, out_channels: int, use_fft: bool = True):
        super().__init__()
        conv_class = AdaptiveConv2d if use_fft else NormalConv2d

        self.double_conv = nn.Sequential(
            conv_class(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_class(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OptimizedDown(nn.Module):
    """优化的下采样模块"""
    def __init__(self, in_channels: int, out_channels: int, use_fft: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            OptimizedDoubleConv(in_channels, out_channels, use_fft)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OptimizedUp(nn.Module):
    """优化的上采样模块"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_fft: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = OptimizedDoubleConv(in_channels, out_channels, use_fft)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = OptimizedDoubleConv(in_channels, out_channels, use_fft)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OptimizedOutConv(nn.Module):
    """优化的输出卷积"""
    def __init__(self, in_channels: int, out_channels: int, use_fft: bool = True):
        super().__init__()
        conv_class = AdaptiveConv2d if use_fft else NormalConv2d
        self.conv = conv_class(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

###########################
#----优化的FFT U-Net模型----#
###########################

class OptimizedFFTUNet(nn.Module):
    """优化的FFT U-Net模型"""
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True,
                 use_fft: bool = True, use_mixed_precision: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_fft = use_fft
        self.use_mixed_precision = use_mixed_precision

        self.inc = OptimizedDoubleConv(n_channels, 64, use_fft)
        self.down1 = OptimizedDown(64, 128, use_fft)
        self.down2 = OptimizedDown(128, 256, use_fft)
        self.down3 = OptimizedDown(256, 512, use_fft)
        factor = 2 if bilinear else 1
        self.down4 = OptimizedDown(512, 1024 // factor, use_fft)
        self.up1 = OptimizedUp(1024, 512 // factor, bilinear, use_fft)
        self.up2 = OptimizedUp(512, 256 // factor, bilinear, use_fft)
        self.up3 = OptimizedUp(256, 128 // factor, bilinear, use_fft)
        self.up4 = OptimizedUp(128, 64, bilinear, use_fft)
        self.outc = OptimizedOutConv(64, n_classes, use_fft)

    def forward(self, x):
        if self.use_mixed_precision and not self.training:
            with autocast():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
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

    def get_performance_stats(self) -> dict:
        """获取所有FFT卷积层的性能统计"""
        stats = {}
        for name, module in self.named_modules():
            if hasattr(module, 'get_performance_stats'):
                layer_stats = module.get_performance_stats()
                if layer_stats['inference_count'] > 0:
                    stats[name] = layer_stats
        return stats

###########################
#----优化的训练和验证函数----#
###########################

def train_model_optimized(model, train_loader, val_loader, optimizer, criterion, num_epochs, device,
                         scheduler=None, save_path='optimized_fft_unet_model.pth', use_mixed_precision=True):
    """优化的训练函数，支持混合精度训练和性能监控"""
    model.to(device)
    scaler = GradScaler() if use_mixed_precision else None
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

    print(f"开始训练，使用{'混合精度' if use_mixed_precision else '单精度'}...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_mixed_precision and scaler:
                with autocast():
                    outputs = model(images)
                    if outputs.shape[2:] != masks.shape[1:]:
                        outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                if use_mixed_precision:
                    with autocast():
                        outputs = model(images)
                        if outputs.shape[2:] != masks.shape[1:]:
                            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    if outputs.shape[2:] != masks.shape[1:]:
                        outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start_time

        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['epoch_times'].append(epoch_time)

        if scheduler:
            scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存至 {save_path}')

    return model, training_history

def benchmark_inference(model, test_loader, device, num_warmup=5, num_runs=20):
    """基准测试推理性能"""
    model.eval()
    model.to(device)

    print("开始推理性能基准测试...")

    # 预热
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_warmup:
                break
            images = images.to(device, non_blocking=True)
            _ = model(images)

    # 实际测试
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break

            images = images.to(device, non_blocking=True)
            torch.cuda.synchronize() if device.type == 'cuda' else None

            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"推理时间统计 (n={len(times)}):")
    print(f"  平均时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  最小时间: {min(times)*1000:.2f} ms")
    print(f"  最大时间: {max(times)*1000:.2f} ms")

    return times

def test_model_optimized(model, test_loader, device, class_names, benchmark=True):
    """优化的测试函数，包含性能基准测试"""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    inference_times = []

    print("开始模型测试...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            # 记录推理时间
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            outputs = model(images)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")

    # 计算指标
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算IoU
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

    # 输出结果
    print(f'\n=== 模型性能评估 ===')
    print(f'整体准确率: {acc:.4f}')
    print(f'加权F1分数: {f1:.4f}')
    print(f'平均IoU: {np.mean(ious):.4f}')

    # 推理性能统计
    avg_inference_time = np.mean(inference_times)
    print(f'\n=== 推理性能统计 ===')
    print(f'平均推理时间: {avg_inference_time*1000:.2f} ms')
    print(f'推理吞吐量: {1/avg_inference_time:.2f} samples/sec')

    # 获取FFT性能统计
    if hasattr(model, 'get_performance_stats'):
        fft_stats = model.get_performance_stats()
        if fft_stats:
            print(f'\n=== FFT性能统计 ===')
            for layer_name, stats in fft_stats.items():
                print(f'{layer_name}: {stats["average_time_per_inference"]*1000:.2f} ms/inference')

    print('\n=== 类别IoU ===')
    for i, (name, iou) in enumerate(zip(class_names, ious)):
        print(f'{name}: {iou:.4f}')

    # 可视化
    plt.figure(figsize=(15, 5))

    # 混淆矩阵
    plt.subplot(1, 3, 1)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 类别IoU
    plt.subplot(1, 3, 2)
    plt.bar(range(len(class_names)), ious)
    plt.xlabel('类别')
    plt.ylabel('IoU')
    plt.title('各类别IoU')
    plt.xticks(range(len(class_names)), class_names, rotation=45)

    # 推理时间分布
    plt.subplot(1, 3, 3)
    plt.hist(np.array(inference_times) * 1000, bins=20, alpha=0.7)
    plt.xlabel('推理时间 (ms)')
    plt.ylabel('频次')
    plt.title('推理时间分布')

    plt.tight_layout()
    plt.savefig('output_plots/test_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("测试结果分析图已保存至 output_plots/test_results_analysis.png")

    # 运行基准测试
    if benchmark:
        benchmark_times = benchmark_inference(model, test_loader, device)

    return acc, ious, avg_inference_time

def compare_implementations(model_normal, model_fft, test_loader, device, class_names):
    """比较普通卷积和FFT卷积的性能"""
    print("=== 实现方案性能对比 ===\n")

    # 测试普通卷积
    print("1. 普通卷积实现:")
    acc_normal, ious_normal, time_normal = test_model_optimized(
        model_normal, test_loader, device, class_names, benchmark=False
    )

    print("\n" + "="*50 + "\n")

    # 测试FFT卷积
    print("2. FFT加速实现:")
    acc_fft, ious_fft, time_fft = test_model_optimized(
        model_fft, test_loader, device, class_names, benchmark=False
    )

    # 性能对比
    print(f"\n=== 性能对比总结 ===")
    print(f"准确率 - 普通卷积: {acc_normal:.4f}, FFT: {acc_fft:.4f}")
    print(f"平均IoU - 普通卷积: {np.mean(ious_normal):.4f}, FFT: {np.mean(ious_fft):.4f}")
    print(f"推理时间 - 普通卷积: {time_normal*1000:.2f} ms, FFT: {time_fft*1000:.2f} ms")

    speedup = time_normal / time_fft
    print(f"加速比: {speedup:.2f}x {'(FFT更快)' if speedup > 1 else '(普通卷积更快)'}")

    return {
        'normal': {'acc': acc_normal, 'ious': ious_normal, 'time': time_normal},
        'fft': {'acc': acc_fft, 'ious': ious_fft, 'time': time_fft},
        'speedup': speedup
    }

def visualize_predictions_optimized(model, test_loader, device, dataset, num_samples=5):
    """优化的预测可视化"""
    model.eval()
    model.to(device)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))

    sample_count = 0
    with torch.no_grad():
        for images, masks in test_loader:
            if sample_count >= num_samples:
                break

            image = images[0].unsqueeze(0).to(device, non_blocking=True)
            true_mask = masks[0].cpu().numpy()

            # 推理
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            output = model(image)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time

            if output.shape[2:] != true_mask.shape:
                output = F.interpolate(output, size=true_mask.shape, mode='bilinear', align_corners=False)
            pred_mask = output.argmax(dim=1)[0].cpu().numpy()

            # 生成彩色掩码
            color_true_mask = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
            color_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

            for id, color in dataset.id_to_color.items():
                color_true_mask[true_mask == id] = color[::-1]
                color_pred_mask[pred_mask == id] = color[::-1]

            # 计算准确率
            accuracy = np.mean(true_mask == pred_mask)

            # 绘制
            axes[sample_count, 0].imshow(images[0].permute(1, 2, 0).numpy())
            axes[sample_count, 0].set_title('输入图像')
            axes[sample_count, 0].axis('off')

            axes[sample_count, 1].imshow(color_true_mask)
            axes[sample_count, 1].set_title('真实标签')
            axes[sample_count, 1].axis('off')

            axes[sample_count, 2].imshow(color_pred_mask)
            axes[sample_count, 2].set_title(f'预测结果\n准确率: {accuracy:.3f}\n推理时间: {inference_time*1000:.1f}ms')
            axes[sample_count, 2].axis('off')

            sample_count += 1

    plt.tight_layout()
    plt.savefig(f'output_plots/predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("预测结果可视化图已保存至 output_plots/predictions_visualization.png")

###########################
#----------主程序----------#
###########################

def main():
    data_dir = './data/camvid'
    n_classes = 12
    # 测试能否跑通
    num_epochs = 5
    batch_size = 2
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # 数据变换
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

    # 数据集
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

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 可视化训练集样本
    print("可视化数据集样本:")
    train_dataset.visualize_sample(0)

    # 计算类别权重
    class_weights = torch.ones(n_classes, dtype=torch.float32).to(device)
    class_weights[11] = 0.5  # Unlabelled类别权重降低

    # 训练模型
    print("="*60)
    print("开始训练模型...")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = OptimizedFFTUNet(n_channels=3, n_classes=n_classes, bilinear=True,
                            use_fft=True, use_mixed_precision=True)  # 训练时关闭FFT
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    trained_model, history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        save_path='optimized_fft_camvid_unet_model.pth',
        use_mixed_precision=True
    )

    # 可视化训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_times'])
    plt.xlabel('Epoch')
    plt.ylabel('时间 (秒)')
    plt.title('每轮训练时间')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('output_plots/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练历史图已保存至 output_plots/training_history.png")

    # 性能对比测试
    # print("="*60)
    # print("开始性能对比测试...")

    # 加载模型进行测试
    # model_normal = OptimizedFFTUNet(n_channels=3, n_classes=n_classes, bilinear=True,
    #                                use_fft=False, use_mixed_precision=True)
    # model_normal.load_state_dict(torch.load('optimized_fft_camvid_unet_model.pth', map_location=device))

    model_fft = OptimizedFFTUNet(n_channels=3, n_classes=n_classes, bilinear=True,
                                use_fft=True, use_mixed_precision=True)
    model_fft.load_state_dict(torch.load('optimized_fft_camvid_unet_model.pth', map_location=device))

    # 执行对比测试
    # comparison_results = compare_implementations(
    #     model_normal, model_fft, test_loader, device, test_dataset.class_names
    # )

    # 可视化预测结果
    # print("="*60)
    # print("可视化预测结果...")

    # print("\n使用FFT加速的预测结果:")
    # visualize_predictions_optimized(model_fft, test_loader, device, test_dataset)

if __name__ == '__main__':
    main()
