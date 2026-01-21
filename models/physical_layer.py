'''
核心功能架构:

输入图像 [B, C, H, W]
  ↓
分割成重叠补丁 (Overlap-Add 策略)
  ↓
对每个补丁中心计算坐标
  ↓
AberrationNet 预测该点的 Zernike 系数
  ↓
ZernikeGenerator 生成局部 PSF 卷积核
  ↓
FFT 频域卷积（高效计算）
  ↓
Hann 窗口加权拼接
  ↓
输出模糊图像 [B, C, H, W]
'''

'''
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SpatiallyVaryingPhysicalLayer                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─ 属性 (Attributes) ────────────────────────────────────────────────────┐  │
│  │  • aberration_net: AberrationNet                                       │  │
│  │  • zernike_generator: DifferentiableZernikeGenerator                   │  │
│  │  • patch_size (P): 128                                                 │  │
│  │  • stride (S): 64                                                      │  │
│  │  • kernel_size (K): 31                                                 │  │
│  │  • window: [128, 128] Hann 窗口 (缓冲)                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌─ 方法 (Methods) ───────────────────────────────────────────────────────┐  │
│  │                                                                         │  │
│  │  get_patch_centers(H, W, device)                                       │  │
│  │  ├─ 输入: 图像尺寸, 设备                                                │  │
│  │  └─ 输出: [N_patches, 2] 归一化坐标                                    │  │
│  │                                                                         │  │
│  │  forward(x_hat)                                                        │  │
│  │  ├─ 步骤 1: Pad(填充)                                                  │  │
│  │  ├─ 步骤 2: Unfold(分割补丁)                                           │  │
│  │  ├─ 步骤 3: Generate Kernels(生成 PSF)                                │  │
│  │  │  ├─ get_patch_centers()                                            │  │
│  │  │  ├─ AberrationNet(坐标) → 系数                                      │  │
│  │  │  └─ ZernikeGenerator(系数) → PSF                                    │  │
│  │  ├─ 步骤 4: FFT Conv(频域卷积)                                         │  │
│  │  ├─ 步骤 5: Window(窗口加权)                                           │  │
│  │  ├─ 步骤 6: Fold(拼接)                                                 │  │
│  │  ├─ 步骤 7: Normalize(归一化)                                          │  │
│  │  └─ 输出: y_hat [B, C, H, W]                                           │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘


数据维度变化轨迹 (以 B=2, C=3, H=512, W=512 为例):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2, 3, 512, 512]  x_hat (输入)
    ↓ Pad
[2, 3, 576, 576]  x_padded
    ↓ Unfold
[2, 3×128×128, 64]  patches_unfolded
    ↓ Reshape
[128, 3, 128, 128]  patches (B*N, C, P, P)

[64, 2]  coords (N_patches, 2)
    ↓ AberrationNet
[64, 15]  coeffs (N_patches, n_coeffs)
    ↓ ZernikeGenerator
[64, 3, 31, 31]  kernels (N_patches, C_k, K, K)

    patches ⊕ kernels (FFT 卷积)
    ↓
[128, 3, 128, 128]  y_patches_large
    ↓ Crop
[128, 3, 128, 128]  y_patches
    ↓ Window
[128, 3, 128, 128]  y_patches (加权)
    ↓ Reshape
[2, 3×128×128, 64]  y_patches_reshaped
    ↓ Fold
[2, 3, 576, 576]  y_accum
    ↓ Crop to [2, 3, 512, 512]
[2, 3, 512, 512]  y_hat (输出)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from .zernike import DifferentiableZernikeGenerator
from .aberration_net import AberrationNet
'''
┌─────────────────────────────────────────────────────────────┐
│ 输入: x_hat [B, C, H, W]                                      │
│ (如 [2, 3, 512, 512] - 批大小 2, RGB 图, 512x512 像素)      │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 1: 填充 (Pad)  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ x_padded [B, C, H_pad, W_pad]                       │
        │ (如 [2, 3, 576, 576] - 确保整除性)                  │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 2: Unfold     │
        │ (分割成补丁)        │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ patches [B*N, C, P, P]                              │
        │ (如 [2*64, 3, 128, 128] - 64 个补丁)               │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 3: 生成卷积核                                  │
        │  ├─ 计算补丁中心坐标                               │
        │  ├─ AberrationNet 预测 Zernike 系数                │
        │  └─ ZernikeGenerator 生成 PSF 核                   │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ kernels [B*N, C_k, K, K]                            │
        │ (如 [128, 3, 31, 31] - 每个补丁一个 PSF 核)        │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  步骤 4: FFT 卷积    │
        │ (频域相乘)          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ y_patches [B*N, C_out, P, P]                        │
        │ (如 [128, 3, 128, 128] - 卷积后补丁)               │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 5: 窗口加权                                    │
        │ y_patches *= window_2d                              │
        │ (补丁边界平滑过渡)                                  │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │  步骤 6: Fold (拼接)                                 │
        │  ├─ 输出拼接                                        │
        │  └─ 权重归一化                                      │
        └──────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ 裁剪回原尺寸        │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────────────────────────┐
        │ 输出: y_hat [B, C_out, H, W]                        │
        │ (模糊图像)                                          │
        └──────────────────────────────────────────────────────┘
'''
class SpatiallyVaryingPhysicalLayer(nn.Module):
    def __init__(self, 
                 aberration_net: nn.Module,
                 zernike_generator: DifferentiableZernikeGenerator,
                 patch_size,
                 stride,
                 pad_to_power_2=True):
        super().__init__()
        self.aberration_net = aberration_net
        self.zernike_generator = zernike_generator
        self.patch_size = patch_size
        self.stride = stride
        self.kernel_size = zernike_generator.kernel_size
        self.pad_to_power_2 = pad_to_power_2
        
        # Precompute window
        # Hann window 2D
        '''
        1D Hann 窗口:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.0 |        ╱╲
    |       ╱  ╲
0.5 |      ╱    ╲
    |     ╱      ╲
0.0 |____╱________╲____ 
    0   32   64   96  128

特点:
• 起点 (n=0): w=0
• 中点 (n=64): w=1 (最大值)
• 终点 (n=127): w≈0
• 平滑过渡，无尖角


2D Hann 窗口 (补丁):
━━━━━━━━━━━━━━━━━
    ┌─────────────────┐
    │     亮 (1.0)    │ ← 中心
    │   ╱         ╲   │
    │  ╱           ╲  │
    │ ╱             ╲ │
    │╱_______________╲│
    │ 暗 (0)         │ ← 边缘
    └─────────────────┘
    
中心最亮，边缘逐渐变暗
        '''
        # Hann 窗口是一个从 0 开始，逐渐升到 1，最后又降到 0 的平滑曲线。
        # Hann 窗口的解决了:Overlap-Add 中的补丁拼接,导致重叠区域的像素被重复计算了两次，能量不守恒的问题。
        # 通过在每个补丁上应用 Hann 窗口，补丁的边缘部分会被平滑地衰减到零，从而在拼接时避免了重复计算的问题。
        # 这样在重叠区域，多个补丁的贡献会自然地加权平均，确保最终图像的亮度和对比度保持一致。
        hann = torch.hann_window(patch_size)
        window_2d = torch.outer(hann, hann)
        self.register_buffer('window', window_2d)

    def get_patch_centers(self, H, W, device):
        # 计算所有补丁的中心坐标，并归一化到 [-1, 1] 范围内
        '''
        原始像素坐标计算:
┌─────────────────────────────────────────────────────┐
│ 补丁 1: 中心 (64, 64)     补丁 2: 中心 (128, 64)    │
│ 补丁 3: 中心 (64, 128)    补丁 4: 中心 (128, 128)   │
│ ...                                                  │
│ 补丁 64: 中心 (512, 512)                            │
│ 总共: √64 = 8 × 8 = 64 个补丁                       │
└─────────────────────────────────────────────────────┘

归一化过程:
y_norm = (y_pixels / 576) * 2 - 1
├─ 像素 (64) → 归一化 (64/576)*2 - 1 ≈ -0.778
├─ 像素 (288) → 归一化 (288/576)*2 - 1 ≈ 0
└─ 像素 (512) → 归一化 (512/576)*2 - 1 ≈ 0.778

输出形状:
┌──────────────────────────────────────────┐
│ coords [64, 2]                           │
│ 每行: [y_norm, x_norm]                   │
│ 如: [[-0.778, -0.778], [-0.778, -0.556], ...]
└──────────────────────────────────────────┘
        '''

        # Calculate number of patches along H and W
        # Unfold formula: L = (Size - Kernel)/Stride + 1
        n_h = (H - self.patch_size) // self.stride + 1
        n_w = (W - self.patch_size) // self.stride + 1
        
        # Note: Unfold might drop pixels if (H - P) % S != 0. 
        # Usually we pad input image to fit. 
        # For this implementation, we assume input is suitable or we handle padding internally.
        # Let's handle padding internally in forward.
        
        # Generate coordinates
        # Center of first patch: P/2
        # Center of second patch: P/2 + S
        # ...
        y_centers = torch.arange(n_h, device=device) * self.stride + self.patch_size / 2
        x_centers = torch.arange(n_w, device=device) * self.stride + self.patch_size / 2
        
        # Normalize to [-1, 1]
        # (y / H) * 2 - 1
        y_norm = (y_centers / H) * 2 - 1
        x_norm = (x_centers / W) * 2 - 1
        
        # Grid [N_h, N_w, 2]
        grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1) # [Nh, Nw, 2] (y, x) order to match AberrationNet
        
        return coords.reshape(-1, 2) # [N_patches, 2]

    def forward(self, x_hat):
        """
        x_hat: [B, C, H, W]
        Returns: y_hat [B, C, H, W]
        """
        B, C, H, W = x_hat.shape
        P = self.patch_size
        S = self.stride
        K = self.kernel_size
        
        # 1. Pad Input to ensure patches cover everything nicely
        # We need H, W to be P + k*S.
        # Or simply Unfold and Fold handles padding?
        # F.fold requires output_size to be specified.
        # If we just pad x_hat so that it fits unfold perfectly.
        '''
        原始图像: 512×512
补丁大小 P = 128, 步长 S = 64

计算补丁数:
n_h = (512 - 128) / 64 + 1 = 7

但最后一个补丁起始位置: 6 × 64 = 384
最后一个补丁范围: [384, 512) - 只覆盖到 512
缺失像素: 无

但如果是 513×513:
n_h = (513 - 128) / 64 + 1 = 7.03 → 7 (整除)
最后像素 513 无法被覆盖

所以需要填充:
513 + pad = 576 (能整除)
pad_h = (64 - (513 - 128) % 64) % 64 = (64 - 1) % 64 = 63
        '''
        pad_h = (S - (H - P) % S) % S
        pad_w = (S - (W - P) % S) % S
        
        # Also need to handle if H < P
        if H < P: pad_h += P - H
        if W < P: pad_w += P - W
        
        # Check if padding is too large for reflect mode
        # Reflect padding requires input_dim >= pad
        # If input size is smaller than padding, reflect mode will crash.
        if H < pad_h or W < pad_w:
            mode_pad = 'replicate' # Fallback for very small images
        else:
            mode_pad = 'reflect'
            
        x_padded = F.pad(x_hat, (0, pad_w, 0, pad_h), mode=mode_pad)
        
        H_pad, W_pad = x_padded.shape[2:]
        
        # 2. Unfold
        # [B, C*P*P, N_blocks]
        '''
[B, C, H_pad, W_pad] 
    ↓
[B, C*P*P, N_patches]  ← 每列是一个 P×P 补丁的展平版本
    ↓
reshape → [B*N_patches, C, P, P]

例如 B=2, C=3, N_patches=64:
[2, 3*128*128, 64]
    ↓
[2, 49152, 64]
    ↓
[128, 3, 128, 128]
        '''
        patches_unfolded = F.unfold(x_padded, kernel_size=P, stride=S)
        N_patches = patches_unfolded.shape[2]
        
        # Reshape to [B * N_patches, C, P, P]
        # Transpose to [B, N_patches, C*P*P]
        patches_unfolded = patches_unfolded.transpose(1, 2)
        # Reshape
        patches = patches_unfolded.reshape(B * N_patches, C, P, P)
        
        # 3. Generate Kernels
        # Get coordinates for ALL patches (same for every item in batch)
        # [N_patches, 2]
        '''
        不同补丁使用不同的卷积核:
        补丁 1 (中心: 图像左上)
  ├─ 坐标 (-0.778, -0.778)
  ├─ AberrationNet 预测系数 [a₁, a₂, ..., a₁₅]
  └─ ZernikeGenerator → PSF 核 K₁ [3, 31, 31]

补丁 2 (中心: 图像中心)
  ├─ 坐标 (0, 0)
  ├─ AberrationNet 预测系数 [a'₁, a'₂, ..., a'₁₅]
  └─ ZernikeGenerator → PSF 核 K₂ [3, 31, 31]

补丁 64 (中心: 图像右下)
  ├─ 坐标 (0.778, 0.778)
  ├─ AberrationNet 预测系数 [a''₁, a''₂, ..., a''₁₅]
  └─ ZernikeGenerator → PSF 核 K₆₄ [3, 31, 31]
        '''
        # 光学系统的像差通常随视场角变化（如边缘失焦）
        # 中心清晰 → 边缘模糊的真实光学现象

        coords_1img = self.get_patch_centers(H_pad, W_pad, x_hat.device)
        
        # Repeat for Batch
        # [B * N_patches, 2]
        coords = coords_1img.repeat(B, 1)
        
        # AberrationNet -> Coeffs
        coeffs = self.aberration_net(coords) # [B*N, Ncoeff]
        
        # ZernikeGenerator -> Kernels
        kernels = self.zernike_generator(coeffs) # [B*N, C_k, K, K]
        
        # Check channel consistency and determine output channels
        C_k = kernels.shape[1]
        if C == C_k:
            C_out = C
        elif C == 1 and C_k > 1:
            C_out = C_k
        elif C > 1 and C_k == 1:
            C_out = C
        else:
            raise ValueError(f"Channel mismatch: Input ({C}) and Kernel ({C_k}) are not compatible for broadcasting.")
        
        # 4. FFT Convolution
        # Pad sizes
        '''
        补丁大小: P = 128
核大小: K = 31 (Zernike PSF)

完整卷积输出大小:
out_size = P + K - 1 = 128 + 31 - 1 = 158

但我们只想要中心的 128×128:
crop_start = K // 2 = 15
crop_end = crop_start + P = 143
y_patches = y_large[15:143, 15:143]  # [128, 128]
        '''
        fft_size = P + K - 1
        if self.pad_to_power_2:
            fft_size = 2 ** math.ceil(math.log2(fft_size))
            
        # Pad signals
        # Patches: [B*N, C, P, P] -> [..., fft_size, fft_size]
        
        X_f = torch.fft.rfft2(patches, s=(fft_size, fft_size))
        K_f = torch.fft.rfft2(kernels, s=(fft_size, fft_size)) # Broadcasts over C
        
        Y_f = X_f * K_f # [B*N, C_out, size, size]
        
        y_patches_large = torch.fft.irfft2(Y_f, s=(fft_size, fft_size))
        
        # Crop result
        # We want PxP output.
        full_size = P + K - 1
        crop_start = K // 2
        
        # Note: If C_out != C (e.g. broadcast from 1 to 3), y_patches now has C_out channels
        y_patches = y_patches_large[..., crop_start : crop_start + P, crop_start : crop_start + P]
        
        # 5. Apply Window and Fold
        # Explicit dimension expansion for window
        window_4d = self.window.view(1, 1, P, P)
        y_patches = y_patches * window_4d
        
        # Reshape for folding: [B, C_out*P*P, N_patches]
        # Use C_out here
        y_patches_reshaped = y_patches.reshape(B, N_patches, C_out*P*P).transpose(1, 2)
        
        output_h = H_pad
        output_w = W_pad
        
        # Output will have C_out channels
        y_accum = F.fold(y_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        
        # 6. Normalization Map
        # Weight patches: ones * window
        # Expand match C_out
        w_patches = window_4d.expand(B * N_patches, C_out, P, P)
        w_patches_reshaped = w_patches.reshape(B, N_patches, C_out*P*P).transpose(1, 2)
        
        w_accum = F.fold(w_patches_reshaped, output_size=(output_h, output_w), kernel_size=P, stride=S)
        
        # Normalize
        y_hat_padded = y_accum / (w_accum + 1e-8)
        
        # Crop back to original size
        y_hat = y_hat_padded[..., :H, :W]
        
        return y_hat
'''
Overlap-Add 可视化:

补丁 1 (像素 [0:128, 0:128])  +  Hann 窗口
补丁 2 (像素 [64:192, 0:128]) +  Hann 窗口
         ↓ 重叠区间 [64:128]
    两个补丁的输出在此相加

Fold 过程:
   [0:64]       [64:128]      [128:192]
  ┌─────────┬─────────┬─────────┐
  │ 补丁1   │ 补丁1/2 │ 补丁2   │
  │ (仅窗口)│ (相加)  │ (仅窗口)│
  └─────────┴─────────┴─────────┘
  
加权归一化：
result[64:128] = (patch1_out + patch2_out) / (w1 + w2)
                = (w1*信号1 + w2*信号2) / (w1 + w2)  ← 加权平均
'''

'''
                    输入图像 x_hat
                        │
                        ▼
                    Unfold (分割)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
    补丁集合                        坐标集合
  (N 个 128×128)                   (N 个坐标)
        │                               │
        │                   AberrationNet(坐标)
        │                        │
        │                        ▼
        │                   Zernike 系数
        │                        │
        │                   ZernikeGenerator
        │                        │
        │                        ▼
        │                    PSF 核集合
        │                   (N 个 31×31)
        │                        │
        └───────┬─────────────────┘
                ▼
        FFT 卷积 (补丁 ⊗ PSF)
                │
                ▼
        卷积后补丁
      (N 个 128×128)
                │
                ├─ Hann 窗口加权 ─────┐
                │                     │
                ▼                     ▼
        加权后补丁              权重补丁
        y_patches             w_patches
                │                     │
                └──────┬──────────────┘
                       ▼
                  Fold (拼接)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   y_accum                       w_accum
  (累积输出)                     (累积权重)
        │                             │
        └──────┬──────────────────────┘
               ▼
        y_accum / w_accum
             (归一化)
               ▼
        裁剪回原尺寸
               ▼
        最终模糊图像 y_hat
'''