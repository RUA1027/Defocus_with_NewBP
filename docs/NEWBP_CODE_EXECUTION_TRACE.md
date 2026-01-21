# NewBP 代码执行路径注解

## 核心执行流程（以一次前向+反向为例）

```python
# ============================================================================
# 文件: models/physical_layer.py - SpatiallyVaryingPhysicalLayer
# ============================================================================

def forward(self, x_hat):
    """
    x_hat: [B, C, H, W]  清晰图像（来自RestorationNet）
    """
    B, C, H, W = x_hat.shape  # 例: [2, 3, 512, 512]
    P = self.patch_size   # 128
    S = self.stride       # 64
    K = self.kernel_size  # 31
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 1: 补丁分割
    # ────────────────────────────────────────────────────────────────────
    x_padded = F.pad(x_hat, ...)
    patches_unfolded = F.unfold(x_padded, kernel_size=P, stride=S)
    #    输入 [B, C, H_pad, W_pad]
    #         ↓
    #    输出 [B, C×P×P, N_patches]
    #         ↓ reshape
    patches = [B*N_patches, C, P, P]  # [128, 3, 128, 128]
    
    N_patches = 64  # (H_pad - P) / S + 1 = (576 - 128) / 64 + 1 = 8
                    # 类似地 W 方向也是 8，所以总共 64 个补丁
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 2: 坐标提取（空间编码关键！）
    # ────────────────────────────────────────────────────────────────────
    coords = self.get_patch_centers(H_pad, W_pad, device)
    #
    # 返回: [N_patches, 2]  每个补丁中心的归一化坐标
    #
    # 例如:
    # coords[0]  = [-0.778, -0.778]  补丁 1 (左上)
    # coords[1]  = [-0.778, -0.389]  补丁 2
    # ...
    # coords[63] = [ 0.778,  0.778]  补丁 64 (右下)
    #
    # 这些坐标包含了"光学系统在观察图像的哪个位置"的信息
    
    coords = coords.repeat(B, 1)  # 复制给Batch中的每个样本
    # 现在: [B*N_patches, 2] = [128, 2]
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 3: 空间变化的像差系数生成 ★★★关键★★★
    # ────────────────────────────────────────────────────────────────────
    coeffs = self.aberration_net(coords)
    #
    # self.aberration_net 输入: [128, 2]  坐标
    # self.aberration_net 输出: [128, 15] 每补丁的Zernike系数
    #
    # 不同坐标 → 不同系数！这就是"空间变化"
    #
    # 物理含义:
    # - 中心补丁 (坐标 [0, 0]): coeffs_center   = [0.10, -0.02, 0.15, ...]
    # - 边缘补丁 (坐标 [0.778, 0.778]): coeffs_edge = [0.15, -0.05, 0.25, ...]
    # 
    # 同一光学参数在不同视场位置有不同强度！
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 4: 空间变化的PSF核生成 ★★★关键★★★
    # ────────────────────────────────────────────────────────────────────
    kernels = self.zernike_generator(coeffs)
    #
    # self.zernike_generator 输入: [128, 15]  Zernike系数
    # self.zernike_generator 输出: [128, 3, 33, 33] 彩色PSF核
    #
    # 返回的 kernels:
    # kernels[0]  = [3, 33, 33]  补丁 1 的 PSF (中心，锐利)
    # kernels[1]  = [3, 33, 33]  补丁 2 的 PSF
    # ...
    # kernels[63] = [3, 33, 33]  补丁 64 的 PSF (边缘，模糊)
    #
    # 重要: 这 128 个核**完全不同**！
    # 它们编码了"光学系统在不同位置的PSF差异"
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 5: NewBP卷积 ★★★★★ NewBP激活★★★★★
    # ────────────────────────────────────────────────────────────────────
    fft_size = 2 ** math.ceil(math.log2(P + K - 1))
    
    if self.use_newbp:
        # ════════════════════════════════════════════════════════════════
        # 使用自定义 NewBP 函数
        # ════════════════════════════════════════════════════════════════
        y_patches = NewBPConvolutionFunction.apply(
            patches,       # [128, 3, 128, 128]
            kernels,       # [128, 3, 33, 33]   ◄─── 非均匀！
            K,             # 33
            P,             # 128
            fft_size       # 256 (下一个2的幂)
        )
        # 返回: [128, 3, 128, 128] 卷积后的补丁
        
        # ════════════════════════════════════════════════════════════════
        # NewBPConvolutionFunction.forward() 做了什么:
        # ════════════════════════════════════════════════════════════════
        #
        # 1. FFT正向卷积（标准）
        #    X_f = rfft2(patches, s=(256, 256))          [128, 3, freq_h, freq_w]
        #    K_f = rfft2(kernels, s=(256, 256))          [128, 3, freq_h, freq_w]
        #    Y_f = X_f * K_f                             逐元素乘法（FFT卷积）
        #    y = irfft2(Y_f, s=(256, 256))               [128, 3, 256, 256]
        #    y_patches = crop(y, [15:143, 15:143])       [128, 3, 128, 128]
        #
        # 2. 保存补丁和核用于反向
        #    ctx.save_for_backward(patches, kernels)
        #
        # 关键点：这一步与标准卷积在**前向**上是相同的！
        # NewBP的创新在反向！
        
    else:
        # 标准PyTorch自动求导（省略）
        ...
    
    # 现在 y_patches [128, 3, 128, 128] 是卷积后的补丁
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 6: 窗口加权 (Hann窗)
    # ────────────────────────────────────────────────────────────────────
    window_4d = self.window.view(1, 1, P, P)  # [1, 1, 128, 128]
    y_patches = y_patches * window_4d          # 补丁边缘逐渐变淡
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 7: Fold（补丁重组）+ 归一化
    # ────────────────────────────────────────────────────────────────────
    # Reshape back to fold format
    y_patches_reshaped = y_patches.reshape(B, N_patches, C * P * P)
    y_patches_reshaped = y_patches_reshaped.transpose(1, 2)  # [B, C*P*P, N]
    
    # Fold: 将补丁拼接回完整图像
    y_accum = F.fold(y_patches_reshaped, output_size=(H_pad, W_pad),
                     kernel_size=P, stride=S)
    
    # 同时累积窗口权重（用于归一化）
    w_accum = F.fold(w_patches_reshaped, output_size=(H_pad, W_pad),
                     kernel_size=P, stride=S)
    
    # 加权平均：处理重叠区域
    y_padded = y_accum / (w_accum + 1e-8)  # [B, C, H_pad, W_pad]
    
    # ────────────────────────────────────────────────────────────────────
    # 步骤 8: 裁剪回原尺寸
    # ────────────────────────────────────────────────────────────────────
    y_hat = y_padded[..., :H, :W]  # [B, C, H, W] = [2, 3, 512, 512]
    
    return y_hat


# ============================================================================
# 文件: models/newbp_convolution.py - NewBPConvolutionFunction
# ============================================================================

class NewBPConvolutionFunction(torch.autograd.Function):
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        当 loss.backward() 被调用时自动执行此函数
        
        grad_output: 来自下游的梯度 [128, 3, 128, 128]
                     代表 Loss 对 y_patches 的导数
        """
        
        patches, kernels = ctx.saved_tensors
        # patches [128, 3, 128, 128]
        # kernels [128, 3, 33, 33]  ◄─── 非均匀！每个补丁不同
        
        K, P = ctx.kernel_size, ctx.patch_size  # 33, 128
        fft_size = ctx.fft_size  # 256
        
        # ════════════════════════════════════════════════════════════════
        # 梯度计算 - 第一部分: ∂L/∂patches (对输入的梯度)
        # ════════════════════════════════════════════════════════════════
        
        # 翻转核（用于反卷积）
        kernels_flipped = torch.flip(kernels, dims=[-2, -1])
        # kernels_flipped [128, 3, 33, 33]
        # 
        # 关键观察：这些翻转的核各不相同！
        # kernels_flipped[0] ≠ kernels_flipped[1] ≠ ... ≠ kernels_flipped[127]
        
        # FFT正向卷积（反向的反卷积实现）
        G_f = torch.fft.rfft2(grad_output, s=(fft_size, fft_size))
        K_flip_f = torch.fft.rfft2(kernels_flipped, s=(fft_size, fft_size))
        
        # ★★★ 这一行是关键！★★★
        grad_X_f = G_f * K_flip_f  # 逐元素乘法
        # 
        # 因为 K_flip_f 中的每个批次元素都不同（非均匀核），
        # 所以这个乘法对每个补丁应用了不同的反卷积核！
        # 
        # 数学上:
        # grad_X_f[i] = G_f[i] * K_flip_f[i]
        #
        # 其中 K_flip_f[i] 是补丁 i 的 kernel 的翻转，
        # 包含了补丁 i 在前向传播中使用的非均匀PSF信息。
        
        # 反FFT
        grad_patches_large = torch.fft.irfft2(grad_X_f, s=(fft_size, fft_size))
        
        # 裁剪
        crop_start = K // 2  # 15
        grad_patches = grad_patches_large[..., crop_start:crop_start+P,
                                               crop_start:crop_start+P]
        # grad_patches [128, 3, 128, 128]
        
        # ════════════════════════════════════════════════════════════════
        # 梯度计算 - 第二部分: ∂L/∂kernels (对PSF核的梯度)
        # ════════════════════════════════════════════════════════════════
        
        X_f = torch.fft.rfft2(patches, s=(fft_size, fft_size))
        
        # 相关运算（不是卷积）
        grad_K_f = torch.conj(X_f) * G_f
        # 
        # 这计算了: dL/dK = X ★ (dL/dY)
        # 其中 ★ 表示相关（correlation），而不是卷积
        
        grad_kernels_large = torch.fft.irfft2(grad_K_f, s=(fft_size, fft_size))
        
        # FFT移位 + 裁剪（获取有效核梯度）
        grad_kernels_shifted = torch.fft.fftshift(grad_kernels_large, dim=(-2, -1))
        center_h, center_w = fft_size // 2, fft_size // 2
        k_start = K // 2
        
        grad_kernels = grad_kernels_shifted[...,
                                           center_h - k_start : center_h - k_start + K,
                                           center_w - k_start : center_w - k_start + K]
        # grad_kernels [128, 3, 33, 33]
        
        # ════════════════════════════════════════════════════════════════
        # 关键点总结
        # ════════════════════════════════════════════════════════════════
        #
        # 1. grad_patches 包含了 grad_output 通过**非均匀卷积核**反卷积的结果
        #
        # 2. grad_kernels 包含了如何调整PSF核以减少Loss的信息
        #    这会传播给AberrationNet，告诉它：
        #    "在补丁 i（坐标 C_i）应该学习系数使得PSF改变"
        #
        # 3. 这种非均匀反卷积就是 NewBP 的核心：
        #    每个补丁用自己的核，梯度也因此非均匀
        
        return grad_patches, grad_kernels, None, None, None


# ============================================================================
# 反向传播链的完整流动
# ============================================================================

# Loss 计算
loss = MSELoss(y_hat, y_target)

# 触发反向传播
loss.backward()

# ──→ step 1: loss 对 y_hat 的梯度
#     grad_y_hat = y_hat - y_target

# ──→ step 2: 梯度通过 Fold 反向操作
#     这涉及复杂的索引重映射（PyTorch自动处理）

# ──→ step 3: 梯度到达 NewBPConvolutionFunction.backward()
#     grad_output [128, 3, 128, 128] ←── 从上游来

# ──→ step 4a: 计算 grad_patches（对patches的梯度）
#     grad_patches = backward_conv2d(grad_output, kernels_flipped)
#                  ↓
#     patches.grad = grad_patches
#                  ↓
#     patches 来自 x_hat，所以梯度继续流向 RestorationNet

# ──→ step 4b: 计算 grad_kernels（对kernels的梯度）
#     grad_kernels = correlation(patches, grad_output)
#                  ↓
#     kernels.grad = grad_kernels
#                  ↓
#     kernels 来自 ZernikeGenerator，梯度继续流向

# ──→ step 5: ZernikeGenerator 的反向
#     ZernikeGenerator.backward():
#       输入: grad_kernels [128, 3, 33, 33]
#       处理: FFT的反向、PSF生成的反向、FFT的反向
#       输出: grad_coeffs [128, 15]

# ──→ step 6: AberrationNet 的反向
#     AberrationNet.backward():
#       输入: grad_coeffs [128, 15]
#       处理: tanh的反向、MLP的反向、FourierFeatureEncoding的反向
#       输出: grad_coords [128, 2]  (通常被忽略，因为coords是固定的)
#                    grad_W [各MLP层的权重梯度]

# ──→ step 7: RestorationNet 也有梯度（来自 grad_patches）
#     RestorationNet.backward():
#       input: grad_patches [128, 3, 128, 128]
#              (来自NewBPConvolutionFunction)
#       输出: grad_W_restoration [RestorationNet的权重梯度]


# ════════════════════════════════════════════════════════════════════════════
# 优化器更新
# ════════════════════════════════════════════════════════════════════════════

# 现在有了所有梯度，优化器进行参数更新：

optimizer.step()

# 这会更新：
# 1. RestorationNet 权重 W (来自 grad_W_restoration)
#    → 学习如何从模糊图像恢复清晰图像
#
# 2. AberrationNet 权重 θ (来自 grad_W_aberration)
#    → 学习空间坐标到像差系数的映射
#
# 这两个学习过程是耦合的！
# - RestorationNet 尝试做最好的恢复
# - AberrationNet 学习真实光学系统的特性
# - 两者共同最小化 y_hat 与真实图像的差异


# ════════════════════════════════════════════════════════════════════════════
# NewBP 的本质：非均匀性在此得到体现
# ════════════════════════════════════════════════════════════════════════════

# 原始补丁 0 (中心，坐标 [0, 0]):
#   前向:  patches[0] ⊗ kernels[0] → y_patches[0]
#          (kernels[0] 是根据中心像差系数生成的，相对锐利)
#
#   反向:  grad_patches[0] = grad_output[0] ⊗ kernels_flipped[0]
#          (反卷积使用相同的锐利核，恢复"中心应该是什么样")

# 原始补丁 63 (边缘，坐标 [0.778, 0.778]):
#   前向:  patches[63] ⊗ kernels[63] → y_patches[63]
#          (kernels[63] 是根据边缘像差系数生成的，更加模糊)
#
#   反向:  grad_patches[63] = grad_output[63] ⊗ kernels_flipped[63]
#          (反卷积使用同样模糊的核，恢复"边缘应该是什么样")

# 这就是非均匀性！
# grad_patches[0] 的恢复方式不同于 grad_patches[63]，
# 因为它们使用了不同的核。
