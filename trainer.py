import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
'''
┌─────────────────────────────────────────────────────────────────┐
│                      一个训练步骤 (train_step)                    │
└─────────────────────────────────────────────────────────────────┘

输入:
  Y ────────────────────────────┐  [模糊图像，B×C×H×W]
  X_gt (可选) ────────────────┐ │  [清晰参考图像]
                              │ │
                    ┌─────────▼─▼──────────────────┐
                    │   Restoration Branch        │
                    │   (恢复/去模糊)              │
                    │   X_hat = restoration_net(Y)│
                    │                              │
                    │   输出: X_hat [B×C×H×W]      │
                    │   (还原后的清晰图像)         │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Physical Simulation      │
                    │   (重新模糊)                │
                    │  Y_hat = physical_layer    │
                    │          (X_hat)            │
                    │                              │
                    │  ├─ 计算像差 coeffs        │
                    │  ├─ 生成 PSF               │
                    │  └─ 卷积模糊 X_hat        │
                    │                              │
                    │   输出: Y_hat [B×C×H×W]   │
                    │   (重新模糊的图像)         │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │      损失函数计算                │
                    │                                  │
                    │  1. loss_data 自一致性         │
                    │     = MSE(Y_hat, Y)             │
                    │     Y 和 Y_hat 应该相近         │
                    │                                  │
                    │  2. loss_sup 监督损失 (可选)   │
                    │     = MSE(X_hat, X_gt)          │
                    │     如果有清晰参考图像          │
                    │                                  │
                    │  3. loss_coeff 系数正则化      │
                    │     = ||coeffs||²               │
                    │     鼓励像差较小                │
                    │                                  │
                    │  4. loss_smooth 平滑约束      │
                    │     = TV(coeffs_map)            │
                    │     鼓励像差在空间上平滑        │
                    │                                  │
                    │  总损失:                        │
                    │  L = loss_data +                │
                    │      λ_sup × loss_sup +        │
                    │      λ_coeff × loss_coeff +    │
                    │      λ_smooth × loss_smooth    │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │    反向传播与优化                │
                    │                                   │
                    │  梯度流:                         │
                    │  L → dL/dW (restoration_net)   │
                    │    → dL/dθ (aberration_net)     │
                    │                                   │
                    │  独立更新两个优化器:            │
                    │  ├─ optimizer_W                 │
                    │  │  (更新 restoration_net)      │
                    │  └─ optimizer_Theta             │
                    │     (更新 aberration_net)       │
                    │                                   │
                    │  梯度裁剪:                       │
                    │  ├─ restoration_net: max=5.0    │
                    │  └─ aberration_net: max=1.0     │
                    └──────────┬───────────────────────┘
                               │
输出:
  ├─ 更新后的 restoration_net 权重
  ├─ 更新后的 aberration_net 权重
  └─ 返回指标字典
'''
class DualBranchTrainer:
    def __init__(self, 
                 restoration_net, 
                 physical_layer,
                 lr_restoration,
                 lr_optics,
                 lambda_sup=1.0, # 当你有清晰的图像作为参考（Ground Truth）时，这部分决定了模型对“清晰图”的还原程度有多在乎。
                 lambda_coeff=0.01, # 防止网络预测出物理上不可能实现的巨大像差。它强制系数趋向于 0，符合“实际光学系统通常接近理想状态”的物理先验。
                 lambda_smooth=0.01, # 约束相邻区域的像差不要突变。因为镜头物理属性是连续的，光学像差在空间分布上应该是平滑变化的（如边缘劣化是渐进的），这能有效抑制噪声。
                 lambda_image_reg=0.0, # 保护复原后的图像不产生过多的伪影或高频噪声。在完全没有清晰图参考的自监督训练中，这个参数至关重要（防止模型通过制造噪声来强行拟合模糊图）。
                 device='cuda',
                 accumulation_steps=1):
        
        self.device = device
        self.restoration_net = restoration_net.to(device)
        self.physical_layer = physical_layer.to(device)
        
        # Access internals for regularization
        self.aberration_net = physical_layer.aberration_net
        
        self.optimizer_W = optim.AdamW(self.restoration_net.parameters(), lr=lr_restoration)
        self.optimizer_Theta = optim.AdamW(self.aberration_net.parameters(), lr=lr_optics)
        
        self.lambda_sup = lambda_sup
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_image_reg = lambda_image_reg
        
        # Gradient Accumulation Settings
        # accumulation_steps = 1: 标准训练（每步更新）
        # accumulation_steps = 4: 梯度累积 4 步后再更新（模拟 4 倍 Batch Size）
        self.accumulation_steps = max(1, accumulation_steps)
        self.accumulation_counter = 0
        
        self.criterion_mse = nn.MSELoss()
        
        # History - 记录还原后的真实 Loss（乘以 accumulation_steps 后，便于观察）
        self.history = {'loss_total': [], 'loss_data': [], 'loss_sup': [], 'grad_norm_W': [], 'grad_norm_Theta': []}

    def train_step(self, Y, X_gt=None, crop_info=None, batch_idx=0):
        """
        执行一个训练步骤，支持梯度累积和全局坐标对齐。
        
        Args:
            Y: 模糊输入 [B, C, H, W]
            X_gt: 清晰参考图像 [B, C, H, W]（可选）
            crop_info: 裁剪信息张量，用于全局坐标对齐。
                      形状：[4,] 或 [B, 4]，表示 [top_norm, left_norm, crop_h_norm, crop_w_norm]
            batch_idx: 当前批在 epoch 中的索引，用于梯度累积判断
        
        Returns:
            dict: 包含损失和梯度范数的字典
        """
        Y = Y.to(self.device)
        if X_gt is not None:
            X_gt = X_gt.to(self.device)
        if crop_info is not None:
            crop_info = crop_info.to(self.device)
        
        # 梯度累积：仅在第一个累积步骤清除梯度
        if self.accumulation_counter == 0:
            self.optimizer_W.zero_grad()
            self.optimizer_Theta.zero_grad()
        
        # 1. Restoration Branch
        X_hat = self.restoration_net(Y)
        
        # 2. Physical Simulation Branch (Reblurring) - 传递 crop_info 用于全局坐标对齐
        Y_hat = self.physical_layer(X_hat, crop_info=crop_info)
        
        # 3. 损失计算
        # 自一致性损失（数据项）
        loss_data = self.criterion_mse(Y_hat, Y)
        
        # 监督损失（如果有清晰参考图像）
        loss_sup = torch.tensor(0.0, device=self.device)
        if X_gt is not None:
            loss_sup = self.criterion_mse(X_hat, X_gt)
            
        # 光学正则化
        # 在网格上评估 AberrationNet 以计算正则化项
        reg_grid_size = 8
        coords = self.physical_layer.get_patch_centers(Y.shape[2], Y.shape[3], self.device)
        # 下采样或取子集
        if coords.shape[0] > 64:
            indices = torch.randperm(coords.shape[0])[:64]
            coords_sample = coords[indices]
        else:
            coords_sample = coords
            
        coeffs = self.aberration_net(coords_sample)  # [N, C]
        
        # L2 正则化：鼓励像差较小
        loss_coeff = torch.mean(coeffs**2)
        
        # 空间平滑性约束
        loss_smooth = torch.tensor(0.0, device=self.device)
        if self.lambda_smooth > 0:
            loss_smooth = self.compute_smoothness_loss()
        
        # 图像总变分正则化（对复原后的图像）
        loss_image_reg = torch.tensor(0.0, device=self.device)
        if self.lambda_image_reg > 0:
            loss_image_reg = self.compute_image_tv_loss(X_hat)
        
        # 总损失
        total_loss = loss_data + \
                     self.lambda_sup * loss_sup + \
                     self.lambda_coeff * loss_coeff + \
                     self.lambda_smooth * loss_smooth + \
                     self.lambda_image_reg * loss_image_reg
        
        # 梯度累积：除以累积步数以模拟更大的 Batch Size
        scaled_loss = total_loss / self.accumulation_steps
        scaled_loss.backward()
        
        # 更新累积计数器
        self.accumulation_counter += 1
        should_step = (self.accumulation_counter >= self.accumulation_steps)
        
        # 梯度裁剪和优化器更新
        gn_W = torch.tensor(0.0, device=self.device)
        gn_Theta = torch.tensor(0.0, device=self.device)
        
        if should_step:
            # 梯度裁剪（可选但对稳定性有帮助）
            gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), 5.0)
            gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), 1.0)
            
            # 优化器步骤
            self.optimizer_W.step()
            self.optimizer_Theta.step()
            
            # 重置累积计数器
            self.accumulation_counter = 0
        
        # 日志记录：记录还原后的真实损失（乘以 accumulation_steps）
        # 这样可以在日志中看到每个有效更新步骤的真实损失值
        if should_step:
            self.history['loss_total'].append(total_loss.item())
            self.history['loss_data'].append(loss_data.item())
            self.history['grad_norm_W'].append(gn_W.item())
            self.history['grad_norm_Theta'].append(gn_Theta.item())
        
        return {
            'loss': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_smooth': loss_smooth.item(), # Added logging
            'loss_image_reg': loss_image_reg.item(),
            'grad_W': gn_W.item(),
            'grad_Theta': gn_Theta.item()
        }

    def compute_smoothness_loss(self, grid_size=16):
        """
        Compute Total Variation (TV) loss on the coefficient map.
        Samples the AberrationNet on a regular grid to estimate smoothness.
        """
        # Create grid [-1, 1]
        y = torch.linspace(-1, 1, grid_size, device=self.device)
        x = torch.linspace(-1, 1, grid_size, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Flatten: [N, 2]
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        
        # Predict: [N, n_coeffs]
        coeffs = self.aberration_net(coords)
        
        # Reshape: [n_coeffs, H, W]
        coeffs_map = coeffs.view(grid_size, grid_size, -1).permute(2, 0, 1)
        
        # TV Loss: mean(|dx| + |dy|)
        dy = torch.abs(coeffs_map[:, 1:, :] - coeffs_map[:, :-1, :]).mean()
        dx = torch.abs(coeffs_map[:, :, 1:] - coeffs_map[:, :, :-1]).mean()
        
        return dy + dx

    def compute_image_tv_loss(self, img):
        """
        Compute Total Variation (TV) loss on the image.
        L_tv = mean(|dI/dx| + |dI/dy|)
        """
        B, C, H, W = img.shape
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return dy + dx

    def save_checkpoint(self, path):
        torch.save({
            'restoration_net': self.restoration_net.state_dict(),
            'aberration_net': self.aberration_net.state_dict(),
            'optimizer_W': self.optimizer_W.state_dict(),
            'optimizer_Theta': self.optimizer_Theta.state_dict()
        }, path)
