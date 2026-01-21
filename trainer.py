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
                 lambda_sup=0.0,
                 lambda_coeff=0.01,
                 lambda_smooth=0.01,
                 device='cuda'):
        
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
        
        self.criterion_mse = nn.MSELoss()
        
        # History
        self.history = {'loss_total': [], 'loss_data': [], 'loss_sup': [], 'grad_norm_W': [], 'grad_norm_Theta': []}

    def train_step(self, Y, X_gt=None):
        """
        Y: Blurred input [B, C, H, W]
        X_gt: Ground truth sharp image (optional) [B, C, H, W]
        """
        Y = Y.to(self.device)
        if X_gt is not None:
            X_gt = X_gt.to(self.device)
            
        self.optimizer_W.zero_grad()
        self.optimizer_Theta.zero_grad()
        
        # 1. Restoration Branch
        X_hat = self.restoration_net(Y)
        
        # 2. Physical Simulation Branch (Reblurring)
        Y_hat = self.physical_layer(X_hat)
        
        # 3. Losses
        # Self-consistency loss (Data term)
        loss_data = self.criterion_mse(Y_hat, Y)
        
        # Supervised loss (if available)
        loss_sup = torch.tensor(0.0, device=self.device)
        if X_gt is not None:
            loss_sup = self.criterion_mse(X_hat, X_gt)
            
        # Regularization on Optics
        # We need access to coefficients to regularize them
        # Let's run aberration net again on a grid to compute reg terms without affecting physical layer graph too much
        # Or better, we can hook into physical layer? 
        # For efficiency, we construct a dummy coordinate grid or sample random points for regularization.
        # But strict OLA implies we used specific coords.
        # Since physical_layer doesn't expose coeffs easily, let's regenerate for regularization on a standard grid.
        
        # Sample a 8x8 grid for smoothness/sparsity regularization
        reg_grid_size = 8
        coords = self.physical_layer.get_patch_centers(Y.shape[2], Y.shape[3], self.device)
        # Downsample or take subset if too many
        if coords.shape[0] > 64:
            indices = torch.randperm(coords.shape[0])[:64]
            coords_sample = coords[indices]
        else:
            coords_sample = coords
            
        coeffs = self.aberration_net(coords_sample) # [N, C]
        
        # L2 Regularization on coefficients (prefer small aberrations)
        loss_coeff = torch.mean(coeffs**2)
        
        # Spatial Smoothness
        # Hard to compute gradients w.r.t u/v efficiently on random samples.
        # If we want explicit smoothness, we should sample a dense grid and compute finite differences.
        loss_smooth = torch.tensor(0.0, device=self.device)
        if self.lambda_smooth > 0:
            loss_smooth = self.compute_smoothness_loss()
        
        total_loss = loss_data + \
                     self.lambda_sup * loss_sup + \
                     self.lambda_coeff * loss_coeff + \
                     self.lambda_smooth * loss_smooth
                     
        # 4. Backward
        total_loss.backward()
        
        # 5. Gradient Clipping (Optional but good for stability)
        gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), 5.0)
        gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), 1.0)
        
        self.optimizer_W.step()
        self.optimizer_Theta.step()
        
        # Logging
        self.history['loss_total'].append(total_loss.item())
        self.history['loss_data'].append(loss_data.item())
        self.history['grad_norm_W'].append(gn_W.item())
        self.history['grad_norm_Theta'].append(gn_Theta.item())
        
        return {
            'loss': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_smooth': loss_smooth.item(), # Added logging
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

    def save_checkpoint(self, path):
        torch.save({
            'restoration_net': self.restoration_net.state_dict(),
            'aberration_net': self.aberration_net.state_dict(),
            'optimizer_W': self.optimizer_W.state_dict(),
            'optimizer_Theta': self.optimizer_Theta.state_dict()
        }, path)
