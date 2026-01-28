import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

'''
================================================================================
                    三阶段解耦训练策略 (Three-Stage Decoupled Training)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Physics Only (物理层单独训练)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 利用成对数据，单独训练 AberrationNet 准确拟合数据集的光学像差特性     │
│                                                                              │
│  数据流:                                                                     │
│    X_gt (清晰图像) ──▶ PhysicalLayer ──▶ Y_hat (重模糊)                     │
│                                                                              │
│  Loss = MSE(Y_hat, Y) + λ_coeff × ||coeffs||² + λ_smooth × TV(coeffs)       │
│                                                                              │
│  冻结: RestorationNet (❄️)     更新: AberrationNet (🔥)                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Restoration with Fixed Physics (固定物理层训练复原网络)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 在已知且准确的物理模型指导下，训练复原网络                            │
│                                                                              │
│  数据流:                                                                     │
│    Y (模糊图像) ──▶ RestorationNet ──▶ X_hat ──▶ PhysicalLayer ──▶ Y_hat   │
│                                                                              │
│  Loss = λ_sup × L1(X_hat, X_gt) + MSE(Y_hat, Y) + λ_image_reg × TV(X_hat)  │
│                                                                              │
│  冻结: AberrationNet (❄️)      更新: RestorationNet (🔥)                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Joint Fine-tuning (联合微调)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  目的: 联合微调，消除模块间的耦合误差                                        │
│                                                                              │
│  数据流:                                                                     │
│    Y ──▶ RestorationNet ──▶ X_hat ──▶ PhysicalLayer ──▶ Y_hat              │
│                                                                              │
│  Loss = 综合损失（所有项）                                                   │
│                                                                              │
│  更新: RestorationNet (🔥) + AberrationNet (🔥)                              │
└─────────────────────────────────────────────────────────────────────────────┘
'''
class DualBranchTrainer:
    """
    三阶段解耦训练器 (Three-Stage Decoupled Trainer)

    支持三种训练模式:
    - 'physics_only': 仅训练物理层 (Stage 1)
    - 'restoration_fixed_physics': 固定物理层训练复原网络 (Stage 2)
    - 'joint': 联合训练所有模块 (Stage 3)
    """

    VALID_STAGES = ('physics_only', 'restoration_fixed_physics', 'joint')

    def __init__(self,
                 restoration_net,
                 physical_layer,
                 lr_restoration,
                 lr_optics,
                 lambda_sup=1.0,
                 lambda_coeff=0.01,
                 lambda_smooth=0.01,
                 lambda_image_reg=0.0,
                 device='cuda',
                 accumulation_steps=1):

        self.device = device
        self.restoration_net = restoration_net.to(device)
        self.physical_layer = physical_layer.to(device)

        # Access internals for regularization
        self.aberration_net = physical_layer.aberration_net

        # 独立优化器
        self.optimizer_W = optim.AdamW(self.restoration_net.parameters(), lr=lr_restoration)
        self.optimizer_Theta = optim.AdamW(self.aberration_net.parameters(), lr=lr_optics)

        # 损失权重
        self.lambda_sup = lambda_sup
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_image_reg = lambda_image_reg

        # 梯度累积
        self.accumulation_steps = max(1, accumulation_steps)
        self.accumulation_counter = 0

        # 损失函数
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        # 当前训练阶段
        self._current_stage = 'joint'

        # History
        self.history = {
            'loss_total': [], 'loss_data': [], 'loss_sup': [],
            'grad_norm_W': [], 'grad_norm_Theta': []
        }

    # =========================================================================
    #                          冻结/解冻工具函数
    # =========================================================================
    def _set_trainable(self, module: nn.Module, trainable: bool):
        for param in module.parameters():
            param.requires_grad = trainable

    def _freeze_restoration(self):
        self._set_trainable(self.restoration_net, False)

    def _unfreeze_restoration(self):
        self._set_trainable(self.restoration_net, True)

    def _freeze_physics(self):
        self._set_trainable(self.aberration_net, False)

    def _unfreeze_physics(self):
        self._set_trainable(self.aberration_net, True)

    def set_stage(self, stage: str):
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

        self._current_stage = stage

        if stage == 'physics_only':
            self._freeze_restoration()
            self._unfreeze_physics()
        elif stage == 'restoration_fixed_physics':
            self._unfreeze_restoration()
            self._freeze_physics()
        elif stage == 'joint':
            self._unfreeze_restoration()
            self._unfreeze_physics()

    # =========================================================================
    #                              核心训练步骤
    # =========================================================================
    def train_step(self, Y, X_gt=None, crop_info=None, batch_idx=0, stage=None):
        """
        执行一个训练步骤，支持三阶段解耦训练、梯度累积和全局坐标对齐。
        """
        current_stage = stage if stage is not None else self._current_stage
        if current_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{current_stage}'")

        Y = Y.to(self.device)
        if X_gt is not None:
            X_gt = X_gt.to(self.device)
        if crop_info is not None:
            crop_info = crop_info.to(self.device)

        if current_stage in ('physics_only', 'restoration_fixed_physics') and X_gt is None:
            raise ValueError(f"Stage '{current_stage}' requires X_gt (ground truth sharp image)")

        # 梯度累积：仅在第一个累积步骤清除梯度
        if self.accumulation_counter == 0:
            if current_stage == 'physics_only':
                self.optimizer_Theta.zero_grad()
            elif current_stage == 'restoration_fixed_physics':
                self.optimizer_W.zero_grad()
            else:
                self.optimizer_W.zero_grad()
                self.optimizer_Theta.zero_grad()

        # ========================== Stage Logic ==============================
        if current_stage == 'physics_only':
            Y_hat = self.physical_layer(X_gt, crop_info=crop_info)
            X_hat = X_gt
            loss_data = self.criterion_mse(Y_hat, Y)
            loss_sup = torch.tensor(0.0, device=self.device)
        elif current_stage == 'restoration_fixed_physics':
            X_hat = self.restoration_net(Y)
            Y_hat = self.physical_layer(X_hat, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_hat, Y)
            loss_sup = self.criterion_l1(X_hat, X_gt)
        else:
            X_hat = self.restoration_net(Y)
            Y_hat = self.physical_layer(X_hat, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_hat, Y)
            loss_sup = torch.tensor(0.0, device=self.device)
            if X_gt is not None:
                loss_sup = self.criterion_mse(X_hat, X_gt)

        # ========================== Regularization ===========================
        loss_coeff = torch.tensor(0.0, device=self.device)
        loss_smooth = torch.tensor(0.0, device=self.device)

        if current_stage in ('physics_only', 'joint'):
            coords = self.physical_layer.get_patch_centers(Y.shape[2], Y.shape[3], self.device)
            if coords.shape[0] > 64:
                indices = torch.randperm(coords.shape[0])[:64]
                coords_sample = coords[indices]
            else:
                coords_sample = coords

            coeffs = self.aberration_net(coords_sample)
            loss_coeff = torch.mean(coeffs**2)

            if self.lambda_smooth > 0:
                loss_smooth = self.compute_smoothness_loss()

        loss_image_reg = torch.tensor(0.0, device=self.device)
        if current_stage in ('restoration_fixed_physics', 'joint') and self.lambda_image_reg > 0:
            loss_image_reg = self.compute_image_tv_loss(X_hat)

        total_loss = loss_data + \
                     self.lambda_sup * loss_sup + \
                     self.lambda_coeff * loss_coeff + \
                     self.lambda_smooth * loss_smooth + \
                     self.lambda_image_reg * loss_image_reg

        scaled_loss = total_loss / self.accumulation_steps
        scaled_loss.backward()

        # ========================== Optimizer Step ============================
        self.accumulation_counter += 1
        should_step = (self.accumulation_counter >= self.accumulation_steps)

        gn_W = torch.tensor(0.0, device=self.device)
        gn_Theta = torch.tensor(0.0, device=self.device)

        if should_step:
            if current_stage == 'physics_only':
                gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), 1.0)
                self.optimizer_Theta.step()
            elif current_stage == 'restoration_fixed_physics':
                gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), 5.0)
                self.optimizer_W.step()
            else:
                gn_W = nn.utils.clip_grad_norm_(self.restoration_net.parameters(), 5.0)
                gn_Theta = nn.utils.clip_grad_norm_(self.aberration_net.parameters(), 1.0)
                self.optimizer_W.step()
                self.optimizer_Theta.step()

            self.accumulation_counter = 0

            self.history['loss_total'].append(total_loss.item())
            self.history['loss_data'].append(loss_data.item())
            self.history['grad_norm_W'].append(gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W)
            self.history['grad_norm_Theta'].append(gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta)

        return {
            'loss': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_sup': loss_sup.item(),
            'loss_coeff': loss_coeff.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_image_reg': loss_image_reg.item(),
            'grad_W': gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W,
            'grad_Theta': gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta,
            'stage': current_stage
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
