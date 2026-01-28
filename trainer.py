import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from typing import Any, Mapping

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
                 lambda_coeff=0.05,
                 lambda_smooth=0.1,
                 lambda_image_reg=0.0,
                 stage_schedule=None,
                 smoothness_grid_size=16,
                 device='cuda',
                 accumulation_steps=4):

        self.device = device
        self.restoration_net = restoration_net.to(device)
        self.physical_layer = physical_layer.to(device)

        # Access internals for regularization
        self.aberration_net = physical_layer.aberration_net

        # 独立优化器
        self.optimizer_W = optim.AdamW(self.restoration_net.parameters(), lr=lr_restoration)
        self.optimizer_Theta = optim.AdamW(self.aberration_net.parameters(), lr=lr_optics)

        # 兼容旧配置（已弃用的固定权重，仅保留字段）
        self.lambda_sup = lambda_sup
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_image_reg = lambda_image_reg

        # 三阶段调度 (可为 dict 或 dataclass)
        default_schedule = {
            'stage1_epochs': 80,
            'stage2_epochs': 80,
            'stage3_epochs': 40
        }
        self.stage_schedule: Any = stage_schedule if stage_schedule is not None else default_schedule

        # 平滑正则采样网格大小
        self.smoothness_grid_size = smoothness_grid_size

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
    #                          阶段调度与冻结策略
    # =========================================================================
    def _get_stage(self, epoch: int) -> str:
        """根据 epoch(0-indexed) 获取当前阶段"""
        if isinstance(self.stage_schedule, Mapping):
            s1 = self.stage_schedule.get('stage1_epochs', 80)
            s2 = self.stage_schedule.get('stage2_epochs', 80)
        else:
            s1 = getattr(self.stage_schedule, 'stage1_epochs', 80)
            s2 = getattr(self.stage_schedule, 'stage2_epochs', 80)

        if epoch < s1:
            return 'physics_only'
        elif epoch < s1 + s2:
            return 'restoration_fixed_physics'
        return 'joint'

    def _get_stage_weights(self, stage: str):
        """根据阶段返回动态 Loss 权重"""
        weights = {
            'w_data': 1.0,
            'w_sup': 0.0,
            'w_smooth': 0.0,
            'w_coeff': 0.0,
            'w_img_reg': 0.0
        }

        if stage == 'physics_only':
            weights.update({'w_data': 1.0, 'w_sup': 0.0, 'w_smooth': 0.1, 'w_coeff': 0.01, 'w_img_reg': 0.0})
        elif stage == 'restoration_fixed_physics':
            weights.update({'w_data': 0.1, 'w_sup': 1.0, 'w_smooth': 0.0, 'w_coeff': 0.0, 'w_img_reg': 0.001})
        elif stage == 'joint':
            weights.update({'w_data': 0.5, 'w_sup': 1.0, 'w_smooth': 0.05, 'w_coeff': 0.01, 'w_img_reg': 0.0001})

        return weights

    def _set_trainable(self, stage: str):
        """根据阶段快速冻结/解冻网络，并切换 train/eval 模式"""
        if stage == 'physics_only':
            for p in self.restoration_net.parameters():
                p.requires_grad = False
            for p in self.aberration_net.parameters():
                p.requires_grad = True
            self.restoration_net.eval()
            self.physical_layer.train()
        elif stage == 'restoration_fixed_physics':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            for p in self.aberration_net.parameters():
                p.requires_grad = False
            self.restoration_net.train()
            self.physical_layer.eval()
        elif stage == 'joint':
            for p in self.restoration_net.parameters():
                p.requires_grad = True
            for p in self.aberration_net.parameters():
                p.requires_grad = True
            self.restoration_net.train()
            self.physical_layer.train()
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

    def set_stage(self, stage: str):
        """兼容旧流程的手动设置（仍可用）"""
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")
        self._current_stage = stage
        self._set_trainable(stage)

    def get_stage(self, epoch: int) -> str:
        return self._get_stage(epoch)

    def get_stage_weights(self, epoch: int):
        return self._get_stage_weights(self._get_stage(epoch))

    # =========================================================================
    #                              核心训练步骤
    # =========================================================================
    def train_step(self, Y_blur, X_gt, epoch, crop_info=None):
        """
        执行一个训练步骤，内部根据 epoch 自动切换阶段并分配动态 Loss 权重。
        """
        current_stage = self._get_stage(epoch)
        self._current_stage = current_stage
        self._set_trainable(current_stage)

        weights = self._get_stage_weights(current_stage)
        w_data = weights['w_data']
        w_sup = weights['w_sup']
        w_smooth = weights['w_smooth']
        w_coeff = weights['w_coeff']
        w_img_reg = weights['w_img_reg']

        Y_blur = Y_blur.to(self.device)
        X_gt = X_gt.to(self.device)
        if crop_info is not None:
            crop_info = crop_info.to(self.device)

        # 梯度累积：仅在第一个累积步骤清除梯度
        if self.accumulation_counter == 0:
            if current_stage == 'physics_only':
                self.optimizer_Theta.zero_grad()
            elif current_stage == 'restoration_fixed_physics':
                self.optimizer_W.zero_grad()
            else:
                self.optimizer_W.zero_grad()
                self.optimizer_Theta.zero_grad()

        # ========================== Forward & Loss ===========================
        loss_data = torch.tensor(0.0, device=self.device)
        loss_sup = torch.tensor(0.0, device=self.device)
        loss_coeff = torch.tensor(0.0, device=self.device)
        loss_smooth = torch.tensor(0.0, device=self.device)
        loss_image_reg = torch.tensor(0.0, device=self.device)

        # Stage 1: 仅物理层
        if current_stage == 'physics_only':
            Y_reblur = self.physical_layer(X_gt, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_reblur, Y_blur)

            if w_coeff > 0 or w_smooth > 0:
                coords = self.physical_layer.get_patch_centers(
                    Y_blur.shape[2], Y_blur.shape[3], self.device
                )
                if coords.shape[0] > 64:
                    indices = torch.randperm(coords.shape[0])[:64]
                    coords = coords[indices]
                coeffs = self.aberration_net(coords)
                if w_coeff > 0:
                    loss_coeff = torch.mean(coeffs**2)
                if w_smooth > 0:
                    loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)

        # Stage 2/3: 复原网络参与
        else:
            X_hat = self.restoration_net(Y_blur)
            Y_reblur = self.physical_layer(X_hat, crop_info=crop_info)
            loss_data = self.criterion_mse(Y_reblur, Y_blur)
            loss_sup = self.criterion_l1(X_hat, X_gt)

            if w_img_reg > 0:
                loss_image_reg = self.compute_image_tv_loss(X_hat)

            if current_stage == 'joint' and (w_coeff > 0 or w_smooth > 0):
                coords = self.physical_layer.get_patch_centers(
                    Y_blur.shape[2], Y_blur.shape[3], self.device
                )
                if coords.shape[0] > 64:
                    indices = torch.randperm(coords.shape[0])[:64]
                    coords = coords[indices]
                coeffs = self.aberration_net(coords)
                if w_coeff > 0:
                    loss_coeff = torch.mean(coeffs**2)
                if w_smooth > 0:
                    loss_smooth = self.physical_layer.compute_coefficient_smoothness(self.smoothness_grid_size)

        # ========================== Weighted Loss ============================
        loss_data_w = w_data * loss_data
        loss_sup_w = w_sup * loss_sup
        loss_coeff_w = w_coeff * loss_coeff
        loss_smooth_w = w_smooth * loss_smooth
        loss_image_reg_w = w_img_reg * loss_image_reg

        total_loss = loss_data_w + loss_sup_w + loss_coeff_w + loss_smooth_w + loss_image_reg_w

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
            self.history['loss_data'].append(loss_data_w.item())
            self.history['grad_norm_W'].append(gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W)
            self.history['grad_norm_Theta'].append(gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta)

        return {
            'loss': total_loss.item(),
            'loss_data': loss_data_w.item(),
            'loss_sup': loss_sup_w.item(),
            'loss_coeff': loss_coeff_w.item(),
            'loss_smooth': loss_smooth_w.item(),
            'loss_image_reg': loss_image_reg_w.item(),
            'loss_data_raw': loss_data.item(),
            'loss_sup_raw': loss_sup.item(),
            'loss_coeff_raw': loss_coeff.item(),
            'loss_smooth_raw': loss_smooth.item(),
            'loss_image_reg_raw': loss_image_reg.item(),
            'grad_W': gn_W.item() if isinstance(gn_W, torch.Tensor) else gn_W,
            'grad_Theta': gn_Theta.item() if isinstance(gn_Theta, torch.Tensor) else gn_Theta,
            'stage': current_stage
        }

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
