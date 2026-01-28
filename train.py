"""
DPDD 物理驱动复原网络 - 训练脚本
================================
使用 utils.model_builder 统一构建组件，支持真实 DPDD 数据集训练。

Usage:
    python train.py --config config/default.yaml
"""

import argparse
import os
import torch
import sys
import math
from typing import Sized, cast
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from utils.model_builder import build_models_from_config, build_trainer_from_config, build_dataloader_from_config
from utils.metrics import PerformanceEvaluator


def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description='DPDD Training Script')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # 2. 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # 3. 设置环境
    device = config.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    config.experiment.device = device
    
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(config.experiment.seed)

    print(f"Device: {device}")
    print(f"Seed: {config.experiment.seed}")

    # 4. 构建数据
    print("\nInitializing DataLoaders...")
    try:
        train_loader = build_dataloader_from_config(config, mode='train')
        val_loader = build_dataloader_from_config(config, mode='val')
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        train_size = len(cast(Sized, train_dataset)) if hasattr(train_dataset, "__len__") else "Unknown"
        val_size = len(cast(Sized, val_dataset)) if hasattr(val_dataset, "__len__") else "Unknown"
        print(f"✓ Train set size: {train_size}")
        print(f"✓ Val set size: {val_size}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'preprocess_dpdd.py' has been run successfully.")
        return

    # 5. 构建模型与训练器
    print("\nBuilding Models...")
    zernike_gen, aberration_net, restoration_net, physical_layer = \
        build_models_from_config(config, device)
    
    print("\nInitializing Trainer...")
    trainer = build_trainer_from_config(config, restoration_net, physical_layer, device)
    
    # 6. 训练循环
    print("\n" + "="*60)
    print("Start Training")
    print("="*60)
    
    epochs = config.experiment.epochs
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}/{epochs}")

        # --- Stage Scheduling (自动基于 epoch) ---
        stage = trainer.get_stage(epoch)
        print(f"  -> Stage: {stage}")
        
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Train E{current_epoch}")
        
        # 获取 accumulation_steps 用于日志打印控制
        acc_steps = getattr(trainer, 'accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(pbar):
            # 处理字典格式的数据
            if isinstance(batch, dict):
                blur_imgs = batch['blur']
                sharp_imgs = batch['sharp']
                crop_info = batch.get('crop_info', None)
            else:
                # 兼容旧格式 (tuple)
                blur_imgs, sharp_imgs = batch
                crop_info = None

            # 传递 crop_info 和 batch_idx 以支持全局坐标对齐和梯度累积
            metrics = trainer.train_step(
                Y_blur=blur_imgs,
                X_gt=sharp_imgs,
                epoch=epoch,
                crop_info=crop_info
            )
            
            epoch_loss += metrics['loss']
            steps += 1
            
            # 仅在实际更新权重时（或每步）更新进度条
            if (batch_idx + 1) % acc_steps == 0:
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'Data': f"{metrics['loss_data']:.4f}",
                    'GradW': f"{metrics.get('grad_W', 0):.2f}"
                })
            
        avg_loss = epoch_loss / max(steps, 1)
        print(f"  -> Avg Train Loss: {avg_loss:.6f}")
        
        # --- Validation Phase (Multi-metric Evaluation) ---
        print("  -> Evaluating metrics on validation set...")
        metrics = PerformanceEvaluator.evaluate_model(
            restoration_net=trainer.restoration_net,
            physical_layer=trainer.physical_layer,
            val_loader=val_loader,
            device=device,
            smoothness_grid_size=config.training.smoothness_grid_size
        )

        try:
            from tabulate import tabulate  # type: ignore
            rows = []
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    rows.append([k, "NaN"])
                else:
                    rows.append([k, f"{v:.6f}"] if isinstance(v, float) else [k, str(v)])
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))
        except Exception:
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    v_str = "NaN"
                else:
                    v_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                print(f"  - {k}: {v_str}")

        # --- Checkpointing ---
        if current_epoch % config.experiment.save_interval == 0:
            os.makedirs(config.experiment.output_dir, exist_ok=True)
            save_path = os.path.join(config.experiment.output_dir, f"checkpoint_epoch{current_epoch:03d}.pt")
            trainer.save_checkpoint(save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")

    print("\nTraining Finished!")

if __name__ == "__main__":
    main()
