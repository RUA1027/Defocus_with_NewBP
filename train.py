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
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from utils.model_builder import build_models_from_config, build_trainer_from_config, build_dataloader_from_config


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
        print(f"✓ Train set size: {len(train_loader.dataset)}")
        print(f"✓ Val set size: {len(val_loader.dataset)}")
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
        
        # --- Training Phase ---
        # 注意: trainer 内部管理网络模式 (eval/train 切换由 trainer 或 net 自身处理吗? 
        # 通常 PyTorch 需要显式调用 model.train()。
        # 这里 train_step 假设是个自动挡，只要调用就会更新。
        # 如果 restoration_net 是 nn.Module, 最好确保它是 training mode。
        trainer.restoration_net.train()
        trainer.aberration_net.train()
        
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Train E{current_epoch}")
        for blur_imgs, sharp_imgs in pbar:
            # 数据搬运到 device 在 train_step 里也会做，但也可以这里做
            metrics = trainer.train_step(Y=blur_imgs, X_gt=sharp_imgs)
            
            epoch_loss += metrics['loss']
            steps += 1
            
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Data': f"{metrics['loss_data']:.4f}"
            })
            
        avg_loss = epoch_loss / max(steps, 1)
        print(f"  -> Avg Train Loss: {avg_loss:.6f}")
        
        # --- Validation Phase (Optional) ---
        # 简单跑一下验证集 Loss
        trainer.restoration_net.eval()
        trainer.aberration_net.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for blur_imgs, sharp_imgs in val_loader:
                blur_imgs = blur_imgs.to(device)
                sharp_imgs = sharp_imgs.to(device)
                
                # 手动前向传播计算 Loss，或者给 Trainer 加一个 val_step?
                # 这里简单手动计算 MSE
                X_hat = trainer.restoration_net(blur_imgs)
                feature_loss = trainer.criterion_mse(X_hat, sharp_imgs)
                val_loss += feature_loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"  -> Avg Val MSE: {avg_val_loss:.6f}")

        # --- Checkpointing ---
        if current_epoch % config.experiment.save_interval == 0:
            os.makedirs(config.experiment.output_dir, exist_ok=True)
            save_path = os.path.join(config.experiment.output_dir, f"checkpoint_epoch{current_epoch:03d}.pt")
            trainer.save_checkpoint(save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")

    print("\nTraining Finished!")

if __name__ == "__main__":
    main()
