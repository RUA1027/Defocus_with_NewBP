"""
使用配置系统的训练脚本示例
===========================

演示如何使用新的配置管理系统替代硬编码参数。

使用方法:
---------
    # 使用默认配置
    python demo_train.py
    
    # 使用自定义配置文件
    python demo_train.py --config config/experiment1.yaml
    
    # 命令行覆盖参数
    python demo_train.py --config config/default.yaml training.epochs=200 data.batch_size=4
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, Config
from models.zernike import DifferentiableZernikeGenerator
from models.aberration_net import AberrationNet, PolynomialAberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from trainer import DualBranchTrainer


def build_models_from_config(config: Config, device: str):
    """根据配置构建所有模型组件
    
    Args:
        config: 配置对象
        device: 计算设备
    
    Returns:
        tuple: (zernike_gen, aberration_net, restoration_net, physical_layer)
    """
    
    # 1. Zernike 生成器
    zernike_gen = DifferentiableZernikeGenerator(
        n_modes=config.physics.n_modes,
        pupil_size=config.physics.pupil_size,
        kernel_size=config.physics.kernel_size,
        oversample_factor=config.physics.oversample_factor,
        wavelengths=config.physics.wavelengths,
        ref_wavelength=config.physics.ref_wavelength,
        device=device
    )
    
    # 2. 像差预测网络
    if config.aberration_net.type == "polynomial":
        aberration_net = PolynomialAberrationNet(
            n_coeffs=config.aberration_net.n_coeffs,
            degree=config.aberration_net.polynomial.degree,
            a_max=config.aberration_net.a_max
        ).to(device)
        print(f"  ├─ 像差网络: PolynomialAberrationNet (degree={config.aberration_net.polynomial.degree})")
    else:
        aberration_net = AberrationNet(
            num_coeffs=config.aberration_net.n_coeffs,
            hidden_dim=config.aberration_net.mlp.hidden_dim,
            a_max=config.aberration_net.mlp.a_max_mlp,
            use_fourier=config.aberration_net.mlp.use_fourier
        ).to(device)
        print(f"  ├─ 像差网络: AberrationNet (hidden_dim={config.aberration_net.mlp.hidden_dim})")
    
    # 3. 图像复原网络
    restoration_net = RestorationNet(
        n_channels=config.restoration_net.n_channels,
        n_classes=config.restoration_net.n_classes,
        bilinear=config.restoration_net.bilinear,
        base_filters=config.restoration_net.base_filters,
        use_coords=config.restoration_net.use_coords
    ).to(device)
    print(f"  ├─ 复原网络: RestorationNet (base_filters={config.restoration_net.base_filters}, use_coords={config.restoration_net.use_coords})")
    
    # 4. 物理卷积层
    physical_layer = SpatiallyVaryingPhysicalLayer(
        aberration_net=aberration_net,
        zernike_generator=zernike_gen,
        patch_size=config.ola.patch_size,
        stride=config.ola.stride,
        pad_to_power_2=config.ola.pad_to_power_2
    ).to(device)
    print(f"  └─ 物理层: OLA (patch={config.ola.patch_size}, stride={config.ola.stride})")
    
    return zernike_gen, aberration_net, restoration_net, physical_layer


def build_trainer_from_config(config: Config, restoration_net, physical_layer, device: str):
    """根据配置构建训练器
    
    Args:
        config: 配置对象
        restoration_net: 复原网络
        physical_layer: 物理卷积层
        device: 计算设备
    
    Returns:
        DualBranchTrainer 对象
    """
    trainer = DualBranchTrainer(
        restoration_net=restoration_net,
        physical_layer=physical_layer,
        lr_restoration=config.training.optimizer.lr_restoration,
        lr_optics=config.training.optimizer.lr_optics,
        lambda_sup=config.training.loss.lambda_sup,
        lambda_coeff=config.training.loss.lambda_coeff,
        lambda_smooth=config.training.loss.lambda_smooth,
        device=device
    )
    
    return trainer


def generate_synthetic_data(config: Config, device: str):
    """生成合成训练数据
    
    Args:
        config: 配置对象
        device: 计算设备
    
    Returns:
        tuple: (x_gt, y_blurred) 清晰图像和模糊图像
    """
    B = config.data.batch_size
    C = config.restoration_net.n_channels
    H = config.data.image_height
    W = config.data.image_width
    
    # 创建测试图像
    x_gt = torch.zeros(B, C, H, W, device=device)
    
    # R 通道 / 单通道: 水平条纹
    if C == 1:
        # 单通道混合模式
        x_gt[:, 0, ::32, :] = 1.0
        x_gt[:, 0, :, ::32] += 1.0
        x_gt[:, 0, ::16, ::16] += 0.5
        x_gt = torch.clamp(x_gt, 0, 1)
    else:
        # R 通道: 水平条纹
        x_gt[:, 0, ::32, :] = 1.0
        # G 通道: 垂直条纹
        x_gt[:, 1, :, ::32] = 1.0
        # B 通道: 棋盘格
        x_gt[:, 2, ::16, ::16] = 0.5
    
    return x_gt


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用配置系统的训练脚本')
    parser.add_argument('--config', '-c', type=str, default='config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('overrides', nargs='*', 
                       help='覆盖参数，格式: key1.key2=value')
    
    args = parser.parse_args()
    
    # 加载配置
    print("\n" + "="*60)
    print("物理驱动盲去卷积网络 - 配置化训练")
    print("="*60)
    
    config = load_config(args.config, args.overrides if args.overrides else None)
    
    # 设置设备
    device = config.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA 不可用，切换到 CPU")
        device = 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(config.experiment.seed)
    print(f"✓ 随机种子: {config.experiment.seed}")
    
    # 构建模型
    print("\n构建模型...")
    zernike_gen, aberration_net, restoration_net, physical_layer = \
        build_models_from_config(config, device)
    
    # 构建训练器
    print("\n配置训练器...")
    trainer = build_trainer_from_config(config, restoration_net, physical_layer, device)
    print(f"  ├─ 学习率 (复原): {config.training.optimizer.lr_restoration}")
    print(f"  ├─ 学习率 (光学): {config.training.optimizer.lr_optics}")
    print(f"  ├─ λ_smooth: {config.training.loss.lambda_smooth}")
    print(f"  └─ λ_coeff: {config.training.loss.lambda_coeff}")
    
    # 生成合成数据
    print("\n生成合成数据...")
    x_gt = generate_synthetic_data(config, device)
    print(f"  └─ 数据形状: {x_gt.shape}")
    
    # 创建模糊图像 (使用随机像差)
    print("\n创建模糊图像...")
    with torch.no_grad():
        # 创建 GT 像差网络
        gt_aberration = PolynomialAberrationNet(
            n_coeffs=config.physics.n_modes, 
            degree=config.aberration_net.polynomial.degree,
            a_max=config.aberration_net.a_max
        ).to(device)
        nn.init.normal_(gt_aberration.poly_weights, mean=0, std=0.5)
        
        gt_layer = SpatiallyVaryingPhysicalLayer(
            gt_aberration, zernike_gen, 
            config.ola.patch_size, 
            config.ola.stride
        ).to(device)
        
        y_blurred = gt_layer(x_gt)
        y_blurred = y_blurred + 0.005 * torch.randn_like(y_blurred)
        y_blurred = torch.clamp(y_blurred, 0, 1)
    print(f"  └─ 模糊图像形状: {y_blurred.shape}")
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    epochs = config.experiment.epochs
    for epoch in range(epochs):
        stats = trainer.train_step(y_blurred, X_gt=None)  # 自监督
        
        if (epoch + 1) % config.experiment.log_interval == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {stats['loss']:.6f} | "
                  f"Smooth: {stats['loss_smooth']:.6f} | "
                  f"Grad_W: {stats['grad_W']:.4f} | "
                  f"Grad_Θ: {stats['grad_Theta']:.4f}")
            
            if stats['grad_Theta'] < 1e-9:
                print("  ⚠ 警告: 光学网络梯度消失!")
        
        # 保存检查点
        if (epoch + 1) % config.experiment.save_interval == 0:
            os.makedirs(config.experiment.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                config.experiment.output_dir, 
                f"checkpoint_epoch{epoch+1:03d}.pt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  ✓ 检查点已保存: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 打印最终配置
    print("\n使用的配置:")
    print("-"*40)
    print(f"物理参数:")
    print(f"  • kernel_size: {config.physics.kernel_size}")
    print(f"  • n_modes: {config.physics.n_modes}")
    print(f"  • wavelengths: {config.physics.wavelengths}")
    print(f"OLA 参数:")
    print(f"  • patch_size: {config.ola.patch_size}")
    print(f"  • stride: {config.ola.stride}")
    print(f"网络参数:")
    print(f"  • base_filters: {config.restoration_net.base_filters}")
    print(f"  • use_coords: {config.restoration_net.use_coords}")


if __name__ == '__main__':
    main()
