import torch
import torch.nn as nn
from config import Config
from models.zernike import DifferentiableZernikeGenerator
from models.aberration_net import AberrationNet, PolynomialAberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from trainer import DualBranchTrainer
from utils.dpdd_dataset import DPDDDataset
from torch.utils.data import DataLoader

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
        pad_to_power_2=config.ola.pad_to_power_2,
        use_newbp=config.ola.use_newbp
    ).to(device)
    name_algo = "NewBP" if config.ola.use_newbp else "Standard"
    print(f"  └─ 物理层: OLA (patch={config.ola.patch_size}, stride={config.ola.stride}, algo={name_algo})")
    
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
        lambda_image_reg=config.training.loss.lambda_image_reg,
        device=device
    )
    
    return trainer

def build_dataloader_from_config(config: Config, mode: str = 'train'):
    """根据配置构建 DataLoader
    
    Args:
        config: 配置对象
        mode: 数据集模式 ('train', 'val', 'test')
    
    Returns:
        DataLoader 对象
    """
    dataset = DPDDDataset(
        root_dir=config.data.data_root, 
        mode=mode, 
        transform=None # Default ToTensor
    )
    
    # 只有训练集需要 shuffle
    shuffle = (mode == 'train')
    
    loader = DataLoader(
        dataset, 
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True if config.experiment.device == 'cuda' else False
    )
    
    return loader
