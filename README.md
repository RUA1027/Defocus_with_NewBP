# Physics-Driven Blind Deconvolution Network

A PyTorch implementation of a physics-driven blind deconvolution network for restoring images blurred by spatially varying optical aberrations. The blur is modeled using Zernike polynomials to parameterize wavefront aberrations, which are then converted to point spread functions (PSFs) through differentiable Fourier optics.

## Overview

This implementation features:

- **Dual-branch architecture**: Image restoration network + Optics identification network
- **Differentiable physics layer**: Zernike → Wavefront → PSF → Spatially-varying convolution
- **Self-supervised training**: Reblurring consistency loss allows training without ground-truth sharp images
- **GPU-efficient**: Overlap-Add (OLA) strategy with FFT convolution

## Architecture

```
Input (Blurred Y) 
    ↓
    ├─→ RestorationNet (U-Net) ─→ Restored X̂
    │                                    ↓
    └─→ AberrationNet (MLP)              │
         ↓                               │
    Zernike Coefficients                 │
         ↓                               │
    PSF Generation (FFT)                 │
         ↓                               │
    Spatially-Varying Blur ←─────────────┘
         ↓
    Reblurred Ŷ
         ↓
    Loss = MSE(Ŷ, Y)
```

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

## Quick Start

### 核心训练 (使用配置系统)

所有训练均通过统一的入口脚本 `demo_train.py` 执行，完全由配置文件驱动。

```bash
# 1. 使用默认配置 (标准训练)
python demo_train.py

# 2. 使用不同的预设配置
python demo_train.py --config config/lightweight.yaml    # 快速测试
python demo_train.py --config config/high_resolution.yaml # 高分辨率

# 3. 命令行覆盖参数 (灵活调参)
python demo_train.py --config config/default.yaml training.epochs=200 data.batch_size=4
```

可用的预设配置：
- `config/default.yaml` - 标准配置 (均衡)
- `config/lightweight.yaml` - 轻量级 (快速笔记本测试)
- `config/high_resolution.yaml` - 高分辨率 (1K+ 图像)
- `config/mlp_experiment.yaml` - 实验性架构 (MLP vs Polynomial)

详细的配置指南请参考 [CONFIG_USAGE_GUIDE.md](CONFIG_USAGE_GUIDE.md)。

This will:

1. Generate synthetic blurred data
2. Train the dual-branch network for 5 epochs
3. Save results to `results/demo_result.png`

### Run Tests

```bash
# Test PSF energy conservation
python -m tests.test_psf_normalization

# Test gradient flow through physics layer
python -m tests.test_backward_flow
```

## File Structure

```
defocus(Claude) - 副本/
├── models/
│   ├── __init__.py           # Module exports
│   ├── zernike.py            # Zernike basis & PSF generation
│   ├── aberration_net.py     # MLP for Zernike coefficients
│   ├── restoration_net.py    # U-Net image restoration
│   └── physical_layer.py     # OLA spatially-varying convolution
├── utils/
│   ├── __init__.py
│   └── visualize.py          # Visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_psf_normalization.py
│   └── test_backward_flow.py
├── trainer.py                # DualBranchTrainer
├── demo_train.py             # Example training script
└── README.md
```

## Key Components

### 1. Zernike PSF Generator (`models/zernike.py`)

Converts Zernike coefficients to PSF kernels:

```python
from models import DifferentiableZernikeGenerator

generator = DifferentiableZernikeGenerator(n_modes=15, pupil_size=64, kernel_size=33)
coeffs = torch.randn(10, 15)  # [Batch, Ncoeffs]
psf_kernels = generator(coeffs)  # [Batch, 1, 33, 33]
```

**Physics pipeline:**

1. Wavefront: Φ = 2π · Σ aₘ · Zₘ(ρ,θ)
2. Pupil: P = A · exp(iΦ)
3. PSF: K = |FFT2(P)|² (normalized)

### 2. Aberration Network (`models/aberration_net.py`)

Predicts spatially-varying Zernike coefficients:

```python
from models import AberrationNet

net = AberrationNet(num_coeffs=15, hidden_dim=64, a_max=2.0)
coords = torch.tensor([[-0.5, 0.3], [0.2, -0.8]])  # Normalized coordinates
coeffs = net(coords)  # [2, 15]
```

### 3. Restoration Network (`models/restoration_net.py`)

U-Net with residual learning:

```python
from models import RestorationNet

net = RestorationNet(n_channels=1, n_classes=1, base_filters=32)
blurred = torch.randn(2, 1, 256, 256)
restored = net(blurred)  # [2, 1, 256, 256]
```

### 4. Physical Layer (`models/physical_layer.py`)

Spatially-varying convolution via Overlap-Add:

```python
from models import SpatiallyVaryingPhysicalLayer

layer = SpatiallyVaryingPhysicalLayer(
    aberration_net=aberration_net,
    zernike_generator=zernike_gen,
    patch_size=128,
    stride=64
)

sharp_image = torch.randn(2, 1, 256, 256)
blurred_image = layer(sharp_image)  # [2, 1, 256, 256]
```

## Training

### Basic Usage

```python
from trainer import DualBranchTrainer

trainer = DualBranchTrainer(
    restoration_net=restoration_net,
    physical_layer=physical_layer,
    lr_restoration=1e-4,
    lr_optics=1e-5
)

# Training step
stats = trainer.train_step(Y_blurred, X_gt=None)  # Unsupervised
print(f"Loss: {stats['loss']}, Grad W: {stats['grad_W']}, Grad Theta: {stats['grad_Theta']}")
```

### Loss Function

- **Data Consistency**: L_data = MSE(Ŷ, Y) where Ŷ = PhysicalLayer(RestorationNet(Y))
- **Supervised** (optional): L_sup = MSE(X̂, X_gt)
- **Regularization**: L_coeff = mean(a²) for coefficient sparsity
- **Total**: L = L_data + λ_sup·L_sup + λ_coeff·L_coeff

## Visualization

```python
from utils import plot_psf_grid, plot_coefficient_maps

# Visualize PSFs across the field
plot_psf_grid(physical_layer, H=256, W=256, device='cuda', filename='psf_grid.png')

# Visualize coefficient spatial distribution
plot_coefficient_maps(physical_layer, H=256, W=256, device='cuda', filename='coeff_maps.png')
```

## Technical Details

### Zernike Polynomials

- Uses Noll indexing (j=1 to 15 by default)
- Modes: Piston, Tilt, Defocus, Astigmatism, Coma, Spherical
- Normalization: RMS = 1 (Noll convention)

### Overlap-Add (OLA) Convolution

- Patch size: P = 128 pixels
- Stride: S = 64 pixels (50% overlap)
- Window: Hann 2D for smooth blending
- Convolution: FFT-based for efficiency

### Optimization

- Separate learning rates: W (1e-4), Θ (1e-5)
- Gradient clipping: W (5.0), Θ (1.0)
- Optimizer: AdamW for both branches

## Verification Results

✅ **PSF Normalization Test**: All kernels sum to 1.0000 ± 1e-5  
✅ **Gradient Flow Test**: Gradients successfully propagate to both W and Θ  
✅ **Demo Training**: Loss converges, gradients remain active

## References

1. **Zernike Polynomials**: Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence"
2. **Fourier Optics**: Goodman, J. W. (2005). "Introduction to Fourier Optics"
3. **U-Net**: Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
4. **Physics-Driven Learning**: Related to "Deep Optics" and "End-to-End Optimization" paradigms

## License

MIT License

## Citation

```bibtex
@software{physics_blind_deconv,
  title={Physics-Driven Blind Deconvolution Network},
  author={Your Name},
  year={2026}
}
```
