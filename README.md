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
pip install torch torchvision numpy matplotlib pyyaml tqdm
```

## Quick Start

### Training with Configuration System

All training workflows are managed by the unified entry point `demo_train.py`, driven entirely by configuration files.

```bash
# 1. Standard Training (Default configuration)
python demo_train.py

# 2. Using Presets
python demo_train.py --config config/lightweight.yaml    # Fast testing on CPU/Laptop
python demo_train.py --config config/high_resolution.yaml # High-res training (1K+ images)

# 3. Command Line Overrides (Flexible tuning)
python demo_train.py --config config/default.yaml training.epochs=200 data.batch_size=4
```

**Available Presets:**
- `config/default.yaml` - Standard balanced configuration.
- `config/lightweight.yaml` - Lightweight setup for debugging or quick testing.
- `config/high_resolution.yaml` - Optimized for high-resolution images.
- `config/mlp_experiment.yaml` - Experimental architecture options.

For a detailed guide on the configuration system, please refer to [CONFIG_USAGE_GUIDE.md](CONFIG_USAGE_GUIDE.md).

This process will:
1. Load dataset (synthetic or DPDD).
2. Train the dual-branch network.
3. Save visualizations to `results/`.

### Data Preparation (DPDD Dataset)

To train on the Dual-Pixel Defocus Deblurring (DPDD) dataset:

1. Download the Canon test set (`test_c`) to `data/dd_dp_dataset_png/test_c`.
2. Run the preprocessing script to format and resize images:
   ```bash
   python data/preprocess_dpdd.py
   ```
3. This will generate the standardized dataset at `data/dpdd_1024/`.

### Run Tests

```bash
# Test PSF energy conservation
python -m tests.test_psf_normalization

# Test gradient flow through physics layer
python -m tests.test_backward_flow

# Test NewBP custom autograd function
python -m tests.test_newbp_jacobian
```

## File Structure

```
defocus/
├── config/                   # Configuration files (YAML)
├── data/                     # Data loading and preprocessing
│   └── preprocess_dpdd.py    # DPDD dataset preparation script
├── models/
│   ├── __init__.py           # Module exports
│   ├── zernike.py            # Zernike basis & PSF generation
│   ├── newbp_convolution.py  # Custom Autograd for physics-aware gradients
│   ├── physical_layer.py     # OLA spatially-varying convolution
│   ├── aberration_net.py     # MLP for Zernike coefficients
│   └── restoration_net.py    # U-Net image restoration
├── utils/
│   ├── __init__.py
│   └── visualize.py          # Visualization utilities
├── tests/                    # Unit tests
├── trainer.py                # DualBranchTrainer
├── demo_train.py             # Main entry point
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

### 4. NewBP Spatial Convolution (`models/newbp_convolution.py`)

A custom Autograd function designed to handle the gradient flow of spatially-varying blur correctly.

```python
from models.newbp_convolution import NewBPSpatialConvolution

# Enable comparison mode (diagonal gradients only) for ablation studies
conv = NewBPSpatialConvolution(use_diagonal_only=False)
```

**Features:**
- **Exact Gradient Decomposition**: Separates gradients into **Direct** (Diagonal) and **Indirect** (Crosstalk) components via a non-diagonal Jacobian.
- **GPU Optimization**: Uses spatial `F.conv2d` with CuDNN optimizations (Winograd/Im2Col) for small kernels ($K \le 33$), significantly faster than FFT-based approaches for this kernel size regime.
- **Ablation Support**: Supports `use_diagonal_only=True` to simulate traditional gradients that ignore spatial crosstalk.

### 5. Physical Layer (`models/physical_layer.py`)

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
- Convolution: **NewBP Spatial Convolution** (see above) allows for precise gradient control during backpropagation.

### Optimization

- Separate learning rates: W (1e-4), Θ (1e-5)
- Gradient clipping: W (5.0), Θ (1.0)
- Optimizer: AdamW for both branches

## Verification Results

✅ **PSF Normalization Test**: All kernels sum to 1.0000 ± 1e-5  
✅ **Gradient Flow Test**: Gradients successfully propagate to both W and Θ  
✅ **NewBP Jacobian Test**: Verified non-diagonal structure of the convolution Jacobian.
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
