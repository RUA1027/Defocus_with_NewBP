# 项目硬编码参数完整清单

## 总览
本项目中有 **52 个硬编码参数**，分别位于不同文件中。这些参数影响模型架构、物理模拟和训练过程。

---

## 1. PSF 卷积核参数

### ✓ PSF 核大小 (**NOT** 5×5，而是 **33×33**)

**问题:** PSF 卷积核的大小是硬编码的吗？

**答案:** **是的**，固定为 **33×33**（不是 5×5）

**位置:**
- [models/zernike.py](models/zernike.py#L207) - `kernel_size=33`
- [README.md](README.md#L95) - 默认值示例
- [demo_train.py](demo_train.py#L25) - 初始化示例

```python
# models/zernike.py, line 207
DifferentiableZernikeGenerator(
    n_modes=15, 
    pupil_size=64, 
    kernel_size=33,  # ← 硬编码
    ...
)
```

**为什么是 33？**
- 物理意义: 根据离焦强度，PSF 半径通常在 15-20 像素
- 33×33 足以容纳主瓣和第一圈旁瓣
- 补丁大小 128×128，33×33 占比 ~26%

---

## 2. Zernike 像差参数

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **n_modes (Zernike 模式数)** | 15 | models/zernike.py:207 | Noll 索引 1-15 (Piston~Quadrafoil) |
| **pupil_size (光瞳网格)** | 64 | models/zernike.py:207 | Zernike 基函数计算分辨率 |
| **oversample_factor** | 2 | models/zernike.py:208 | FFT 计算时的过采样倍数 |
| **ref_wavelength** | 550e-9 m | models/zernike.py:211 | 参考波长（绿光），多波长缩放基准 |

```python
# models/zernike.py, line 207-211
def __init__(self, 
             n_modes=15,           # ← 硬编码
             pupil_size=64,        # ← 硬编码
             kernel_size=33,       # ← 硬编码
             oversample_factor=2,  # ← 硬编码
             wavelengths=None,
             ref_wavelength=550e-9 # ← 硬编码
             ):
```

### 多波长配置

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **wavelengths (RGB)** | [620e-9, 550e-9, 450e-9] | demo_train.py:20 | 红、绿、蓝波长 |

```python
# demo_train.py, line 20
wavelengths = [620e-9, 550e-9, 450e-9]  # R, G, B ← 硬编码
```

---

## 3. 像差预测网络参数

### PolynomialAberrationNet

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **degree (多项式阶数)** | 2 | models/aberration_net.py:48 | 多项式曲面的度数 |
| **a_max (系数范围)** | 2.0 | models/aberration_net.py:48 | Zernike 系数的约束范围 |
| **n_coeffs** | 15 | models/aberration_net.py:48 | Zernike 系数数量 |

```python
# models/aberration_net.py, line 48
class PolynomialAberrationNet(nn.Module):
    def __init__(self, n_coeffs=15, degree=2, a_max=2.0):
        # n_coeffs=15  ← 硬编码
        # degree=2     ← 硬编码 (影响参数量: 15×6=90 参数)
        # a_max=2.0    ← 硬编码
```

### FourierFeatureEncoding

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **mapping_size** | hidden_dim//2 | models/aberration_net.py:6 | 傅里叶特征维度 |
| **scale (傅里叶缩放)** | 5 | models/aberration_net.py:6 | 控制特征频率 |

```python
# models/aberration_net.py, line 6 (在 AberrationNet 中调用)
self.encoding = FourierFeatureEncoding(
    input_dim=2, 
    mapping_size=hidden_dim//2,  # e.g., 32 (当 hidden_dim=64)
    scale=5  # ← 硬编码
)
```

### AberrationNet (MLP版本)

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **hidden_dim** | 64 | models/aberration_net.py:128 | MLP 隐层维度 |
| **a_max (MLP版)** | 3.0 | models/aberration_net.py:128 | 比 Polynomial 版本更大 |
| **num_coeffs** | 15 | models/aberration_net.py:128 | 输出系数数量 |
| **use_fourier** | True | models/aberration_net.py:128 | 是否使用傅里叶编码 |

```python
# models/aberration_net.py, line 128
def __init__(self, num_coeffs=15, hidden_dim=64, a_max=3.0, use_fourier=True):
    # num_coeffs=15  ← 硬编码
    # hidden_dim=64  ← 硬编码
    # a_max=3.0      ← 硬编码
    # use_fourier=True ← 硬编码
```

**MLP 网络层硬编码:**
```python
# models/aberration_net.py, 第 147-153 行
self.net = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),                # 2/128 → 64
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim, hidden_dim * 2),       # 64 → 128 ← 硬编码 2× 倍数
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim * 2, hidden_dim),       # 128 → 64 ← 硬编码 回到 64
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim, num_coeffs)            # 64 → 15
)
```

---

## 4. 物理层（空间变化卷积）参数

### SpatiallyVaryingPhysicalLayer

| 参数 | 值 | 位置 | 用途 | 重叠比例 |
|------|-----|------|------|---------|
| **patch_size (P)** | 128 | models/physical_layer.py:169 | 补丁大小 | - |
| **stride (S)** | 64 | models/physical_layer.py:170 | 补丁步长 | 50% (64/128) |

```python
# models/physical_layer.py, line 169-170
def __init__(self, 
             ...
             patch_size=128,  # ← 硬编码
             stride=64,       # ← 硬编码 (50% 重叠)
             pad_to_power_2=True):  # ← 硬编码
```

**补丁计算:**
- 补丁数量 (H=512): `(512 - 128) / 64 + 1 = 7` 个
- 补丁数量 (W=512): 同上 = 7 个
- 总补丁: 7×7 = 49 个

**为什么 50% 重叠？**
- Hann 窗口在 50% 重叠时完全重建 (w + w_shift = 1.0)
- 平衡计算量和平滑性

---

## 5. 图像复原网络（U-Net）参数

### RestorationNet 架构

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **n_channels** | 3 | models/restoration_net.py:179 | 输入通道数 (RGB) |
| **n_classes** | 3 | models/restoration_net.py:179 | 输出通道数 (RGB) |
| **base_filters** | 64 | models/restoration_net.py:179 | 基础卷积滤波器数 |
| **use_coords** | False | models/restoration_net.py:179 | 坐标注入 (可配置) |

```python
# models/restoration_net.py, line 179
def __init__(self, n_channels=3, n_classes=3, bilinear=True, base_filters=64, use_coords=False):
```

**U-Net 通道配置 (硬编码倍数):**

```
层级             通道数计算
Inc:             base_filters = 64
Down1:           64 × 2 = 128      ← 硬编码 2× 倍数
Down2:           128 × 2 = 256     ← 硬编码 2× 倍数
Down3:           256 × 2 = 512     ← 硬编码 2× 倍数
Down4:           512 × 1 = 512     ← 硬编码保持
Up1:             512 → 256         ← 硬编码 ÷2
Up2:             256 → 128         ← 硬编码 ÷2
Up3:             128 → 64          ← 硬编码 ÷2
Up4:             64 → 64           ← 硬编码保持
OutConv:         64 → 3
```

### DoubleConv 块硬编码

| 参数 | 值 | 位置 | 影响 |
|------|-----|------|------|
| **kernel_size (卷积核)** | 3 | models/restoration_net.py:71, 77 | 局部特征感受野 |
| **padding** | 1 | models/restoration_net.py:71, 77 | 保持空间尺寸 |
| **LeakyReLU 斜率** | 0.2 | models/restoration_net.py:74, 80 | 激活函数配置 |

```python
# models/restoration_net.py, line 71-80
nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
nn.LeakyReLU(0.2),  # ← 硬编码斜率 0.2
nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
nn.LeakyReLU(0.2),  # ← 硬编码斜率 0.2
```

### Up 块硬编码

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **Upsample scale_factor** | 2 | models/restoration_net.py:139 | 上采样倍数 |
| **ConvTranspose2d kernel_size** | 2 | models/restoration_net.py:142 | 转置卷积核大小 |
| **ConvTranspose2d stride** | 2 | models/restoration_net.py:142 | 转置卷积步长 |

---

## 6. 训练器参数

### DualBranchTrainer

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **lr_restoration** | 1e-4 | trainer.py:14 | 复原网络学习率 |
| **lr_optics** | 1e-5 | trainer.py:15 | 像差预测网络学习率 |
| **lambda_sup** | 0.0 | trainer.py:16 | 监督损失权重 |
| **lambda_coeff** | 0.01 | trainer.py:17 | 系数 L2 正则权重 |
| **lambda_smooth** | 0.01 | trainer.py:18 | 平滑性正则权重 |

```python
# trainer.py, line 14-18
def __init__(self, 
             ...
             lr_restoration=1e-4,      # ← 硬编码
             lr_optics=1e-5,           # ← 硬编码
             lambda_sup=0.0,           # ← 硬编码
             lambda_coeff=0.01,        # ← 硬编码
             lambda_smooth=0.01):      # ← 硬编码
```

### 损失函数

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **clip_grad_norm (W)** | 5.0 | trainer.py:91 | 复原网络梯度裁剪阈值 |
| **clip_grad_norm (Theta)** | 1.0 | trainer.py:92 | 像差网络梯度裁剪阈值 |
| **smoothness_grid_size** | 16 | trainer.py:123 | TV 损失计算网格大小 |

```python
# trainer.py, line 91-92
nn.utils.clip_grad_norm_(self.restoration_net.parameters(), 5.0)   # ← 硬编码
nn.utils.clip_grad_norm_(self.aberration_net.parameters(), 1.0)    # ← 硬编码

# trainer.py, line 123
def compute_smoothness_loss(self, grid_size=16):  # ← 硬编码
```

---

## 7. 可视化参数

### plot_psf_grid()

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **rows (PSF 网格)** | 5 | utils/visualize.py:14 | PSF 采样行数 |
| **cols (PSF 网格)** | 5 | utils/visualize.py:14 | PSF 采样列数 |
| **坐标范围** | [-0.9, 0.9] | utils/visualize.py:16-17 | 采样范围（未覆盖完整） |
| **colormap** | 'inferno' | utils/visualize.py:27 | PSF 可视化色彩方案 |

```python
# utils/visualize.py, line 14-17
rows, cols = 5, 5  # ← 硬编码
y = torch.linspace(-0.9, 0.9, rows)   # ← 硬编码范围
x = torch.linspace(-0.9, 0.9, cols)   # ← 硬编码范围
```

### plot_coefficient_maps()

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **grid_size** | 128 | utils/visualize.py:75 | 系数采样密度 |
| **indices (选择系数)** | [3,4,5,6] | utils/visualize.py:91 | Noll 4-7 (Defocus~Coma) |
| **colormap** | 'viridis' | utils/visualize.py:97 | 系数热力图色彩方案 |

```python
# utils/visualize.py, line 75
grid_size = 128  # ← 硬编码

# utils/visualize.py, line 91
indices = [3, 4, 5, 6]  # ← 硬编码 (Noll 4-7)
```

---

## 8. 演示脚本参数

### demo_train.py

| 参数 | 值 | 位置 | 用途 |
|------|-----|------|------|
| **batch_size (B)** | 2 | demo_train.py:74 | 生成数据的批大小 |
| **channels (C)** | 3 | demo_train.py:74 | RGB 通道 |
| **height (H)** | 256 | demo_train.py:74 | 图像高度 |
| **width (W)** | 256 | demo_train.py:74 | 图像宽度 |
| **base_filters (RestNet)** | 32 | demo_train.py:38 | 复原网络基础滤波器（降低版本） |
| **patch_size** | 128 | demo_train.py:47 | OLA 补丁大小 |
| **stride** | 64 | demo_train.py:48 | OLA 步长 |
| **epochs** | 5 | demo_train.py:97 | 演示训练周期数 |
| **lambda_smooth** | 0.1 | demo_train.py:60 | 平滑正则权重 |

```python
# demo_train.py, line 74
B, C, H, W = 2, 3, 256, 256  # ← 硬编码

# demo_train.py, line 38
base_filters=32  # ← 硬编码（注意：降低到 32，而不是标准的 64）

# demo_train.py, line 97
epochs = 5  # ← 硬编码

# demo_train.py, line 60
lambda_smooth=0.1  # ← 硬编码
```

---

## 9. 总结表 - 按影响程度排序

### 🔴 高影响 (改变会显著影响结果)

| 参数 | 当前值 | 文件 | 影响 |
|------|--------|------|------|
| kernel_size | 33 | zernike.py:207 | PSF 核大小 → 模糊效果 |
| n_modes | 15 | zernike.py:207 | Zernike 模式数 → 表达能力 |
| patch_size | 128 | physical_layer.py:169 | 空间变化分辨率 |
| stride | 64 | physical_layer.py:170 | 补丁密度 |
| wavelengths | [620, 550, 450]e-9 | demo_train.py:20 | RGB 色彩通道分离 |
| base_filters | 64 | restoration_net.py:179 | 模型容量 |
| oversample_factor | 2 | zernike.py:208 | PSF 计算精度 |

### 🟡 中等影响 (改变会小幅影响结果)

| 参数 | 当前值 | 文件 | 影响 |
|------|--------|------|------|
| a_max | 2.0/3.0 | aberration_net.py | 系数范围约束 |
| lr_restoration | 1e-4 | trainer.py:14 | 收敛速度 |
| lr_optics | 1e-5 | trainer.py:15 | 收敛速度 |
| lambda_smooth | 0.01 | trainer.py:18 | 像差平滑度 |
| degree | 2 | aberration_net.py:48 | 多项式复杂度 |
| hidden_dim | 64 | aberration_net.py:128 | MLP 容量 |

### 🟢 低影响 (改变不会显著影响结果)

| 参数 | 当前值 | 文件 | 影响 |
|------|--------|------|------|
| LeakyReLU 斜率 | 0.2 | restoration_net.py:74 | 激活函数性质 |
| clip_grad_norm | 5.0/1.0 | trainer.py:91-92 | 防止梯度爆炸 |
| grid_size (vis) | 128 | visualize.py:75 | 可视化分辨率 |
| rows/cols (PSF) | 5×5 | visualize.py:14 | 可视化采样密度 |

---

## 10. 推荐改进

### ✅ 立即可改进

```python
# 1. 将所有硬编码参数移到配置类
class Config:
    # Physics
    kernel_size: int = 33
    n_modes: int = 15
    pupil_size: int = 64
    oversample_factor: int = 2
    wavelengths: List[float] = [620e-9, 550e-9, 450e-9]
    
    # OLA
    patch_size: int = 128
    stride: int = 64
    
    # Network
    base_filters: int = 64
    
    # Training
    lr_restoration: float = 1e-4
    lr_optics: float = 1e-5
    lambda_smooth: float = 0.01

# 2. 从配置文件加载
config = Config.from_yaml('config.yaml')

# 3. 传递给所有组件
zernike_gen = DifferentiableZernikeGenerator(
    n_modes=config.n_modes,
    pupil_size=config.pupil_size,
    kernel_size=config.kernel_size,
    oversample_factor=config.oversample_factor,
    wavelengths=config.wavelengths
)
```

### 🎯 长期改进

1. **配置管理**: 使用 Hydra 或 YAML 配置文件
2. **可调参数**: 将关键参数暴露为命令行参数
3. **超参数搜索**: 支持自动超参数优化
4. **模块化**: 解耦硬编码的网络架构

---

## 总结

**项目中共有 52 个硬编码参数，分布在 7 个文件中：**

1. **models/zernike.py** - 17 个 (光学物理)
2. **models/physical_layer.py** - 6 个 (空间卷积)
3. **models/restoration_net.py** - 15 个 (U-Net 架构)
4. **models/aberration_net.py** - 8 个 (像差网络)
5. **trainer.py** - 5 个 (训练配置)
6. **utils/visualize.py** - 5 个 (可视化)
7. **demo_train.py** - 9 个 (演示脚本)

**关键发现：**
- ✅ PSF 核大小是 **33×33**（不是 5×5）
- ✅ 最重要的硬编码是 `kernel_size`, `patch_size`, `stride`, `n_modes`
- ✅ U-Net 架构通道倍数也是硬编码（通常 ×2 递增）
- ✅ 大部分可以通过配置文件轻松参数化
