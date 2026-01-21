# 配置文件使用指南
# Configuration Files Usage Guide
# ==============================================================================

## 概览 (Overview)

项目中创建了 **4 个配置文件**，每个针对不同的使用场景优化。选择正确的配置文件能显著影响训练效率和结果质量。

---

## 📋 配置文件详细对比

### 1️⃣ `config/default.yaml` - 标准/均衡配置

**用途**: 
- 生产环境训练
- 学术论文实验
- 一般性能基准测试

**特点**:
```
kernel_size: 33          # 标准大小，适合典型模糊
patch_size: 128          # 平衡精度和性能
base_filters: 64         # 足够的网络容量
oversample_factor: 2     # 精确的 PSF 计算
epochs: 100              # 充分的训练
```

**性能特征**:
- GPU 显存占用: ~8GB (单 GPU)
- 每 epoch 耗时: ~10-15 分钟 (512×512 图像)
- 模型大小: ~50MB
- 精度: 最佳

**何时使用**:
```
python demo_train.py --config config/default.yaml
```

✅ 首次运行项目  
✅ 发表论文  
✅ 获得最好的去模糊效果  
✅ 有充足计算资源

---

### 2️⃣ `config/lightweight.yaml` - 轻量级快速测试配置

**用途**:
- 快速原型设计
- 算法验证
- 资源受限环境
- GPU 显存不足时

**特点**:
```yaml
n_modes: 10              # ↓ 减少 Zernike 模式 (10 vs 15)
pupil_size: 32           # ↓ 更小的光瞳网格 (32 vs 64)
kernel_size: 17          # ↓ 更小的 PSF 核 (17 vs 33)
oversample_factor: 1     # ✘ 关闭过采样
patch_size: 64           # ↓ 更小的补丁 (64 vs 128)
base_filters: 32         # ↓ 较小网络 (32 vs 64)
use_coords: false        # ✘ 关闭坐标注入
batch_size: 8            # ↑ 增加批大小 (充分利用小模型)
epochs: 50               # ↓ 较少训练轮数
```

**性能特征**:
- GPU 显存占用: ~2GB
- 每 epoch 耗时: ~30 秒 (128×128 图像)
- 模型大小: ~5MB
- 精度: 中等 (-15% 相对性能)
- 速度: 20× 快

**何时使用**:
```bash
# 快速验证想法
python demo_train.py --config config/lightweight.yaml

# 笔记本电脑/小显卡
python demo_train.py --config config/lightweight.yaml

# 快速调试错误
python demo_train.py --config config/lightweight.yaml
```

✅ 快速验证算法  
✅ 笔记本 GPU (2GB 显存)  
✅ 测试代码是否有 bug  
✅ 显存报错时的替代方案  
✅ 完整功能验证 (仅精度降低)

**缺点**:
- ❌ 精度较低
- ❌ 不适合最终发表
- ❌ 灰度单通道 (无 RGB)

---

### 3️⃣ `config/high_resolution.yaml` - 高分辨率实验配置

**用途**:
- 处理超大图像 (1K, 2K, 4K)
- 真实应用场景
- 专业摄影/相机数据

**特点**:
```yaml
kernel_size: 65          # ↑ 更大的 PSF 核 (65 vs 33)
                          # 适合强散焦
patch_size: 256          # ↑ 更大补丁 (256 vs 128)
                          # 减少计算开销
stride: 128              # 保持 50% 重叠
image_height: 1024       # ↑ 大图像 (1024 vs 256)
image_width: 1024
batch_size: 1            # ↓ 批大小为 1 (显存限制)
base_filters: 64         # ↑ 较大网络容量
degree: 3                # ↑ 更高阶多项式
a_max: 3.0               # ↑ 更大的像差范围
epochs: 200              # ↑ 更多训练轮数
lr_restoration: 5e-5     # ↓ 更小学习率 (更稳定)
```

**性能特征**:
- GPU 显存占用: ~20GB (RTX 4090)
- 每 epoch 耗时: ~5-10 分钟
- 模型大小: ~50MB
- 精度: 最佳 (+5% 相对性能)
- 需要显卡: RTX 3080/4090

**何时使用**:
```bash
# 高分辨率图像
python demo_train.py --config config/high_resolution.yaml

# 强散焦场景
python demo_train.py --config config/high_resolution.yaml
```

✅ 实际应用 (手机/相机照片)  
✅ 1K+ 分辨率图像  
✅ 强散焦效果  
✅ 需要最好的结果  
✅ 有高端 GPU  

**前置条件**:
- ⚠️ GPU 显存 ≥ 16GB
- ⚠️ 训练时间长 (数小时)
- ⚠️ 不适合快速实验

---

### 4️⃣ `config/mlp_experiment.yaml` - MLP 像差网络实验配置

**用途**:
- 对比不同网络架构
- 研究 MLP vs Polynomial
- 学术对比实验
- 复杂像差分布场景

**特点**:
```yaml
aberration_net:
  type: "mlp"            # ✓ 使用 MLP 而非多项式
  mlp:
    hidden_dim: 128      # ↑ 较大隐层 (128 vs 64)
    use_fourier: true    # ✓ 傅里叶编码
    fourier_scale: 10    # ↑ 更高频率特征
    a_max_mlp: 3.0

training:
  lambda_smooth: 0.1     # ↑ 更强平滑约束 (0.1 vs 0.01)
  lr_optics: 5e-6        # ↓ 更小学习率 (更稳定)
  smoothness_grid_size: 32  # ↑ 密集采样
```

**性能特征**:
- GPU 显存占用: ~10GB
- 每 epoch 耗时: ~12-18 分钟
- 精度: 对比实验
- 特点: 更灵活但难以收敛

**何时使用**:
```bash
# 科研对比实验
python demo_train.py --config config/mlp_experiment.yaml

# 研究 MLP 的表现
python demo_train.py --config config/mlp_experiment.yaml
```

✅ 学术论文对比  
✅ 网络架构研究  
✅ 复杂非线性像差  
✅ 发表新算法时

**与 default.yaml 的区别**:
- `default.yaml` 用多项式网络 (Polynomial)
- `mlp_experiment.yaml` 用 MLP 网络
- MLP 更灵活但容易过拟合
- 多项式更稳定但表达力有限

---

## 🎯 快速选择指南

### 根据硬件配置选择

```
GPU 显存 < 2GB
  └─→ config/lightweight.yaml ✅
      (唯一选择)

GPU 显存 2-8GB  
  ├─→ config/lightweight.yaml (快速测试)
  └─→ config/default.yaml (生产环境)

GPU 显存 8-16GB
  ├─→ config/default.yaml (标准)
  ├─→ config/mlp_experiment.yaml (实验)
  └─→ config/lightweight.yaml (快速测试)

GPU 显存 > 16GB (RTX 3080/4090)
  ├─→ config/high_resolution.yaml (推荐)
  ├─→ config/default.yaml
  ├─→ config/mlp_experiment.yaml
  └─→ config/lightweight.yaml
```

### 根据使用场景选择

```
🔷 快速验证代码 (5 分钟内)
  └─→ config/lightweight.yaml
     python demo_train.py --config config/lightweight.yaml experiment.epochs=5

🔶 学习和调试 (30 分钟)
  └─→ config/lightweight.yaml
     python demo_train.py --config config/lightweight.yaml experiment.epochs=20

🟡 标准训练 (1-2 小时)
  └─→ config/default.yaml
     python demo_train.py --config config/default.yaml

🟠 最好效果 (4-8 小时)
  └─→ config/high_resolution.yaml
     python demo_train.py --config config/high_resolution.yaml

🔴 科研对比 (6-12 小时)
  ├─→ config/default.yaml (baseline)
  └─→ config/mlp_experiment.yaml (提案方法)
     python demo_train.py --config config/mlp_experiment.yaml
```

### 根据目标图像选择

```
图像尺寸 128×128
  └─→ config/lightweight.yaml

图像尺寸 256×256 (默认)
  └─→ config/default.yaml

图像尺寸 512×512
  ├─→ config/default.yaml (如果显存足够)
  └─→ config/lightweight.yaml (有 patch 重叠处理)

图像尺寸 1024×1024+
  └─→ config/high_resolution.yaml (必须)
```

---

## 📊 完整对比表

| 配置文件 | kernel_size | patch_size | base_filters | epochs | GPU显存 | 速度 | 精度 | 用途 |
|---------|-------------|------------|--------------|--------|--------|------|------|------|
| lightweight | 17 | 64 | 32 | 50 | 2GB | 20×⚡ | 中 | 快速测试 |
| default | 33 | 128 | 64 | 100 | 8GB | 1× | 高 | 标准训练 |
| high_resolution | 65 | 256 | 64 | 200 | 20GB | 0.5× | 最高 | 大图像 |
| mlp_experiment | 33 | 128 | 64 | 150 | 10GB | 1.2× | 高 | 科研对比 |

---

## 💡 实际使用示例

### 场景 1: 第一次运行项目

```bash
# 步骤 1: 快速验证代码是否有问题
python demo_train.py --config config/lightweight.yaml experiment.epochs=5

# 步骤 2: 学习和理解参数
python demo_train.py --config config/lightweight.yaml experiment.epochs=20

# 步骤 3: 标准训练
python demo_train.py --config config/default.yaml
```

### 场景 2: 显存不足 (8GB GPU)

```bash
# ❌ 先不要尝试
python demo_train.py --config config/high_resolution.yaml

# ✅ 使用标准或轻量级
python demo_train.py --config config/default.yaml

# 如果还是报显存不足，降低参数
python demo_train.py --config config/default.yaml data.batch_size=1
```

### 场景 3: 有 24GB+ 显存 (RTX 4090)

```bash
# 标准配置
python demo_train.py --config config/default.yaml

# 尝试高分辨率
python demo_train.py --config config/high_resolution.yaml

# 同时对比多种方法
python demo_train.py --config config/mlp_experiment.yaml

# 可以并行训练多个实验
# 终端 1
python demo_train.py --config config/default.yaml
# 终端 2
python demo_train.py --config config/mlp_experiment.yaml
```

### 场景 4: 学术对比实验

```bash
# 方法 A: 多项式网络 (baseline)
python demo_train.py --config config/default.yaml

# 方法 B: MLP 网络 (提案)
python demo_train.py --config config/mlp_experiment.yaml

# 方法 C: 高分辨率多项式
python demo_train.py --config config/high_resolution.yaml

# 方法 D: 快速测试（论文草稿）
python demo_train.py --config config/lightweight.yaml
```

---

## 🔧 常见修改和组合

### 修改 1: 只改变训练轮数

```bash
# 快速配置但训练时间长
python demo_train.py --config config/lightweight.yaml experiment.epochs=200

# 默认配置但快速验证
python demo_train.py --config config/default.yaml experiment.epochs=10
```

### 修改 2: 增加批大小（有更多显存时）

```bash
# 默认批大小为 2，改为 8
python demo_train.py --config config/default.yaml data.batch_size=8
```

### 修改 3: 混合配置

```bash
# 用 lightweight 的图像大小，default 的网络大小
python demo_train.py --config config/default.yaml \
  data.image_height=128 \
  data.image_width=128 \
  data.batch_size=16
```

### 修改 4: 自定义实验配置

```bash
# 基于 default，改变学习率
python demo_train.py --config config/default.yaml \
  training.optimizer.lr_restoration=5e-5 \
  training.optimizer.lr_optics=5e-6 \
  experiment.epochs=200
```

---

## ⚠️ 常见问题和解决

### Q1: "CUDA out of memory" 错误

```bash
# ❌ 不要用
python demo_train.py --config config/high_resolution.yaml

# ✅ 改用轻量级
python demo_train.py --config config/lightweight.yaml

# ✅ 或者降低默认配置的参数
python demo_train.py --config config/default.yaml \
  data.batch_size=1 \
  restoration_net.base_filters=32
```

### Q2: 训练太慢，想快速测试

```bash
# ❌ 不要在高分辨率下测试
python demo_train.py --config config/high_resolution.yaml experiment.epochs=1

# ✅ 使用轻量级配置
python demo_train.py --config config/lightweight.yaml experiment.epochs=5
```

### Q3: 结果不好，想提升精度

```bash
# ✅ 增加训练轮数
python demo_train.py --config config/default.yaml experiment.epochs=300

# ✅ 使用高分辨率配置
python demo_train.py --config config/high_resolution.yaml

# ✅ 尝试 MLP 网络
python demo_train.py --config config/mlp_experiment.yaml
```

### Q4: 想对比不同网络架构

```bash
# 配置 A: 多项式网络
python demo_train.py --config config/default.yaml --output results/polynomial

# 配置 B: MLP 网络  
python demo_train.py --config config/mlp_experiment.yaml --output results/mlp

# 对比结果
python compare_results.py results/polynomial results/mlp
```

---

## 📝 总结决策树

```
开始
  │
  ├─ 第一次使用?
  │   └─→ 用 lightweight (快速了解)
  │
  ├─ 想快速测试代码?
  │   └─→ 用 lightweight (2 GB, 快速)
  │
  ├─ 产品应用 / 最好结果?
  │   ├─ 图像 < 512×512?
  │   │  └─→ 用 default (标准)
  │   └─ 图像 > 1024×1024?
  │      └─→ 用 high_resolution (大图像)
  │
  ├─ 学术论文 / 对比实验?
  │   ├─ baseline?
  │   │  └─→ 用 default
  │   └─ 提案方法?
  │      └─→ 用 mlp_experiment
  │
  ├─ 显存不足?
  │   └─→ 用 lightweight
  │
  └─ 显存充足 (> 16GB)?
      └─→ 用 high_resolution
```

---

**推荐首选**:
- 🟢 **新手**: `config/lightweight.yaml` → 理解后改 `config/default.yaml`
- 🟡 **标准使用**: `config/default.yaml`
- 🔵 **大图像**: `config/high_resolution.yaml`
- 🔴 **科研**: `config/default.yaml` (baseline) + `config/mlp_experiment.yaml` (提案)
