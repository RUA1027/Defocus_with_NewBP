# Implementation Plan - Defocus_with_NewBP Enhancements

## 已完成的修改概述

### 1. 数据预处理阶段 (Data Preprocessing)

**修改文件**: [data/preprocess_dpdd.py](data/preprocess_dpdd.py)

- ✅ 现在使用 `train_c`, `val_c`, `test_c` 三个独立文件夹作为数据源
- ✅ 跳过 resize 和图像重存步骤，直接使用原始图像 (1680×1120)
- ✅ 转变为数据验证脚本，检查数据集完整性

### 2. 数据加载模块 (Data Loading)

**修改文件**: [utils/dpdd_dataset.py](utils/dpdd_dataset.py)

- ✅ `DPDDDataset` 类现在直接从原始数据文件夹加载
- ✅ 引入 **虚拟长度机制** (`repeat_factor=100`)，每张图在一个 Epoch 内被随机裁剪 100 次
- ✅ 训练集使用随机裁剪 (512×512)
- ✅ 验证集使用固定中心裁剪 (1024×1024)
- ✅ 测试集使用全分辨率 (1680×1120)
- ✅ 新增 `DPDDTestDataset` 类用于测试集加载

### 3. 配置文件 (Configuration)

**修改文件**: [config/default.yaml](config/default.yaml)

- ✅ 数据路径更新为 `./data/dd_dp_dataset_png`
- ✅ 三阶段训练 Epoch 数调整为: Stage1=50, Stage2=200, Stage3=50 (总计 300 epochs)
- ✅ 添加 `repeat_factor: 100` 配置
- ✅ 添加 TensorBoard 配置
- ✅ 添加熔断机制配置
- ✅ 添加 checkpoint 策略配置
- ✅ 更新损失权重配置，添加详细注释

### 4. 训练器增强 (Trainer Enhancements)

**修改文件**: [trainer.py](trainer.py)

- ✅ 添加 **TensorBoard 日志支持**，记录训练/验证指标、梯度分布、图像
- ✅ 添加 **熔断机制** (Circuit Breaker)，在阶段切换时检查验证指标
- ✅ Stage 3 **学习率自动减半**
- ✅ 添加 `update_best_metrics()` 方法跟踪各阶段最佳指标
- ✅ 增强 `save_checkpoint()` 支持更多元数据
- ✅ 添加 `load_checkpoint()` 方法支持训练恢复

### 5. DataLoader 构建 (Model Builder)

**修改文件**: [utils/model_builder.py](utils/model_builder.py)

- ✅ 更新 `build_dataloader_from_config()` 支持新的数据集配置
- ✅ 添加 `repeat_factor` 参数传递
- ✅ 添加 `build_test_dataloader_from_config()` 用于测试集
- ✅ 集成 TensorBoard 和熔断机制配置到 trainer 构建

### 6. 评估模块增强 (Metrics Enhancement)

**修改文件**: [utils/metrics.py](utils/metrics.py)

- ✅ 添加 `evaluate_stage1()` 方法，专门用于 Stage 1 评估（重模糊一致性）
- ✅ 添加 `evaluate_full_resolution()` 方法，用于测试集全分辨率评估
- ✅ 返回每张图像的详细结果

### 7. 训练脚本 (Training Script)

**修改文件**: [train.py](train.py)

- ✅ 集成阶段特定的验证逻辑
- ✅ 阶段切换时执行熔断机制检查
- ✅ 各阶段独立保存最佳模型 (`best_stage1_physics.pt`, `best_stage2_restoration.pt`, `best_stage3_joint.pt`)
- ✅ 每 20 个 Epoch 定期存档
- ✅ 支持 `--resume` 参数恢复训练
- ✅ 打印详细的训练计划和阶段信息

### 8. 测试脚本 (Testing Script)

**新建文件**: [test.py](test.py)

- ✅ 全分辨率测试集评估
- ✅ 完整指标: PSNR, SSIM, LPIPS, Re-blur Error
- ✅ 可选保存对比图像 (`--save-images`)
- ✅ 可选保存复原结果 (`--save-restored`)
- ✅ 生成 JSON 和 CSV 格式结果报告
- ✅ 打印最佳/最差样本列表

---

## 使用说明

### 1. 验证数据集
```bash
python data/preprocess_dpdd.py
```

### 2. 开始训练
```bash
python train.py --config config/default.yaml
```

### 3. 恢复训练
```bash
python train.py --config config/default.yaml --resume results/checkpoint_epoch100.pt
```

### 4. 查看 TensorBoard
```bash
tensorboard --logdir results/
```

### 5. 测试评估
```bash
python test.py --checkpoint results/best_stage3_joint.pt --save-images
```

---

## 三阶段训练策略

| 阶段 | Epochs | 目标 | 验证判据 |
|------|--------|------|----------|
| Stage 1 | 1-50 | 训练物理层，学习像差分布 | Re-blur MSE |
| Stage 2 | 51-250 | 固定物理层，训练复原网络 | PSNR & SSIM |
| Stage 3 | 251-300 | 联合微调，学习率减半 | Combined (PSNR + 物理约束) |

---

## Stage Status

- **Stage 1: Data & Config Refactor** ✅ **Completed**
- **Stage 2: Trainer & Logic Enhancements** ✅ **Completed**  
- **Stage 3: Verification** ✅ **Completed**
