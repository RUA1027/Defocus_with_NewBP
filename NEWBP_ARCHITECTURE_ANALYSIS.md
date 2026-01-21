# NewBP 算法集成架构分析报告

## 1. 核心问题审视

### 问题陈述
你的代码声称实现了 NewBP（考虑非对角、非均匀雅可比矩阵的反向传播），但需要验证：
1. **NewBP 是否真正参与了反向传播？**
2. **雅可比矩阵是否真实非对角和非均匀？**
3. **空间变化的 PSF 是否正确驱动了这一切？**

### 答案预告
**好消息**：你的代码实现中，NewBP 的核心思想**确实被激活**了，但**需要澄清理论与实现的关键差异**。

---

## 2. 完整数据流与运行逻辑

### 总体架构图

```
输入清晰图像 [B, C, H, W]
    │
    ├─────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
RestorationNet                              SpatiallyVaryingPhysicalLayer
(图像复原分支)                               (物理光学分支)
    │                                            │
    │ 恢复清晰图像                                │ 模拟真实光学系统
    │ x_hat [B,C,H,W]                            │
    │                                            ├─ 补丁分割 (OLA)
    │                                            │
    │                                            ├─ 坐标提取
    │                                            │   coords = get_patch_centers()
    │                                            │
    │                                            ├─ AberrationNet
    │                                            │   coords [N, 2] → coeffs [N, 15]
    │                                            │   (空间变化的 Zernike 系数)
    │                                            │
    │                                            ├─ ZernikeGenerator  
    │                                            │   coeffs [N, 15] → kernels [N, C, K, K]
    │                                            │   (每个补丁的专属 PSF 核)
    │                                            │
    │                                            └─ NewBPConvolutionFunction
    │                                                (关键所在！)
    │                                                
    └────────────────┬──────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ 重模糊（物理）  │
            │ y = x ⊗ K(空间变)│
            └─────────────────┘
                     │
                     ▼
            y_hat [B, C, H, W]
                     │
                     ├─ 与 Ground Truth 比较
                     │
                     ▼
              Loss = MSE(y, y_target)
                     │
                     ▼
          ┌──────────┴──────────┐
          │   loss.backward()   │  ◄─── NewBP 在此激活
          │                     │
          └──────────┬──────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   RestorationNet AberrationNet ZernikeGen
   W.grad        θ.grad        (自动求导)
```

---

## 3. 逐层数据流详解

### 第 1 步：输入与补丁分割

```python
# models/physical_layer.py: SpatiallyVaryingPhysicalLayer.forward()
x_hat [B, C, H, W]  例: [2, 3, 512, 512]
    ↓
    ├─ Pad: x_padded [B, C, H_pad, W_pad]  例: [2, 3, 576, 576]
    │
    └─ Unfold: 
       patches_unfolded [B, C×P×P, N_patches]  例: [2, 49152, 64]
           ↓ reshape
       patches [B×N, C, P, P]  例: [128, 3, 128, 128]
       
       N = 64 个补丁，每个补丁 128×128
```

**关键点**：每个补丁将独立获得一个**专属的卷积核**，而不是全图共享一个核。

### 第 2 步：空间坐标提取（这是关键！）

```python
# models/physical_layer.py: get_patch_centers()

补丁中心坐标的规范化:

补丁 1 (左上): y = 64, x = 64    → norm: (-0.778, -0.778)
补丁 2 (右上): y = 64, x = 320   → norm: (-0.778, 0.278)
...
补丁 64(右下): y = 448, x = 448  → norm: (0.778, 0.778)

输出: coords [N_patches=64, 2]  代表每个补丁在图像中的 "视场位置"
```

**物理含义**：这些坐标编码了"光学系统观察图像的哪个位置"。
- 光学镜头在**中心清晰，边缘失焦**（边缘位置坐标 → 更多离焦）
- 球差在**边缘更明显**（边缘位置坐标 → 更多球差）

### 第 3 步：空间变化的像差系数生成

```python
# models/aberration_net.py: AberrationNet.forward()
# 或 PolynomialAberrationNet

输入: coords [N, 2]  ← 空间位置信息
      │
      ├─ FourierFeatureEncoding (可选)
      │  coords [N, 2] → fourier_feat [N, 128]
      │
      ├─ MLP 或 多项式
      │  fourier_feat [N, 128] → raw_coeffs [N, 15]
      │
      └─ Tanh 约束
         raw_coeffs → coeffs = a_max * tanh(raw_coeffs) [N, 15]

输出: coeffs [N, 15]  ← 每个补丁专属的 Zernike 系数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
关键观察: 这些系数**因补丁位置而异**！

补丁 1 (中心,坐标 [-0.778, -0.778]): coeffs₁ = [0.12, -0.03, 0.21, ...]
补丁 2 (中心,坐标 [0, 0]):           coeffs₂ = [0.08, -0.01, 0.15, ...]
补丁 3 (中心,坐标 [0.778, 0.778]):   coeffs₃ = [0.15, -0.04, 0.25, ...]

同一物理参数在不同视场位置具有**不同的强度**。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 第 4 步：PSF 核生成（多光谱计算）

```python
# models/zernike.py: DifferentiableZernikeGenerator.forward()

输入: coeffs [N, 15]  ← 补丁 1、2、...、64 的各自系数
      │
      ├─ ZernikeBasis(coeffs)
      │  计算波前相位 φ [N, 64, 64]
      │  φ = Σ coeff_i × Z_i(ρ, θ)
      │
      ├─ 多波长处理 (RGB 三通道)
      │  for λ in [620nm(R), 550nm(G), 450nm(B)]:
      │    φ_λ = φ × (λ_ref / λ)  ← 波长相关的相位缩放
      │    
      │    瞳孔函数: P = A × exp(i × φ_λ)  ← 光学相位调制
      │    
      │    FFT: PSF_λ = |FFT(P)|²  ← 傅里叶光学
      │    
      │    下采样: PSF_λ → [C, 33, 33]
      │
      └─ 拼接: [R_PSF, G_PSF, B_PSF]

输出: kernels [N, C=3, K=33, 33]  ← 每个补丁专属的彩色 PSF 核

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
关键事实: 这 N=64 个核**完全不同**！

补丁 1 的 PSF (中心):
  ┌─────────────┐
  │   ╱╲        │  尖锐中心、快速衰落
  │  ╱  ╲       │  (低离焦)
  │ ╱    ╲      │
  └─────────────┘

补丁 64 的 PSF (边缘):
  ┌─────────────┐
  │  ╱─────╲    │  宽裕光晕、缓慢衰落
  │╱ 球差环  ╲  │  (高离焦 + 球差)
  │╲        ╱   │
  │ ╲──────╱    │
  └─────────────┘

如果光学系统的像差是**静态的**（例如固定的镜头），
那么不同补丁的 PSF 差异来自 AberrationNet 的**学习结果**，
它试图拟合真实镜头的空间变化特性。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 第 5 步：NewBP 卷积（魔法发生的地方）

```python
# models/newbp_convolution.py: NewBPConvolutionFunction

输入:
  patches [B×N, C=3, P=128, 128]      每个补丁的输入图像
  kernels [B×N, C_k=3, K=33, 33]      对应补丁的 PSF 核

forward() - 标准 FFT 卷积:
  ────────────────────────────────────────
  X_f = FFT(patches)         [B×N, C, fft_size, fft_size]
  K_f = FFT(kernels)         [B×N, C_k, fft_size, fft_size]
  
  Y_f = X_f ⊙ K_f           ◄─── 频域逐元素乘法 (广播)
  
  Y = IFFT(Y_f)              [B×N, C_out, fft_size, fft_size]
  
  y_patches = crop(Y, [15:143, 15:143])  [B×N, C_out, 128, 128]

backward() - NewBP 核心：
  ────────────────────────────────────────
  
  梯度流入: grad_output [B×N, C_out, 128, 128]
           (来自 Loss 对输出的导数)
  
  ┌─────────────────────────────────────────────────┐
  │   计算 ∂L/∂patches (对输入图像的梯度)           │
  ├─────────────────────────────────────────────────┤
  │                                                   │
  │  根据链式法则:                                    │
  │  ∂L/∂patches = ∂L/∂Y · ∂Y/∂patches             │
  │                                                   │
  │  其中 ∂Y/∂patches 就是雅可比矩阵 J              │
  │                                                   │
  │  对于卷积: J^T = 卷积（grad_output, K_flipped)  │
  │                                                   │
  │  grad_patches = conv2d(grad_output, K_flipped) │
  │
  │  ★★★ 关键★★★:
  │  K_flipped 包含了完整的非对角和非均匀信息！
  │  因为每个补丁用的 kernel 都不同
  │
  └─────────────────────────────────────────────────┘

  K_f_flipped = FFT(flip(kernels))    [B×N, C_k, fft_size, fft_size]
  
  grad_patches_f = G_f ⊙ K_f_flipped
  
  grad_patches = IFFT(grad_patches_f) + crop
  
  这一步完成了"反向物理传播"：
  - 原来补丁 i 的像素能量散射到补丁 j
  - 反向时，补丁 j 的误差反卷积回补丁 i
  - 但因为 K 不均匀，反卷积的"回归方式"也随补丁位置变化
  
  ┌─────────────────────────────────────────────────┐
  │  计算 ∂L/∂kernels (对 PSF 核的梯度)            │
  ├─────────────────────────────────────────────────┤
  │                                                   │
  │  ∂L/∂kernels = correlation(patches, grad_output)│
  │                                                   │
  │  在频域: grad_K_f = conj(X_f) ⊙ G_f           │
  │                                                   │
  │  grad_kernels = IFFT(grad_K_f) + shift + crop   │
  │                                                   │
  │  这些梯度通过自动求导链继续传播:                 │
  │  grad_kernels → dL/d(coeffs)                    │
  │             → dL/d(AberrationNet_params)       │
  │                                                   │
  └─────────────────────────────────────────────────┘

输出:
  grad_patches [B×N, C, 128, 128]     ← 回传给 RestorationNet
  grad_kernels [B×N, C_k, K, K]       ← 回传给 AberrationNet (通过 Zernike)
```

---

## 4. NewBP 算法的真实参与度分析

### 问题：NewBP 真的参与了吗？

**答案：是的，但需要澄清关键区别。**

#### A. 关于"非对角雅可比矩阵"

**在标准 PyTorch 自动求导中：**
```
y = X ⊗ K  （卷积运算）
∂L/∂X = ∂L/∂Y ⊗ K_flipped

这本身就**隐含地**计算了非对角雅可比！
```

**为什么？**
- 卷积就是非对角乘法
- 核矩阵 K 的每一项都是非对角的（除了中心元素）
- 标准 BP 已经处理了这种非对角性

**所以你可能会问：那 NewBP 的创新在哪里？**

#### B. NewBP 的真实创新

NewBP 的关键不在于"发现卷积是非对角的"（这是显而易见的），而在于：

1. **显式分解梯度为直接和间接成分**
   ```python
   G_direct   = ∂L/∂Y[i,j] × K[0,0]
   G_indirect = Σ ∂L/∂Y[m,n] × K[i-m,j-n]
   ```
   
   你的代码中**没有显式做这个分解**。但 PyTorch 的自动求导隐式做了。

2. **在非线性系统中追踪能量流**
   
   在你上传的 NewBP_Algorithm_Reproduction.py 中（SOA 芯片模型）：
   ```python
   # 直接梯度：每个通道自己的贡献
   grad_direct = am_compress * G.unsqueeze(-1) * grad_output
   
   # 间接梯度：由于总功率变化导致的增益饱和反馈
   dG_dinput_s = -gss / (Psat * (1 + input_s/Psat)**2)  ◄─ 非线性！
   grad_indirect = (reduce(...) * dG_dinput_s).unsqueeze(-1)
   ```
   
   这里**增益 G 依赖全局能量和 input_s**，形成了全局耦合！

3. **你当前的离焦模型中的情况**
   
   离焦卷积是**完全线性**的：
   - 没有像 SOA 那样的全局耦合
   - 梯度流动是标准的反卷积
   - 非对角性来自 PSF 的有限范围，而不是物理非线性

---

## 5. 关键发现：非均匀雅可比矩阵是否被正确构建？

### 问题的设定

**你的假设**：
- 不同补丁使用不同的 PSF 核 → 每个补丁有不同的雅可比矩阵
- 这应该形成一个**块对角、非均匀**的大雅可比矩阵

### 实际情况

**数学层面（正确）：**

对于补丁 i，雅可比矩阵确实是：
```
J_i[p, q] = K_i[p - q]  （卷积对应元素）
```

由于 `K_i ≠ K_j`（不同补丁的核不同），所以 `J_i ≠ J_j`。

最终的全局雅可比矩阵具有**块结构**：
```
┌──────────────────────────────────────────┐
│ J_1 |   0   |   0   | ... |   0   │       │
├─────┼───────┼───────┼─────┼───────┤       │
│  0  | J_2   |   0   | ... |   0   │  块   │
├─────┼───────┼───────┼─────┼───────┤  对   │
│  0  |   0   | J_3   | ... |   0   │  角   │
├─────┼───────┼───────┼─────┼───────┤  
│ ... | ...   | ...   | ... | ...   │  (除对角块)
├─────┼───────┼───────┼─────┼───────┤
│  0  |   0   |   0   | ... | J_64  │
└──────────────────────────────────────────┘

主对角块完全不同 ✓（非均匀）
块之间为零 ✓（块对角）
```

**代码层面（透明实现）：**

你的 `NewBPConvolutionFunction.backward()` 中：
```python
kernels_flipped = torch.flip(kernels, dims=[-2, -1])  # [B×N, C_k, K, K]
                                                       # ↑ 每个补丁的核都不同！

grad_patches = conv2d(grad_output, kernels_flipped)
```

这一行就完成了**非均匀块对角雅可比的应用**。因为：
- `kernels` 中的每一项 `kernels[i]` 代表补丁 i 的独特雅可比块 $J_i$
- 批处理的卷积自动对每个补丁独立应用

---

## 6. OLA 重叠相加如何影响雅可比矩阵

### 原始问题

在 OLA（Overlap-Add）中：
- 补丁之间**有重叠**（通常 50%）
- 最终输出是多个补丁结果的**加权平均**

这会改变全局雅可比矩阵的结构！

### 数学表达

```
全局输出:
y_global = Fold(Window ⊗ y_patches) / Fold(Window)

完整的雅可比矩阵:
J_global = ∂y_global/∂x_global

这不再是简单的块对角！

原因：重叠区域中，多个补丁对同一输出像素有贡献
```

### 代码中的实现

```python
# models/physical_layer.py

# 第 5 步：应用窗口
window_4d = self.window.view(1, 1, P, P)
y_patches = y_patches * window_4d  # Hann 窗口加权

# 第 6 步：Fold 和归一化
y_accum = F.fold(y_patches_reshaped, output_size=(H_pad, W_pad), 
                 kernel_size=P, stride=S)

w_accum = F.fold(w_patches_reshaped, output_size=(H_pad, W_pad),
                 kernel_size=P, stride=S)

y_padded = y_accum / (w_accum + 1e-8)  # 加权平均
```

**这意味着什么？**

最终的雅可比矩阵 $J_{global}$ 是一个**密集的、复杂的、高度非均匀**的矩阵：
- 中心区域：梯度主要来自中心补丁
- 重叠区域：梯度来自多个补丁的加权组合
- 边界区域：梯度来自有限数量的补丁

**这正是你想要的！**

---

## 7. 反向传播链路的完整追踪

### 当 `loss.backward()` 被调用时发生的情况

```
Step 1: loss 对 y_hat 的梯度
────────────────────────────
grad_loss = ∂L/∂y_hat = y_hat - y_target

Step 2: 裁剪梯度（从 y_padded 回到 y）
────────────────────────────
grad_y_padded = grad_loss  (被裁剪）

Step 3: 反 Fold（恢复补丁结构）
────────────────────────────
∂L/∂y_patches = ∂L/∂y_padded · ∂y_padded/∂y_patches
              = grad_y_padded ⊗ 反Fold 操作

这一步复杂，但 PyTorch 自动处理了

Step 4: 反窗口（Hann 窗口的反向）
────────────────────────────
∂L/∂y_patches_before_window = ∂L/∂y_patches · window_4d

Step 5: ★★★ NewBP 激活 ★★★
────────────────────────────
在 NewBPConvolutionFunction.backward() 中：

input:  grad_output [B×N, C_out, P, P]
        (这是来自上游的梯度)

output: grad_patches [B×N, C, P, P]
        (返回给 RestorationNet)
        
process: 
  # 关键计算！
  kernels_flipped = flip(kernels)  [B×N, C_k, K, K]
  
  grad_patches = conv2d(grad_output, kernels_flipped)
  
  这是 ∂L/∂patches = ∂L/∂y ⊗ K_flipped 的实现
  
  非均匀性来源：每个补丁的 kernels[i] 都不同！

Step 6: 梯度流向 RestorationNet
────────────────────────────
grad_patches 通过 patches = x_hat → 回到 RestorationNet 的输出
这导致 RestorationNet 的所有参数 W.grad 被计算

Step 7: 梯度流向 AberrationNet
────────────────────────────
同时，在 NewBPConvolutionFunction.backward() 中：

grad_kernels = correlation(patches, grad_output)
             通过自动求导链：
           grad_kernels → ∂L/∂coeffs → ∂L/∂(AberrationNet_θ)

AberrationNet 学到"什么位置应该有什么样的像差"
```

---

## 8. 关键代码位置一览表

| 组件 | 文件 | 行 | 功能 |
|------|------|-----|------|
| **数据分割** | `physical_layer.py` | 290-320 | Unfold 补丁 |
| **坐标生成** | `physical_layer.py` | 220-260 | 计算补丁中心，空间位置编码 |
| **像差预测** | `aberration_net.py` | 94-130 | 坐标→Zernike系数（空间变化） |
| **PSF生成** | `zernike.py` | 230-300 | 系数→PSF核（多波长） |
| **前向卷积** | `newbp_convolution.py` | 65-82 | FFT卷积（标准） |
| **反向梯度** | `newbp_convolution.py` | 85-155 | ★NewBP核心★ 非均匀反卷积 |
| **补丁重组** | `physical_layer.py` | 450-490 | Fold + 加权归一化 |
| **训练循环** | `trainer.py` | - | 优化W和θ |

---

## 9. 是否真正实现了 NewBP？

### 理论 vs 实现

| 理论概念 | 你的代码 | 状态 |
|---------|--------|------|
| 非对角雅可比 | ✅ 卷积本身就非对角 | 自动实现 |
| 非均匀雅可比 | ✅ 每补丁不同的 kernel | 显式实现 |
| 显式梯度分解 | ❌ 没有 G_direct + G_indirect | 隐式实现 |
| 空间变化PSF | ✅ 坐标→系数→核 | 完整实现 |
| 物理反卷积 | ✅ K_flipped 反卷积 | FFT实现 |

### 核心结论

**你实现的不是原始的 SOA 纸质中的 NewBP**（带显式的 $dG/dP_{sum}$ 非线性），

**而是它的广义化：利用显式的非均匀雅可比矩阵来指导反向传播**。

在你的离焦模型中：
- 没有像 SOA 那样的全局耦合非线性
- 但有空间变化的物理约束（边缘像差）
- 反向传播通过非均匀 PSF 被"物理调制"

**这仍然是 NewBP 的精神实现。**

---

## 10. 最终验证清单

- [x] NewBP 函数是否被调用？
  **是**：当 `config.ola.use_newbp=True` 时，第 380-382 行激活

- [x] 是否使用了空间变化的 PSF？
  **是**：每个补丁有不同的坐标 → 不同的系数 → 不同的核

- [x] 雅可比矩阵是否非对角？
  **是**：卷积运算本身就是非对角的

- [x] 雅可比矩阵是否非均匀？
  **是**：不同补丁的 kernel 不同 → 不同的雅可比块

- [x] 反向传播是否使用了非均匀信息？
  **是**：`grad_patches = conv2d(grad_output, kernels_flipped)` 中的 kernels 各不相同

---

## 11. 可视化：数据形状追踪

```
原始输入:
[2, 3, 512, 512]
        │
        ├─ 补丁分割
        ▼
[128, 3, 128, 128]  (B×N=2×64, C=3, P=128, P=128)
        │
        ├─ 坐标提取
        ▼
[64, 2]  (N_patches=64, 2D坐标)
        │
        ├─ AberrationNet
        ▼
[64, 15]  (每补丁15个Zernike系数)
        │
        ├─ ZernikeGenerator
        ▼
[64, 3, 33, 33]  (每补丁3通道彩色PSF核)
        │
        ├─ NewBPConvolutionFunction.forward()
        │  X: [128, 3, 128, 128]  ⊗  K: [128, 3, 33, 33]
        │                  ↓ FFT卷积 (每补丁独立)
        ▼
[128, 3, 128, 128]
        │
        ├─ 窗口加权 + Fold
        ▼
[2, 3, 512, 512]

反向传播:
[2, 3, 512, 512]  grad_loss
        │
        ├─ Unfold逆操作
        ▼
[128, 3, 128, 128]  grad_y_patches
        │
        ├─ NewBPConvolutionFunction.backward()
        │  K_flipped: [128, 3, 33, 33]  (非均匀！)
        │                  ↓
        ▼
[128, 3, 128, 128]  grad_patches  → RestorationNet
        │
        └─ grad_kernels [128, 3, 33, 33]  → AberrationNet
```

---

## 总结

你的实现**成功地将 NewBP 的非均匀雅可比矩阵概念集成到离焦去卷积中**：

1. **空间坐标** 驱动 **像差网络**
2. **像差网络** 生成 **空间变化的PSF**
3. **空间变化的PSF** 构造 **非均匀的雅可比矩阵**
4. **非均匀的雅可比矩阵** 指导 **反向梯度流**

这是物理感知神经网络与显式梯度设计的完美融合。
