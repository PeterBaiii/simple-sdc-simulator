# Silent Data Corruption Detection Based on Metamorphic Attention Bounds

基于注意力机制蜕变关系边界的GPU静默数据损坏（SDC）检测原型实现。

## 项目简介

本项目探索在Transformer注意力层中使用解析推导的边界作为蜕变关系（metamorphic relations）来检测GPU计算中的静默数据损坏。通过在注意力计算的中间张量中注入单比特翻转故障，并检查输出是否违反理论边界，来评估这种检测方法的可行性。

### 核心思想

对于标准点积自注意力机制，我们推导出了关于偏差 εᵢ = qᵢᵀkⱼ* - qᵢᵀAttn(xᵢ) 的理论上下界：

```
√d · γᵢ/(1 + e^γᵢ) ≤ εᵢ ≤ min(qᵢᵀkⱼ* - mean(qᵢᵀkⱼ), √d·τ(γᵢ, n))
```

其中：
- γᵢ = aᵢⱼ* - max_{j≠j*} aᵢⱼ 是softmax的margin
- τ(γᵢ, n) 是基于Lambert-W函数的紧致上界
- d 是注意力头的维度

当注入的故障导致输出违反这些边界时，我们将其标记为潜在的SDC。

## 目录结构

```
.
├── attention_injection.py    # 单层注意力实验（高斯输入）
├── analyzer.py               # 结果分析工具
├── visualizer.py             # 结果可视化工具
├── TinyLM/                   # 最小Transformer实验
│   ├── tinylm_injection.py   # TinyLM故障注入实验
│   ├── analyzer.py           # TinyLM结果分析
│   └── visualizer.py         # TinyLM结果可视化
├── utils/                    # 工具函数
│   ├── debug.py              # 调试输出
│   ├── check_nan.py          # NaN检测
│   ├── bound_fixing.py       # 边界诊断
│   └── return_top2.py        # Top-2分析
└── requirements.txt          # 依赖库
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0（用于Lambert-W函数）
- Matplotlib >= 3.7.0（用于可视化）

## 实验说明

### 实验1：单层注意力 + 高斯输入

**配置**：
- 输入：从高斯分布 N(0, σ²) 采样的合成张量
- 模型：单个多头注意力块（B=2, H=4, L=8, Dh=16）
- 注入位置：q, k, v, scores, weights, out
- 比特范围：0-31（IEEE 754单精度浮点）

**运行实验**：

```bash
python attention_injection.py
```

**实验参数（在脚本中修改）**：

```python
# 模型配置
B, H, L, Dh = 2, 4, 8, 16  # batch, heads, length, head_dim

# 高斯分布标准差
sigmas = [1.0, 1.67]  # σ=1.0（窄范围）和 σ≈1.67（扩展范围）

# 注入配置
injection_locations = ["q", "scores", "weights", "out"]
inj_idx = (0, 0, 0, 0)  # (batch, head, seq_i, seq_j)
bit_range = range(32)   # 所有32个比特位
```

**主要发现**（来自实验）：
- **输入范围影响**：σ=1.0时检测率为9.4%，σ≈1.67时提高到18.5%
- **位置敏感性**：输出张量（out）最容易监测（14.7% → 39.5%）
- **比特位选择性**：
  - 低位尾数（bit 0-20）：几乎无检测（~0%）
  - 高位指数/符号位（bit 23-31）：检测率显著提升
  - 例如：bit 30在out位置的检测率达94-100%

### 实验2：TinyLM Transformer

**配置**：
- 模型：单层Transformer（vocab=38, embd=64, heads=4, block_size=32）
- 数据：GPT生成的字符级语料（字母、数字、标点）
- 训练：400步，学习率1e-4
- 注入：固定位置 (b=0, h=0, i=2, j=7)

**运行实验**：

```bash
cd TinyLM
python tinylm_injection.py
```

**实验参数（在脚本中修改）**：

```python
# 模型配置
vocab_size = 38
n_embd = 64
n_heads = 4
block_size = 32

# 注入配置
injection_locations = ["q", "scores", "weights", "out"]
inj_idx = (0, 0, 2, 7)  # 固定注入位置
n_test_batches = 16     # 测试批次数
bit_range = range(32)
```

**主要发现**（来自实验）：
- **整体检测率**：26.2% (537/2048)
- **按位置检测率**：
  - weights: 33.2%
  - q: 32.8%
  - out: 31.3%
  - scores: 7.6%（最低）
- **比特位模式**：
  - scores：只在高位比特（23-30）有检测（25-31%）
  - 其他位置：大多数比特位约31%，bit 30达94%
- **批次间变化**：检测率在批次间变化大（0%-100%），标准差44-48%

## 结果分析

### 分析工具

```bash
# 分析单层注意力实验结果
python analyzer.py

# 分析TinyLM实验结果
cd TinyLM
python analyzer.py
```

分析器会：
- 扫描所有JSON结果文件
- 按注入位置、比特位、标准差（单层）或批次（TinyLM）汇总
- 生成统计报告

### 可视化

```bash
# 可视化单层注意力结果
python visualizer.py

# 可视化TinyLM结果
cd TinyLM
python visualizer.py
```

生成的图表包括：
- **按比特位检测率**：展示IEEE 754各区域的检测模式
- **按位置检测率**：不同注入位置的对比
- **热图**：比特位 × 注入位置的2D可视化

## 实验输出

### 输出文件

实验会生成以下文件：

1. **完整结果**：`*_results_YYYYMMDD_HHMMSS.json`
   - 包含所有配置和每个比特位的详细统计
   
2. **摘要**：`*_summary_YYYYMMDD_HHMMSS.json`
   - 仅包含汇总统计和检测率

### JSON结构示例

```json
{
  "metadata": {
    "timestamp": "2025-12-19T10:30:00",
    "configuration": {
      "B": 2, "H": 4, "L": 8, "Dh": 16,
      "injection_locations": ["q", "scores", "weights", "out"],
      "sigmas": [1.0, 1.67],
      "total_tests": 4096
    }
  },
  "summary": {
    "by_injection_location": {
      "out": {"detected": 150, "total": 1024, "rate_percent": 14.65}
    },
    "by_sigma": {
      "1.0": {"detected": 96, "total": 2048, "rate_percent": 9.38}
    },
    "overall": {
      "detected": 384, "total": 4096, "rate_percent": 9.38
    }
  }
}
```

## 关键观察

### 1. 比特位敏感性

检测几乎完全由被翻转的比特位驱动：
- **尾数位**（0-22）：边界对细粒度扰动不敏感
- **指数/符号位**（23-31）：检测率急剧上升
- 这符合浮点语义：指数翻转改变数量级，符号翻转改变符号

### 2. 输入分布影响（单层实验）

更大的输入范围（σ≈1.67 vs 1.0）显著提高检测率：
- 更大的激活值 → 更大的logits → 边界更紧
- 小输入值时，边界之间有较大"松弛"，许多故障隐藏其中

### 3. 稀疏性和聚合（TinyLM实验）

- 64维嵌入 + 小词汇表 → 稀疏表示
- 单比特注入到一个头、一个位置
- 影响被其他头和位置稀释
- 批次间的大变化表明边界的紧致度依赖于瞬时注意力分布

### 4. 检测覆盖的局限性

即使在有利条件下（合成数据、小模型、无mask），绝对覆盖率仍然有限（10-30%），表明：
- 需要更紧的边界
- 可能需要额外的不变量
- 或与其他检测机制结合

## 扩展方向

基于实验结果，未来工作可以：

1. **改进边界**：
   - 推导更紧的上下界
   - 支持masked attention
   - 扩展到梯度检测（训练时）

2. **实验扩展**：
   - 测试更大模型（GPT-2等）
   - 多层同时注入
   - 与GPU故障模拟器集成

3. **检测增强**：
   - 结合多个等价计算路径
   - 利用不同浮点格式
   - 采样检测以降低开销

## 引用

如果使用本代码，请引用：

```bibtex
@article{bai2025sdc,
  title={Silent Data Corruption Detection Based on Metamorphic Attention Bounds},
  author={Bai, Xinyu},
  year={2025},
  institution={University of Illinois Urbana-Champaign}
}
```

## 相关工作

- [Dixit et al., 2021] - Silent Data Corruptions at Scale
- [Ma et al., 2025] - Understanding Silent Data Corruption in LLM Training
- [Hari et al., 2020] - Estimating Silent Data Corruption Rates Using a Two-Level Model

## 许可证

本项目仅用于研究目的。

## 联系方式

- 作者：Xinyu Bai
- 邮箱：xbai@illinois.edu
- 机构：University of Illinois Urbana-Champaign