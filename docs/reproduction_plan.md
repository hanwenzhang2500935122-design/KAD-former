# KAD-Former 复现方案

> 论文：*Semantic Alignment and Knowledge Injection for Cross-Modal Reasoning in Intelligent Horticultural Decision Support Systems*
> 期刊：Horticulturae 2026, 12, 23 | DOI: 10.3390/horticulturae12010023
> 作者：中国农业大学 曹宇涵 等

---

## 一、论文核心概要

### 1.1 问题定义
果树病害智能识别，覆盖 4 种果树（苹果/梨/葡萄/桃）的多类病害分类。核心挑战：
- 黑盒模型缺乏可解释性
- 弱病害特征识别不稳定
- 跨区域泛化能力差

### 1.2 框架架构 KAD-Former
三个核心模块：
1. **AKG（农业知识图谱）**：结构化表示"果树-病害-症状"语义关系，含 EAKD 边缘感知知识蒸馏
2. **SAM（语义对齐模块）**：双向跨模态注意力，将视觉特征和知识嵌入投影到统一语义空间
3. **KGA（知识引导注意力模块）**：以视觉特征为 Query、知识嵌入为 Key/Value，替代纯数据驱动的注意力

### 1.3 关键性能指标（论文报告）
| 指标 | 值 |
|------|-----|
| Accuracy | 0.946 ± 0.003 |
| F1-Score | 0.933 ± 0.003 |
| mAP | 0.938 ± 0.003 |
| Consistency@5 | 0.826 ± 0.006 |
| DGS（跨区域泛化） | 0.917 ± 0.004 |

---

## 二、复现环境配置

### 2.1 硬件需求（论文原始配置）
- GPU：NVIDIA A100 × 多卡（80GB 显存）
- CPU：AMD EPYC 7742 级别
- 内存：512GB
- 存储：NVMe SSD

### 2.2 替代硬件方案（复现推荐）
- **最低配置**：单卡 RTX 3090/4090（24GB 显存）或 V100（32GB）
- **中等配置**：单卡 A100（40GB）或 2× RTX 4090
- 通过减小 batch size（32→16/8）和梯度累积适配显存

### 2.3 软件环境
```bash
# 创建 conda 环境
conda create -n kad-former python=3.10 -y
conda activate kad-former

# 核心框架
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-geometric==2.3.1
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/repo.html

# 图像处理
pip install opencv-python==4.8.0.74
pip install albumentations==1.3.1
pip install timm==0.9.7          # 预训练 ViT backbone

# 知识图谱与嵌入
pip install transformers==4.37.0  # 文本编码器 (BERT)
pip install sentence-transformers
pip install networkx

# 训练工具
pip install wandb                # 实验跟踪
pip install tqdm
pip install matplotlib seaborn   # 可视化
pip install scikit-learn         # 评估指标
pip install einops               # 张量操作

# 边缘检测 (EAKD 模块)
pip install kornia
```

### 2.4 目录结构
```
KG_Transformer/
├── data/
│   ├── raw/                     # 原始图像
│   │   ├── apple/
│   │   ├── pear/
│   │   ├── grape/
│   │   └── peach/
│   ├── processed/               # 预处理后图像 (224×224)
│   ├── knowledge_graph/         # AKG 数据
│   │   ├── entities.json        # 实体定义
│   │   ├── relations.json       # 关系三元组
│   │   └── node_descriptions/   # 节点文本描述
│   └── splits/                  # 5-fold 划分
├── configs/
│   └── default.yaml             # 训练配置
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit_backbone.py      # ViT-B 视觉骨干
│   │   ├── akg.py               # 农业知识图谱模块
│   │   ├── sam.py               # 语义对齐模块
│   │   ├── kga.py               # 知识引导注意力模块
│   │   └── kad_former.py        # 完整 KAD-Former
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集与增强
│   │   └── kg_builder.py        # 知识图谱构建
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标
│   │   ├── visualization.py     # 热力图可视化
│   │   └── seed.py              # 随机种子
│   └── train.py                 # 训练入口
├── scripts/
│   ├── build_kg.py              # AKG 构建脚本
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── ablation.py              # 消融实验
├── requirements.txt
└── README.md
```

---

## 三、数据准备（阶段一）

### 3.1 数据来源分析
论文使用多源异构数据，实际复现需自行获取：

| 来源 | 内容 | 获取方式 |
|------|------|----------|
| 实际果园采集 | 苹果/梨/葡萄/桃病害图像（4个地区） | 需自行采集或替代 |
| 农业诊断平台 | 省级农技推广站图像 | 需联系获取 |
| 公开数据集 | PlantVillage、AI Challenger 等 | 公开可下载 |
| 专家标注记录 | 症状-病害结构化描述 | 需农学专家 |

### 3.2 公开数据集替代方案（推荐优先使用）
```bash
# PlantVillage（最常用植物病害数据集）
# 下载地址: https://github.com/spMohanty/PlantVillage-Dataset
# 包含苹果多种病害

# AI Challenger 农作物病害
# https://challenge.ai.iqiyi.com/

# 另外，论文中的4种果树×多种病害类别需要手动筛选子集
```

### 3.3 数据预处理流程
```python
# 1. 图像尺寸统一: Resize → 224×224 (ViT 标准 patch 大小 16×16, 共 14×14=196 tokens)
# 2. 数据划分: 70% train / 15% val / 15% test, 5-fold CV
# 3. 数据增强策略:
#    - 随机裁剪 + 水平翻转 + 颜色抖动 (基础)
#    - CutMix（式3-4）: 面向病斑区域的结构化混合
#    - 跨域模拟增强: 背景替换、颜色风格随机化、噪声注入
#    - 多尺度裁剪: 式6, s ∈ [smin, smax]
#    - 伪病斑模拟: 式7-8, 合成病斑增强弱症状样本
```

### 3.4 病害类别定义（参考论文 Table 1）
需要整理 4 种果树对应的主要病害：
- **苹果**: 轮纹病 (ring rot)、褐斑病 (brown spot)、黑星病 (black spot)、白粉病 (powdery mildew)、锈病 (rust)
- **梨**: 黑斑病、锈病、白粉病、轮纹病
- **葡萄**: 霜霉病 (downy mildew)、白粉病、黑痘病、灰霉病
- **桃**: 缩叶病、褐腐病、穿孔病

---

## 四、AKG 农业知识图谱构建（阶段二）

### 4.1 实体类型
```
ENTITY_TYPES = {
    "fruit_tree": ["苹果", "梨", "葡萄", "桃", ...],
    "disease": ["轮纹病", "褐斑病", "白粉病", ...],
    "symptom": ["同心轮纹", "褐色斑点", "白色粉状物", ...],
    "stage": ["早期", "中期", "晚期"],
    "location": ["叶片", "果实", "枝干"]
}
```

### 4.2 关系类型
```
RELATION_TYPES = {
    "species_disease": "果树-病害",
    "disease_symptom": "病害-症状",
    "symptom_stage": "症状-阶段",
    "symptom_location": "症状-部位"
}
```

### 4.3 三元组示例
```json
[
    {"head": "苹果", "relation": "species_disease", "tail": "轮纹病"},
    {"head": "轮纹病", "relation": "disease_symptom", "tail": "同心轮纹状病斑"},
    {"head": "同心轮纹状病斑", "relation": "symptom_stage", "tail": "早期"},
    {"head": "同心轮纹状病斑", "relation": "symptom_location", "tail": "果实"}
]
```

### 4.4 节点文本描述
每个节点 vᵢ 关联一段专家定义的文本描述 Tᵢ，包含：
- 典型病斑颜色/形状
- 边缘特征（清晰/模糊/晕圈）
- 纹理特征
- 好发部位和时期

### 4.5 AKG 编码流程（对应论文 Section 2.3.2）

```
Step 1: 文本编码
  Tᵢ → TextEncoder(BERT) → Hᵢ ∈ R^(n×dt)
  eᵢ = Pool(Hᵢ) ∈ R^dt       (式10, dt=768)
  xᵢ⁽⁰⁾ = W₀eᵢ + b₀ ∈ R^dk  (式11, dk=256)

Step 2: GNN 图编码
  hᵢ⁽ˡ⁺¹⁾ = σ(Σⱼ∈N(i) Wᵣ⁽ˡ⁾hⱼ⁽ˡ⁾)  (式12)
  - 多层 GNN，shared + type-specific branching
  - 关系感知消息传递

Step 3: EAKD 边缘感知知识蒸馏
  LEAKD = ‖E(I) - E(K)‖₂²  (式13)
  - E(I): 视觉边缘特征（kornia 边缘检测）
  - E(K): 知识边缘向量（症状描述的边缘相关属性）
```

### 4.6 知识图谱构建工具
- **GNN 框架**: DGL 或 PyTorch Geometric
- **文本编码器**: `bert-base-chinese`（中文农业文本），预训练 + 农业领域 MLM 微调
- **图结构**: 异构图，关系类型感知

---

## 五、模型实现（阶段三）

### 5.1 ViT 视觉骨干 (Section 2.3.1)
```
Backbone: ViT-B/16 (timm 库预训练)
  - 输入: 224×224 RGB
  - Patch size: 16×16
  - 视觉 token 数: Nv = 14×14 = 196
  - 视觉维度: Cv = 768
  - 输出: Xv ∈ R^(196×768)
```

### 5.2 SAM 语义对齐模块 (Section 2.3.3)
```
输入:
  - Xv ∈ R^(196×768)  (视觉特征)
  - Xk ∈ R^(Nk×256)   (知识嵌入, Nk≈200)

Step 1: 线性投影到统一空间 (ds=512)
  Zv = XvWv + 1bv⊤     (式14)
  Zk = XkWk + 1bk⊤     (式14)

Step 2: 双流 Transformer ×2 layers
  - Self-Attention (各模态独立)  (式15)
  - Cross-Attention (双向)        (式16-17)
  - FFN
  - 注意力头数: h=8, 头维度: dh=64

Step 3: 语义对齐损失
  Lalign = Σᵢ ‖cvᵢ - ckᵢ‖₂² + λ Σᵢ≠j max(0, m - ‖cvᵢ - ckⱼ‖₂²)  (式18)

输出:
  - Ẑv ∈ R^(196×512)
  - Ẑk ∈ R^(Nk×512)
```

### 5.3 KGA 知识引导注意力模块 (Section 2.3.4)
```
输入:
  - Q = Ẑv (视觉特征作为 Query)
  - K = V = Ẑk (知识嵌入作为 Key/Value)

结构: 3 个并行知识注意力块
  Block i:
    1. Linear Mapping: 768 → 256
    2. Conv Reconstruction: reshape → (14,14,256) → 3×3 Conv → 空间平滑
    3. Cross-modal Attention: A = softmax(QK^T / √dk)  (式20-22)
    4. Residual Fusion

输出: X' = AV  (式21)
  → 视觉 tokens 与知识 reasoning 的联合表示
```

### 5.4 总损失函数
```
L_total = L_cls + α·L_align + β·L_EAKD

其中:
  L_cls: 标准交叉熵分类损失
  L_align: 语义对齐损失 (式18)
  L_EAKD: 边缘感知知识蒸馏损失 (式13)
  α, β: 超参数权重
```

---

## 六、训练策略（阶段四）

### 6.1 训练超参数
| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 1×10⁻⁴ |
| 权重衰减 | 0.05 |
| Batch Size | 32 (显存不足可降至 16) |
| 训练轮数 | 100-200 epochs |
| 学习率调度 | Cosine Annealing + Warmup (5 epochs) |
| 数据划分 | 70/15/15, 5-fold CV |
| 随机种子 | 42 |
| 梯度裁剪 | max_norm=1.0 |

### 6.2 训练策略
```
Phase 1 (Epoch 1-20):  冻结文本编码器(BERT)，仅训练 ViT + SAM + KGA + GNN
Phase 2 (Epoch 21-50): 解冻 BERT 最后 2 层，小学习率 (1×10⁻⁵) 微调
Phase 3 (Epoch 51+):   全模型端到端微调
```

### 6.3 训练监控
- Weights & Biases 记录 loss 曲线、学习率、各指标
- 每 epoch 在验证集评估 accuracy、F1、mAP
- 保存 best checkpoint（按 validation accuracy）

---

## 七、评估与消融实验（阶段五）

### 7.1 评估指标实现
```python
# Accuracy, F1-Score, mAP (式24-26)
# sklearn: accuracy_score, f1_score, average_precision_score

# Consistency@K (式27): 注意力区域与专家标注症状区域的重叠
def consistency_at_k(attention_map, expert_mask, k=5):
    """计算 Top-K 注意力区域与专家标注区域的重叠率"""
    top_k_regions = get_top_k_regions(attention_map, k)
    overlap = (top_k_regions & expert_mask).any()
    return overlap

# DGS 跨区域泛化分数 (式28)
def compute_dgs(source_acc, target_accs):
    """DGS = (1/M) * Σ(target_acc_m / source_acc)"""
    return np.mean(target_accs) / source_acc
```

### 7.2 消融实验设计（对应论文 Table 4）
| 变体 | AKG | SAM | KGA | PLS | 目的 |
|------|-----|-----|-----|-----|------|
| ViT-B baseline | ✗ | ✗ | ✗ | ✗ | 基线 |
| ViT-B + AKG | ✓ | ✗ | ✗ | ✗ | 知识图谱贡献 |
| ViT-B + AKG + SAM | ✓ | ✓ | ✗ | ✗ | 对齐模块贡献 |
| KAD-Former (Full) | ✓ | ✓ | ✓ | ✓ | 完整模型 |
| w/o AKG | ✗ | ✓ | ✓ | ✓ | 移除知识图谱 |
| w/o SAM | ✓ | ✗ | ✓ | ✓ | 移除对齐模块 |
| w/o KGA | ✓ | ✓ | ✗ | ✓ | 移除注意力引导 |
| w/o PLS | ✓ | ✓ | ✓ | ✗ | 移除伪病斑增强 |

### 7.3 Baseline 模型对比
| 模型 | 类型 | 来源 |
|------|------|------|
| ResNet50 | CNN | torchvision |
| EfficientNet-B3 | CNN | timm |
| ConvNeXt-T | CNN | timm |
| ViT-B | Transformer | timm |
| DeiT-B | Transformer | timm |
| Swin-T | Transformer | timm |
| Pure KG | 纯知识推理 | 自实现 (TransE) |
| K-ViT | 知识增强ViT | 自实现 |

### 7.4 可解释性可视化
- KGA 注意力热力图 → Grad-CAM 式叠加显示
- 对齐空间 t-SNE 可视化
- 跨区域 attention 分布对比

---

## 八、复现时间规划

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 环境搭建 + 数据获取与预处理 | 3-5 天 |
| 2 | AKG 知识图谱构建（实体/关系/描述/GNN编码） | 5-7 天 |
| 3 | 模型实现（ViT骨干 + SAM + KGA） | 7-10 天 |
| 4 | 训练与调参 | 5-7 天 |
| 5 | 评估 + 消融实验 + Baseline对比 | 5-7 天 |
| 6 | 可视化 + 结果整理 | 3-5 天 |
| **总计** | | **约 4-6 周** |

---

## 九、复现难点与注意事项

### 9.1 关键挑战
1. **数据获取**：论文使用私有采集数据（4个地区、4种果树），复现需用公开数据集替代，类别和数量可能不完全匹配
2. **知识图谱构建**：需要农业专家知识或大量文献整理，节点描述的质量直接影响 AKG 效果
3. **EAKD 模块**：边缘感知知识蒸馏需要视觉边缘特征与知识边缘向量的对齐，实现细节论文未完全公开
4. **超参数敏感性**：α、β 权重、GNN 层数、对齐维度等需要细致调参

### 9.2 降级复现策略
如果完整复现困难，可按优先级降级：
- **P0（核心）**: ViT-B + AKG + SAM + KGA 主干流程跑通
- **P1（重要）**: 5-fold CV 评估 + Accuracy/F1/mAP 指标
- **P2（扩展）**: EAKD 边缘蒸馏 + PLS 伪病斑增强
- **P3（锦上添花）**: Consistency@5 + DGS 跨域实验 + 热力图可视化

### 9.3 论文未明确说明的实现细节（需自行推断）
- GNN 具体层数和 hidden size
- SAM 中 margin m 和权重 λ 的具体值
- EAKD 中边缘检测网络的具体架构
- KGA 中 3 个并行块的差异（是否共享权重）
- α、β 损失权重的具体数值
