# KAD-Former Mini-Demo 完整实现 Prompt

将此 prompt 全文发送给 AI（Claude / GPT / 或其他代码 Agent），AI 将按顺序完成从环境搭建到训练验证的全部工作。

---

## 项目背景

复现论文 *Semantic Alignment and Knowledge Injection for Cross-Modal Reasoning in Intelligent Horticultural Decision Support Systems* (Horticulturae 2026)，框架名 KAD-Former。核心模块：AKG（农业知识注入） + SAM（语义对齐） + KGA（知识引导注意力）。

当前阶段：**最小可运行 Demo**。用 PlantVillage 公开数据集 + 已有知识图谱 JSON，实现完整版 AKG（BERT+GNN），SAM 和 KGA 做 Demo 级简化。

---

## 工作目录

```
D:\KG_Transformer\
```

## 已有文件

- `D:\KG_Transformer\data\knowledge_graph\apple_kg.json` — 苹果病害知识图谱三元组
- `D:\KG_Transformer\docs\reproduction_plan.md` — 完整复现方案（参考）

## 要创建的目录结构

```
D:\KG_Transformer\
├── data/
│   ├── knowledge_graph/
│   │   ├── apple_kg.json              (已有)
│   │   └── node_embeddings.pt         (由 kg_builder.py 生成)
│   └── plantvillage/                  (自动下载到此)
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit_backbone.py
│   │   ├── akg.py
│   │   ├── sam.py
│   │   ├── kga.py
│   │   └── kad_former.py
│   ├── kg_builder.py
│   ├── train.py
│   └── utils.py
├── checkpoints/
├── logs/
└── requirements.txt
```

---

## 任务要求

请按以下 5 个阶段顺序完成。每个阶段完成后请报告进度和结果。

---

## 阶段一：环境搭建与数据准备

### 1.1 创建 requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0
torch-geometric>=2.3.0
scikit-learn
matplotlib
seaborn
tqdm
einops
```

### 1.2 编写 `src/dataset.py`

要求：
- 从 torchvision.datasets 自动下载 PlantVillage 数据集，放到 `data/plantvillage/`
- 筛选苹果类别，仅保留 4 类：
  - `Apple___Apple_scab`（黑星病，标签 0）
  - `Apple___Black_rot`（黑腐病，标签 1）
  - `Apple___Cedar_apple_rust`（锈病，标签 2）
  - `Apple___healthy`（健康，标签 3）
- 图像预处理：Resize 到 224×224，归一化（ImageNet stats）
- 训练集增强：RandomHorizontalFlip + ColorJitter
- 验证集/测试集：仅 Resize + 归一化
- 数据划分：70% train / 15% val / 15% test
- 提供 `get_dataloaders(batch_size=16, num_workers=4)` 函数，返回 (train_loader, val_loader, test_loader)

### 1.3 验证

打印每个数据集的样本数和类别分布。

---

## 阶段二：知识图谱离线预计算（完整版 AKG）

**目标**：读 `apple_kg.json` → 构建 PyG 异构图 → BERT 编码节点文本 → GNN 关系传播 → 输出 `node_embeddings.pt`。

此阶段由 `src/kg_builder.py` 独立完成，运行一次即可，训练时直接加载 `.pt` 文件。

### 2.1 编写 `src/kg_builder.py`

**`apple_kg.json` 结构**：
- `entities[]`：`id`（节点ID）、`label`（中文名）、`type`（节点类别）、`description`（文本描述）
- `triples[]`：`head` / `relation` / `tail` 三元组

要求实现以下核心函数（仅列出接口，具体实现由 AI 自行决定）：

```python
def load_kg_json(json_path: str) -> dict:
    """读取 JSON 文件"""
    ...

def build_node_index(kg: dict) -> tuple:
    """
    为所有节点分配整数索引 (0..N-1)。
    返回 (id_to_idx, idx_to_id, idx_to_type, idx_to_description)
    """
    ...

def build_edges(kg: dict, id_to_idx: dict) -> dict:
    """
    按关系类型分组边。
    返回 {relation_type: [(head_idx, tail_idx), ...]}
    """
    ...

def encode_descriptions_with_bert(
    descriptions: list[str],
    model_name: str = "bert-base-chinese",
    device: str = "cuda",
) -> torch.Tensor:
    """
    用 BERT 对每个节点的 description 取 [CLS] 向量。
    输入: descriptions (N,)
    输出: bert_embeddings (N, 768)
    
    注意：节点文本为中文，必须用中文 BERT。
    """
    ...

def build_hetero_graph(
    num_nodes: int,
    edges_by_relation: dict,
    bert_embeddings: torch.Tensor,
) -> HeteroData:
    """
    构建 PyG HeteroData 对象，每种关系类型作为独立边类型。
    """
    ...

def propagate_gnn(
    data: HeteroData,
    hidden_dim: int = 256,
    num_layers: int = 2,
) -> torch.Tensor:
    """
    用 HeteroConv + SAGEConv 做关系感知消息传播。
    输入 dim: 768 (BERT)，隐藏 dim: 256，输出 dim: 256。
    输出: node_embeddings (N, 256)
    """
    ...

def main(json_path: str, output_path: str):
    """
    完整流程：load → build_node_index → BERT encode → build edges
    → build hetero graph → GNN propagate → save
    
    保存内容到 output_path：
    {
        'node_embeddings': (N, 256),
        'idx_to_id': list[str],
        'idx_to_type': list[str],
        'disease_to_idx': {disease_id: node_idx}  # 4 个病害映射
    }
    """
    ...
```

### 2.2 运行 kg_builder.py

```bash
python src/kg_builder.py
```

输出 `data/knowledge_graph/node_embeddings.pt`。

---

## 阶段三：模型模块

### 3.1 编写 `src/models/vit_backbone.py`

```python
class ViTBackbone(nn.Module):
    """
    输入: x (batch, 3, 224, 224)
    输出: (patch_tokens, cls_token)
      - patch_tokens: (batch, 196, 768)
      - cls_token:    (batch, 768)
    """
    def __init__(self, model_name="vit_small_patch16_224", pretrained=True):
        ...
    def forward(self, x):
        ...
```

要求：用 timm 创建 ViT-S/16，截取 patch tokens 和 CLS token。冻结部分底层参数，仅训练顶部几层。

### 3.2 编写 `src/models/akg.py`

```python
class AKG(nn.Module):
    """
    完整版 AKG：加载预计算的 GNN 节点嵌入，按类别索引查表。
    
    输入: labels (batch,) — 训练时用 GT，推理时用预测标签
    输出: Xk (batch, 1, 256)
    """
    def __init__(self, embeddings_path: str):
        ...
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        ...
```

要求：构造时加载 `node_embeddings.pt`，根据 `disease_to_idx` 取出 4 个病害节点的嵌入作为查找表。forward 时按 labels 索引返回对应的知识向量。

### 3.3 编写 `src/models/sam.py`

```python
class SAM(nn.Module):
    """
    完整版 SAM：双流 Self-Attention + 双向 Cross-Attention
    
    输入:
      - Xv: (batch, 196, 768)  视觉 patch tokens
      - Xk: (batch, 1, 256)    知识嵌入
    输出:
      - Zv: (batch, 196, 512)  对齐后的视觉特征
      - Zk: (batch, 1, 512)    对齐后的知识特征
    """
    def __init__(self, vision_dim=768, knowledge_dim=256, unified_dim=512, num_heads=8):
        ...
    def forward(self, Xv, Xk):
        ...
```

要求：
- 两路投影对齐到统一维度后，各自过一层 Self-Attention
- 再做双向 Cross-Attention：V→K（Q=视觉, K/V=知识）和 K→V（Q=知识, K/V=视觉）
- 残差连接 + LayerNorm

### 3.4 编写 `src/models/kga.py`

```python
class KGALite(nn.Module):
    """
    Demo 版 KGA：单块知识引导 Cross-Attention
    
    输入:
      - Zv: (batch, 196, 512)  SAM 输出的视觉特征
      - Zk: (batch, 1, 512)    SAM 输出的知识特征
    输出:
      - Zv': (batch, 196, 512) 知识引导后的视觉特征
    """
    def __init__(self, dim=512, num_heads=8):
        ...
    def forward(self, Zv, Zk):
        ...
```

要求：Q=Zv、K=V=Zk 的 Cross-Attention，加残差连接和 LayerNorm，输出维度不变。

---

## 阶段四：组装 KAD-Former 与训练脚本

### 4.1 编写 `src/models/kad_former.py`

```python
class KADFormerLite(nn.Module):
    """
    KAD-Former 最小 Demo 版
    
    数据流：
      Image → ViT Backbone → patch_tokens(196,768) + cls_token(768)
      Label → AKG(node_embeddings) → knowledge_vec(1,256)
      patch_tokens + knowledge_vec → SAM → Zv(196,512) + Zk(1,512)
      Zv + Zk → KGA → Zv'(196,512)
      Zv' pool + cls_token → classifier → logits(4)
    """
    def __init__(self, num_classes=4, embeddings_path: str):
        ...
    def forward(self, x, labels=None):
        """
        x: (batch, 3, 224, 224)
        labels: (batch,) 训练时传入 GT，推理时设为 None 则用粗分类预测
        """
        ...
```

要求：
- labels=None 时先用 ViT CLS token 粗分类得到预测标签，再查 AKG
- 视觉特征聚合：KGA 输出全局平均池化后与 CLS token 拼接，过分类头
- 总损失仅用交叉熵

### 4.2 编写 `src/train.py`

要求：
- 训练 30 epochs
- 优化器：AdamW, lr=1e-4, weight_decay=0.05
- 学习率调度：CosineAnnealingLR
- 每 epoch 评估验证集 Accuracy 和 F1
- 保存 best model（按 val accuracy）
- 支持 `--baseline` 参数：仅跑纯 ViT 做对比
- 记录训练日志

### 4.3 编写 `src/utils.py`

```python
def set_seed(seed=42):
    ...

def compute_metrics(preds, labels):
    """返回 acc, f1_macro, f1_weighted"""
    ...

class AverageMeter:
    """记录和计算平均值"""
    ...
```

---

## 阶段五：训练与验证

### 5.1 运行顺序

先运行 kg_builder 生成节点嵌入（仅需一次）：
```bash
cd D:\KG_Transformer
python src/kg_builder.py
```

再训练：
```bash
python src/train.py                # KAD-Former
python src/train.py --baseline     # 纯 ViT 对比
```

### 5.2 成功标准

1. 前向传播不报错：输入 `(16, 3, 224, 224)` + `labels (16,)`，输出 `(16, 4)`
2. 损失正常下降：30 epochs 内 train loss 稳定下降
3. 验证集精度：4 类苹果病害上 val accuracy > 0.80
4. 消融对比：KAD-Former 的 accuracy 不低于纯 ViT baseline

---

## 代码规范

- 所有 Python 代码使用类型注解
- 关键函数写 docstring
- 使用 `if __name__ == "__main__"` 保护入口
- 训练时自动检测并使用 GPU

## 注意事项

1. PlantVillage 首次下载约 800MB，需稳定网络
2. `apple_kg.json` 节点文本为中文，BERT 必须用 `bert-base-chinese`
3. kg_builder.py 需要的 BERT 模型首次运行会自动下载到缓存
4. 所有脚本在 `D:\KG_Transformer\` 目录下执行
5. 遇到 import 错误时，用 `pip install` 安装缺失的包
6. 显存不足时（< 8GB），将 batch_size 降到 8
