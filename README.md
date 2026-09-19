# HierL2GDTA

Hierarchical L2G Drug-Target Affinity (DTA) 预测模型。

基于 ESM-2 蛋白质序列编码器和 ChemBERTa 配体 SMILES 编码器，通过层级图结构（Hierarchical L2G）实现药物-靶点亲和力预测。

---

## 目录结构

```
HierL2GDTA/
├── bindingdb/                  # BindingDB 数据预处理
│   └── bindingdb.py
├── davis/                      # Davis 数据预处理
│   └── davis.py
├── lib/                        # 第三方预训练模型权重（离线加载）
│   ├── ChemBERTa/models/       # ChemBERTa-77M-MTR（配体编码器）
│   └── esm/hub/checkpoints/    # ESM-2 650M（蛋白质编码器）
├── src/
│   ├── encoders/
│   │   ├── esm_encoder.py      # ESM-2 蛋白质序列编码器
│   │   └── chemberta_encoder.py # ChemBERTa 配体 SMILES 编码器
│   ├── model.py                # HierL2G 模型定义
│   ├── config.py               # 默认模型配置
│   ├── data.py                 # 数据集与 DataLoader
│   ├── engine.py               # 训练/评估循环
│   ├── train.py                # 主训练入口
│   ├── extract_features.py     # 特征预提取（支持断点续提）
│   ├── checkpoint.py           # 检查点保存/加载
│   ├── metrics.py              # 评估指标
│   └── utils.py                # 工具函数
├── tools/
│   └── split_dataset.py        # 数据集随机划分（7:2:1）
└── README.md
```

---

## 第一步：安装环境依赖

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（推荐，训练和特征提取均支持 GPU）

### 安装依赖

```bash
pip install torch transformers fair-esm pandas numpy rdkit tqdm scipy
```

> ESM Python 包（`fair-esm`）需要通过 pip 安装，模型权重则需手动下载（见下一步）。

---

## 第二步：下载预训练模型

项目使用两个预训练模型，均以离线模式加载（`local_files_only=True`），请下载到 `lib/` 目录对应位置。

### 2.1 ESM-2 650M（蛋白质编码器）

| 项目 | 内容 |
|------|------|
| 模型 | `esm2_t33_650M_UR50D` |
| 参数量 | 6.5 亿 / 33 层 |
| 输出维度 | 1280 维 / 残基 |
| 放置路径 | `lib/esm/hub/checkpoints/esm2_t33_650M_UR50D.pt` |

**下载方式：**

```bash
# 创建目录
mkdir -p lib/esm/hub/checkpoints

# 下载权重（约 2.5 GB）
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
    -P lib/esm/hub/checkpoints/
```

也可从 [fair-esm GitHub Release 页](https://github.com/facebookresearch/esm#available-models) 手动下载。

### 2.2 ChemBERTa-77M-MTR（配体编码器）

| 项目 | 内容 |
|------|------|
| 模型 | `DeepChem/ChemBERTa-77M-MTR` |
| 参数量 | 77M |
| 输出维度 | 384 维 / token |
| 放置路径 | `lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/` |

**下载方式一：Python 脚本（推荐）**

```python
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/snapshots/main")
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="DeepChem/ChemBERTa-77M-MTR",
    local_dir=str(model_dir),
    local_dir_use_symlinks=False
)
```

**下载方式二：git clone（需安装 git-lfs）**

```bash
git lfs install
git clone https://huggingface.co/DeepChem/ChemBERTa-77M-MTR \
    lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/snapshots/main
```

> 代码会自动扫描 `snapshots/` 下的子目录，不依赖具体的 commit hash 目录名，下载到任意子目录名均可。

### 2.3 验证模型就绪

下载完成后，目录结构应如下：

```
lib/
├── esm/hub/checkpoints/
│   └── esm2_t33_650M_UR50D.pt
└── ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/
    └── snapshots/
        └── main/         # 或任意 commit hash 目录名
            ├── config.json
            ├── pytorch_model.bin  # 或 model.safetensors
            ├── tokenizer.json
            ├── vocab.json
            ├── merges.txt
            └── ...
```

---

## 第三步：下载原始数据集

### 3.1 Davis 数据集

```bash
# 下载到 davis/ 目录
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/davis_v1.tgz -P davis/

# 解压
tar -xzf davis/davis_v1.tgz -C davis/
```

解压后得到 `davis/davis.csv`。

### 3.2 BindingDB 数据集（202604 版本）

从 [BindingDB 官网](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes) 下载 `BindingDB_All.tsv`，放入 `bindingdb/` 目录：

```bash
# 下载（请以官网最新链接为准）
wget "https://www.bindingdb.org/bind/downloads/BindingDB_All_202604.tsv.zip" -P bindingdb/

# 解压
unzip bindingdb/BindingDB_All_202604.tsv.zip -d bindingdb/
```

解压后得到 `bindingdb/BindingDB_All.tsv`。

---

## 第四步：标准训练流程

### 4.1 BindingDB

```bash
# 1. 预处理
python bindingdb/bindingdb.py bindingdb/BindingDB_All.tsv bindingdb/bindingdb_filtered.csv

# 2. 划分（7:2:1，输出到 data/）
python tools/split_dataset.py --input bindingdb/bindingdb_filtered.csv --output_dir data/bindingdb

# 3. 提取特征
python src/extract_features.py \
    --input data/bindingdb/splited.csv \
    --cache_dir feature_cache/bindingdb \
    --device cuda

# 4. 训练
python src/train.py \
    --data_file data/bindingdb/splited.csv \
    --cache_root feature_cache/bindingdb \
    --save_dir checkpoints/bindingdb \
    --use_amp
```

### 4.2 Davis

```bash
# 1. 预处理
python davis/davis.py --input davis/davis.csv --output davis/davis_filtered.csv

# 2. 划分（7:2:1，输出到 data/）
python tools/split_dataset.py --input davis/davis_filtered.csv --output_dir data/davis

# 3. 提取特征
python src/extract_features.py \
    --input data/davis/splited.csv \
    --cache_dir feature_cache/davis \
    --device cuda

# 4. 训练
python src/train.py \
    --data_file data/davis/splited.csv \
    --cache_root feature_cache/davis \
    --save_dir checkpoints/davis \
    --use_amp
```

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--batch_size` | 16 | 批大小 |
| `--epochs` | 50 | 最大训练轮数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--max_seq_len` | 1400 | 序列最大长度 |
| `--max_c_len` | 16 | 蛋白上下文 token 数 |
| `--target_col` | 自动识别 | 标签列（pKd / pKi / affinity） |
| `--use_amp` | 关闭 | 混合精度训练 |

**优化器：** AdamW + ReduceLROnPlateau（factor=0.5, patience=3）+ MSE Loss

**输出（`save_dir` 下）：** `best_model.pt` / `last_model.pt` / `test_metrics.json` / `test_predictions.csv`

> 特征提取支持断点续提和蛋白质去重编码；训练支持断点续训（从 `last_model.pt` 自动恢复）。

---

## 第五步：个人数据集训练流程

如果你有自己的 DTA 数据集（包含配体 SMILES、蛋白质序列、亲和力标签），可以按照以下步骤直接训练，无需使用 bindingdb / davis 的预处理脚本。

### 5.1 数据格式要求

准备一个 CSV 文件，**至少包含以下列**：

| 列名 | 必填 | 说明 |
|------|------|------|
| `smiles` | ✅ | 配体 SMILES 字符串 |
| `protein` | ✅ | 蛋白质氨基酸序列 |
| `pKd`（或 `pKi` / `affinity`） | ✅（训练模式） | 亲和力标签，数值类型 |
| `split` | ❌ | 划分标记（train / val / test），没有则需用 split_dataset.py 划分 |

> 标签列名支持**大小写不敏感**自动识别：优先查找 `pKd`，其次 `pKi`，再次 `affinity`。
> 也可以通过 `--target_col` 手动指定任意列名。

### 5.2 快速三步法

```bash
# 假设你的数据文件是 mydata/data.csv，包含 smiles、protein、pKd 三列

# 第一步：划分数据集（7:2:1）
python tools/split_dataset.py \
    --input mydata/data.csv \
    --output_dir data/mydata

# 第二步：提取特征
python src/extract_features.py \
    --input data/mydata/splited.csv \
    --cache_dir feature_cache/mydata \
    --batch_size 64 \
    --device cuda

# 第三步：训练
python src/train.py \
    --data_file data/mydata/splited.csv \
    --cache_root feature_cache/mydata \
    --save_dir checkpoints/mydata \
    --use_amp \
    --epochs 50 \
    --batch_size 16
```

### 5.3 进阶：自定义划分

如果你的数据集已有预设的划分（如按蛋白聚类划分、按药物聚类划分等），可以在 CSV 中加入 `split` 列（值为 `train` / `val` / `test`），直接跳过 `split_dataset.py`，从提取特征开始：

```bash
# CSV 已有 split 列，直接提特征
python src/extract_features.py \
    --input mydata/data_with_split.csv \
    --cache_dir feature_cache/mydata \
    --device cuda

# 直接训练
python src/train.py \
    --data_file mydata/data_with_split.csv \
    --cache_root feature_cache/mydata \
    --save_dir checkpoints/mydata \
    --use_amp
```

### 5.4 进阶：已有特征缓存

如果特征已经由他人提取好（包含 `a_{idx}.pt`、`b_{idx}.pt`、`c_{idx}.pt`），可以直接训练：

```bash
python src/train.py \
    --data_file mydata/splited.csv \
    --cache_root /path/to/feature_cache \
    --save_dir checkpoints/mydata \
    --use_amp
```

特征缓存支持两种目录布局：
- **按 split 分子目录**：`cache_root/train/a_0.pt` 等（推荐，与 extract_features.py 输出一致）
- **扁平目录**：`cache_root/a_0.pt` 等（兼容旧格式）

### 5.5 数据建议

- **SMILES**：建议长度 ≤ 300，超长相干可能被截断
- **蛋白质序列**：建议长度 50 ~ 1400（可通过 `--max_seq_len` 调整）
- **标签**：pKd / pKi 形式效果最好，取值范围通常在 4~12 之间
- **数据量**：建议训练集至少 1000 条以上，否则容易过拟合

---

## 第六步：预测（推理）

训练好模型后，可以用 `src/predict.py` 对新的 SMILES / 蛋白质对进行亲和力预测。

### 6.1 输入格式

CSV 至少包含 `smiles` 和 `protein` 两列。如果有真实标签列（`pKd` / `pKi` / `affinity`），会自动计算预测指标。

**不需要 `split` 列。**

### 6.2 预测命令

**最简用法：**

```bash
python src/predict.py \
    --input data/predict/my_data.csv \
    --model_path checkpoints/bindingdb/best_model.pt \
    --device cuda
```

**完整参数示范：**

```bash
python src/predict.py \
    --input data/predict/candidates.csv \
    --model_path checkpoints/bindingdb/best_model.pt \
    --output predict/candidates_result.csv \
    --cache_dir feature_cache/predict/candidates \
    --target_col pKd \
    --device cuda \
    --batch_size 32 \
    --max_seq_len 1400 \
    --max_c_len 16
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|-------|------|
| `--input` | ✅ | - | 输入 CSV 路径 |
| `--model_path` | ✅ | - | 模型 checkpoint 路径（best_model.pt 或 last_model.pt） |
| `--output` | ❌ | `predict/<csv名>_predictions.csv` | 输出 CSV 路径 |
| `--cache_dir` | ❌ | `feature_cache/predict/<csv文件名>/` | 特征缓存目录 |
| `--target_col` | ❌ | 自动识别 | 标签列名（有标签时自动计算指标） |
| `--device` | ❌ | `cuda` | 设备：cuda / cpu |
| `--batch_size` | ❌ | 32 | 批大小 |
| `--max_seq_len` | ❌ | 1400 | 序列最大长度 |
| `--max_c_len` | ❌ | 16 | 蛋白上下文 token 数 |
| `--no_extract` | ❌ | 关闭 | 跳过特征提取（特征已缓存时使用） |

### 6.3 输出说明

输出 CSV 包含原 CSV 所有列，并新增 `prediction` 列（预测的亲和力值，已还原为原始尺度）。

如果输入 CSV 有标签列，还会在终端打印 MSE / RMSE / MAE / R² / Pearson 等指标。

**个人数据集存放建议**：输入数据放 `data/predict/` 目录（如 `data/predict/candidates.csv`），特征缓存自动存到 `feature_cache/predict/candidates/`，预测结果默认输出到 `predict/candidates_predictions.csv`。

### 6.4 特性

- **自动特征提取**：首次运行自动提取并缓存特征，后续重复预测直接复用
- **断点续提**：中断后重跑自动跳过已完成的样本
- **蛋白质去重**：同一蛋白质序列只编码一次
- **自动反归一化**：预测值自动还原为原始标签尺度（从 checkpoint 中读取 mean/std）
- **可选指标计算**：有真实标签时自动计算评估指标

---

## 模型架构

HierL2G 模型采用层级化的局部到全局（Local-to-Global）特征交互策略：

1. **局部编码**：分别使用 ESM-2 和 ChemBERTa 提取蛋白质和配体的初始表征
2. **局部交互**：通过多尺度卷积核（1/3/5）+ 空洞卷积提取局部结构模式
3. **层级图交互**：通过多层 Transformer 块实现蛋白质-配体的层级交互
4. **全局聚合**：注意力池化 + 全连接层输出亲和力预测值

默认配置：
- 隐藏维度：640
- Transformer 头数：4
- Transformer 块数：2
- GRU 层数：2
- Dropout：0.15

---

## 常见问题

**Q: 特征提取很慢？**
A: 蛋白质特征提取是瓶颈。同一蛋白序列只会编码一次，数据集中蛋白重复度越高速度越快。可适当调大 `--batch_size`。

**Q: 如何恢复中断的训练？**
A: 直接重新运行 `train.py` 即可，默认会从 `last_model.pt` 自动恢复。使用 `--no_resume` 可禁用。

**Q: 如何恢复中断的特征提取？**
A: 直接重新运行 `extract_features.py` 即可，会自动跳过已完成的样本。

**Q: 支持 pKi 标签吗？**
A: 支持。标签列名支持大小写不敏感自动识别（pKd / pKi / affinity），也可以用 `--target_col` 手动指定。

**Q: Windows 下可以运行吗？**
A: 代码本身是跨平台的，但硬链接在 Windows 部分文件系统下可能降级为文件复制（不影响功能，仅多占磁盘空间）。推荐 Linux 环境。
