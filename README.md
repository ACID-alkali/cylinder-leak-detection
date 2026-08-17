# 缸盖密封测试数据分析及预警 (Cylinder Head Leak Test Early Warning)

本项目旨在通过分析气密性测试设备的“充气压力”、“泄漏率”、“保压时间”等历史与实时数据，实现对工件密封性能的**早期异常预警**。它将取代原有的事后单纯靠固定阈值判定的模式。

## 📁 目录结构 (Directory Structure)

项目按照标准的 Data Science 模板构建：

- **`data/`**: 包含 `raw`（原始测试数据）、`reference`（工艺配置参数表）、`processed`（清洗后的特征矩阵）、`models`（预警模型权重）。
- **`docs/`**: 实习项目相关资料（项目书、PPT等）。
- **`notebooks/`**: 用于打草稿和探索性数据分析（EDA、聚类实验）的 Jupyter Notebook。
- **`src/`**: 核心生产级代码。
  - **`data_pipeline/`**: 包含 `clean_feature.py`（负责数据提取、清洗与特征工程）。
  - **`models/`**: 包含无监督聚类（`clustering.py`）、模型训练（`train.py`）和推理判定（`predict.py`）。
  - **`app/`**: 最终的交互式预警看板（`dashboard.py`）。

## 🚀 快速启动 (Quick Start)

### 1. 数据清洗与特征提取
首先运行数据清洗脚本，它将自动根据配置解析原始 XLS 文件，提取机器学习所需的斜率、边界距离等关键特征。

```bash
python src/data_pipeline/clean_feature.py --channel 1
```
清洗后的特征与自动生成的分析图表将保存在 `data/processed_features` 下。

### 2. 探索性分析与聚类 (🚧 WIP)
目前处于 Phase 2 阶段，请在 `notebooks` 中利用刚刚清洗出来的高质量数据，提取异常模式（如“爬坡型泄漏”、“高位震荡型泄漏”）。
