"""
Phase 4 - 推理模块 (Prediction / Inference)
============================================
加载 Phase 3 训练好的 XGBoost 模型，对新数据进行风险预测。
提供统一的 API 接口供 Dashboard 和生产环境调用。

使用方式:
    from predict import RiskPredictor
    predictor = RiskPredictor("1")               # 加载 Channel 1 模型
    result = predictor.predict(data_row_dict)     # 单条推理
    batch_results = predictor.predict_batch(df)   # 批量推理
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


RISK_NAMES = {0: "正常 (Normal)", 1: "预警 (Warning)", 2: "高危 (Severe)"}
RISK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

# 动态解析项目根目录，兼容本地 Windows 和 Streamlit Cloud (Linux)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class RiskPredictor:
    """单通道风险预测器。"""

    def __init__(self, channel: str,
                 model_dir: Path = DEFAULT_MODEL_DIR):
        self.channel = channel
        ch_dir = model_dir / f"channel{channel}"

        # 加载模型
        model_path = ch_dir / f"{channel}_xgboost_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

        # 加载元数据
        meta_path = ch_dir / f"{channel}_model_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.features = self.meta["features"]
        self.class_names = self.meta.get("class_names", ["Normal", "Warning", "Severe"])

    def predict(self, data: dict | pd.Series) -> dict:
        """
        对单条测试数据进行风险推理。

        参数:
            data: 包含所需特征的字典或 Series

        返回:
            {
                "risk_level": 0/1/2,
                "risk_name": "正常/预警/高危",
                "risk_color": "#4CAF50/#FF9800/#F44336",
                "probabilities": [p0, p1, p2],
                "explanation": "诊断原因文本",
                "top_factors": [("feature_name", value, importance), ...]
            }
        """
        if isinstance(data, dict):
            data = pd.Series(data)

        # 提取特征向量
        feature_values = []
        missing = []
        for f in self.features:
            if f in data.index and pd.notna(data[f]):
                feature_values.append(float(data[f]))
            else:
                feature_values.append(0.0)
                missing.append(f)

        X = np.array([feature_values])

        # 预测
        pred_label = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0].tolist() if hasattr(self.model, "predict_proba") else [0, 0, 0]

        # 生成解释
        explanation = self._generate_explanation(data, pred_label, proba)

        # 特征贡献排名
        top_factors = self._get_top_factors(data)

        return {
            "risk_level": pred_label,
            "risk_name": RISK_NAMES.get(pred_label, "Unknown"),
            "risk_color": RISK_COLORS.get(pred_label, "#999999"),
            "probabilities": proba,
            "explanation": explanation,
            "top_factors": top_factors,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对批量数据进行风险推理，返回带预测结果的 DataFrame。
        """
        available = [f for f in self.features if f in df.columns]
        subset = df[available].fillna(0).copy()

        # 补齐缺失特征列
        for f in self.features:
            if f not in subset.columns:
                subset[f] = 0.0
        subset = subset[self.features]

        X = subset.values
        labels = self.model.predict(X)
        result = df.copy()
        result["risk_predicted"] = labels
        result["risk_name"] = [RISK_NAMES.get(int(l), "Unknown") for l in labels]

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            for i in range(proba.shape[1]):
                result[f"risk_prob_{i}"] = proba[:, i]

        return result

    def _generate_explanation(self, data: pd.Series, pred: int, proba: list) -> str:
        """根据输入特征值生成可读的诊断原因。"""
        parts = []

        leakrate = data.get("Leakrate", None)
        spec_pos = data.get("leak_spec_position", None)
        slope_20 = data.get("leak_roll_slope_20", None)
        zscore = data.get("leak_rolling_zscore_200", None)
        margin = data.get("leak_margin_to_upper", None)

        if pred == 0:
            parts.append("✅ 当前工件泄漏率处于安全范围")
            if leakrate is not None:
                parts.append(f"   泄漏率 = {leakrate:.4f}")
            if spec_pos is not None:
                parts.append(f"   规格位置 = {spec_pos:.1%}（远离上限）")

        elif pred == 1:
            parts.append("⚠️ 触发预警！")
            if spec_pos is not None and spec_pos >= 0.75:
                parts.append(f"   → 规格位置 {spec_pos:.1%} 已接近上限 75% 临界区")
            if slope_20 is not None and slope_20 > 0.005:
                parts.append(f"   → 短期斜率 {slope_20:.6f} 呈上升趋势")
            if zscore is not None and abs(zscore) >= 2.0:
                parts.append(f"   → Z-Score = {zscore:.2f}，显著偏离长期基线")
            if leakrate is not None:
                parts.append(f"   当前泄漏率 = {leakrate:.4f}")

        elif pred == 2:
            parts.append("🚨 高危！建议立即排查！")
            if leakrate is not None:
                parts.append(f"   → 泄漏率 = {leakrate:.4f}")
            if margin is not None and margin < 0:
                parts.append(f"   → 已超出上限！超标量 = {abs(margin):.4f}")
            if zscore is not None:
                parts.append(f"   → Z-Score = {zscore:.2f}")

        confidence = max(proba) * 100
        parts.append(f"   模型置信度: {confidence:.1f}%")

        return "\n".join(parts)

    def _get_top_factors(self, data: pd.Series, top_n: int = 5) -> list:
        """获取对当前预测影响最大的特征。"""
        if not hasattr(self.model, "feature_importances_"):
            return []

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        factors = []
        for idx in indices:
            fname = self.features[idx]
            fval = data.get(fname, None)
            fimp = float(importances[idx])
            factors.append((fname, fval, fimp))

        return factors


def load_channel_data(channel: str,
                      data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """加载指定通道的带风险标签数据（Phase 3 输出）。"""
    risk_path = data_dir / f"channel{channel}" / "data" / f"channel{channel}_risk_labeled.csv"
    if risk_path.exists():
        return pd.read_csv(risk_path, parse_dates=["DateTime"])

    # 回退到 Phase 2 输出
    labeled_path = data_dir / f"channel{channel}" / "data" / f"channel{channel}_labeled.csv"
    if labeled_path.exists():
        return pd.read_csv(labeled_path, parse_dates=["DateTime"])

    raise FileNotFoundError(f"No data found for channel {channel}")


def get_available_channels(model_dir: Path = DEFAULT_MODEL_DIR) -> list[str]:
    """扫描 data/models/ 目录，返回已训练模型的通道列表。"""
    channels = []
    if model_dir.exists():
        for d in sorted(model_dir.iterdir()):
            if d.is_dir() and d.name.startswith("channel"):
                ch = d.name.replace("channel", "")
                channels.append(ch)
    return channels


# ============================================================
# 命令行测试入口
# ============================================================

if __name__ == "__main__":
    print("Testing RiskPredictor...")
    channels = get_available_channels()
    print(f"Available channels: {channels}")

    for ch in channels:
        print(f"\n--- Channel {ch} ---")
        predictor = RiskPredictor(ch)
        df = load_channel_data(ch)
        print(f"  Loaded {len(df)} rows")

        # 随机抽检 3 条
        samples = df.sample(3, random_state=42)
        for _, row in samples.iterrows():
            result = predictor.predict(row)
            print(f"  [{result['risk_name']}] Leakrate={row.get('Leakrate', 'N/A'):.4f}, "
                  f"Prob={[f'{p:.2f}' for p in result['probabilities']]}")

    print("\n[*] Prediction module test complete!")
