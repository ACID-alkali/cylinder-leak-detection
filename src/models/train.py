"""
Phase 3: 预警模型训练
====================
本脚本基于 Phase 2 生成的带标签数据，构建多级风险预警模型。

核心思路：
1. 风险标签工程：将 Phase 2 的无监督结果转化为三级风险标签
   - Level 0 (正常): 泄漏率处于安全区间，无趋势恶化
   - Level 1 (预警): 泄漏率接近上限或存在恶化趋势
   - Level 2 (高危): 被 Isolation Forest 标记为异常 或 泄漏超标
2. 梯度提升模型 (XGBoost) 训练
3. SPC 统计过程控制规则辅助判定
4. 模型评估与持久化

运行方式：
    python src/models/train.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 配置
# ============================================================

# 用于训练的特征列表 —— 这些特征在"当前时刻"可获取，不含未来信息
MODEL_FEATURES = [
    # 泄漏率原始值与滚动统计
    "Leakrate",
    "leak_roll_mean_5",
    "leak_roll_mean_20",
    "leak_roll_std_5",
    "leak_roll_std_20",
    "leak_roll_slope_5",
    "leak_roll_slope_20",
    "leak_roll_slope_50",
    "leak_roll_slope_100",
    "leak_roll_range_20",
    # 极限边缘
    "leak_spec_position",
    "leak_margin_to_upper",
    "leak_nearest_margin",
    # 差分与异常指标
    "leak_diff_1",
    "leak_diff_5",
    "leak_rolling_zscore_200",
    # 压力特征
    "pressure_diff_abs",
    "Pressure Measured",
    "pressure_fill_minus_measure",
    # 时间上下文
    "hour",
    "weekday",
    "seconds_since_prev",
]

# SPC 西部电气 (Western Electric) 规则的参数
SPC_CONFIG = {
    "zone_a_sigma": 3.0,    # ±3σ 超限 → 立即报警
    "zone_b_sigma": 2.0,    # ±2σ 区域
    "zone_c_sigma": 1.0,    # ±1σ 区域
    "run_length": 7,        # 连续同侧运行长度阈值
    "trend_length": 6,      # 连续递增/递减趋势长度
}


# ============================================================
# 风险标签工程
# ============================================================

def create_risk_labels(df: pd.DataFrame) -> pd.Series:
    """
    基于 Phase 2 的多维无监督分析结果，合成三级风险标签。

    Label 2 (高危):
        - Isolation Forest 判定为异常 (iso_forest_label == -1)
        - 或 泄漏超标 (leak_out_of_spec == True)
    Label 1 (预警):
        - 泄漏率 Spec Position >= 0.75 (接近上限 75% 位置)
        - 或 |z-score| >= 2 (显著偏离长期基线)
        - 或 DBSCAN 噪声点 (dbscan_cluster == -1)
    Label 0 (正常):
        - 以上条件均不满足
    """
    risk = pd.Series(0, index=df.index, dtype=int)

    # Level 1: 预警条件
    if "leak_spec_position" in df.columns:
        risk = risk.where(~(df["leak_spec_position"] >= 0.75), 1)

    if "leak_rolling_zscore_200" in df.columns:
        risk = risk.where(~(df["leak_rolling_zscore_200"].abs() >= 2.0), 1)

    if "dbscan_cluster" in df.columns:
        risk = risk.where(~(df["dbscan_cluster"] == -1), 1)

    # Level 2: 高危条件（覆盖 Level 1）
    if "iso_forest_label" in df.columns:
        risk = risk.where(~(df["iso_forest_label"] == -1), 2)

    if "leak_out_of_spec" in df.columns:
        risk = risk.where(~(df["leak_out_of_spec"] == True), 2)

    return risk


# ============================================================
# SPC 规则引擎
# ============================================================

def apply_spc_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    实现 Western Electric SPC 规则，为每条记录计算 SPC 预警标志。

    规则 1: 单点超出 ±3σ
    规则 2: 连续 7 点在中心线同侧
    规则 3: 连续 6 点递增或递减
    规则 4: 连续 2/3 点落在 ±2σ 之外
    """
    out = df.copy()

    if "Leakrate" not in out.columns:
        return out

    leak = out["Leakrate"]
    mu = leak.rolling(200, min_periods=50).mean()
    sigma = leak.rolling(200, min_periods=50).std()
    sigma = sigma.replace(0, np.nan)

    cfg = SPC_CONFIG

    # 规则 1: 单点超出 ±3σ
    out["spc_rule1_3sigma"] = ((leak - mu).abs() > cfg["zone_a_sigma"] * sigma)

    # 规则 2: 连续 run_length 点在均值同侧
    above = (leak > mu).astype(int)
    below = (leak < mu).astype(int)
    above_run = above.rolling(cfg["run_length"], min_periods=cfg["run_length"]).sum()
    below_run = below.rolling(cfg["run_length"], min_periods=cfg["run_length"]).sum()
    out["spc_rule2_run"] = (above_run == cfg["run_length"]) | (below_run == cfg["run_length"])

    # 规则 3: 连续 trend_length 点递增或递减
    increasing = (leak.diff() > 0).astype(int)
    decreasing = (leak.diff() < 0).astype(int)
    inc_run = increasing.rolling(cfg["trend_length"], min_periods=cfg["trend_length"]).sum()
    dec_run = decreasing.rolling(cfg["trend_length"], min_periods=cfg["trend_length"]).sum()
    out["spc_rule3_trend"] = (inc_run == cfg["trend_length"]) | (dec_run == cfg["trend_length"])

    # 规则 4: 连续 3 点中有 2 点超出 ±2σ
    beyond_2sigma = ((leak - mu).abs() > cfg["zone_b_sigma"] * sigma).astype(int)
    out["spc_rule4_2sigma"] = beyond_2sigma.rolling(3, min_periods=3).sum() >= 2

    # SPC 综合预警
    spc_cols = ["spc_rule1_3sigma", "spc_rule2_run", "spc_rule3_trend", "spc_rule4_2sigma"]
    out["spc_any_warning"] = out[spc_cols].any(axis=1)

    return out


# ============================================================
# 模型训练
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val, channel: str):
    """训练 XGBoost 多分类模型。"""
    from sklearn.utils.class_weight import compute_sample_weight

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("  [!] XGBoost not found, using sklearn GradientBoosting as fallback.")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    # 处理类别不平衡：自动计算样本权重
    sample_weights = compute_sample_weight("balanced", y_train)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model


def evaluate_model(model, X_val, y_val, feature_names: list,
                   channel: str, plot_dir: Path, report_dir: Path):
    """模型评估：分类报告、混淆矩阵、特征重要性。"""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_val)

    # 分类报告
    report_text = classification_report(
        y_val, y_pred,
        target_names=["Normal (0)", "Warning (1)", "Severe (2)"],
        zero_division=0,
    )
    print(f"\n  Classification Report:\n{report_text}")

    # 保存报告
    report_path = report_dir / f"{channel}_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Phase 3 Training Report - Channel: {channel}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Validation set size: {len(y_val)}\n")
        f.write(f"Label distribution in validation:\n")
        for lbl in [0, 1, 2]:
            count = int((y_val == lbl).sum())
            f.write(f"  Level {lbl}: {count} ({count/len(y_val)*100:.1f}%)\n")
        f.write(f"\n{report_text}\n")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Warning", "Severe"],
                yticklabels=["Normal", "Warning", "Severe"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"[{channel}] Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_confusion_matrix.png", dpi=200)
    plt.close()

    # 特征重要性
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(15, len(feature_names))
        top_idx = indices[:top_n]
        ax.barh(range(top_n),
                importances[top_idx][::-1],
                color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"[{channel}] Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{channel}_feature_importance.png", dpi=200)
        plt.close()

        # 写入报告
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\nTop {top_n} Feature Importances:\n")
            for rank, idx in enumerate(top_idx, 1):
                f.write(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}\n")

    print(f"  [Report] Saved to {report_path}")
    return report_text


def plot_risk_timeline(df: pd.DataFrame, channel: str, plot_dir: Path):
    """绘制风险等级时间线。"""
    if "DateTime" not in df.columns or "risk_level" not in df.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    dt = df["DateTime"]
    leak = df["Leakrate"] if "Leakrate" in df.columns else pd.Series(0, index=df.index)

    # 上图: 泄漏率 + 风险着色
    colors = {0: "steelblue", 1: "orange", 2: "red"}
    for level, color in colors.items():
        mask = df["risk_level"] == level
        label = {0: "Normal", 1: "Warning", 2: "Severe"}[level]
        axes[0].scatter(dt[mask], leak[mask], c=color, s=5, alpha=0.4, label=label)

    if "Limit1" in df.columns:
        axes[0].axhline(y=df["Limit1"].iloc[0], color="red", linestyle="--", alpha=0.4)
    if "Limit3" in df.columns:
        axes[0].axhline(y=df["Limit3"].iloc[0], color="green", linestyle="--", alpha=0.4)

    axes[0].set_title(f"[{channel}] Leakrate with Risk Level Coloring")
    axes[0].set_ylabel("Leakrate (cm³/min)")
    axes[0].legend(fontsize=8)

    # 下图: 模型预测概率（如果有的话）
    prob_cols = [c for c in df.columns if c.startswith("risk_prob_")]
    if prob_cols:
        for pc in prob_cols:
            level_name = pc.replace("risk_prob_", "Level ")
            axes[1].plot(dt, df[pc], linewidth=0.5, alpha=0.7, label=level_name)
        axes[1].set_title(f"[{channel}] Model Predicted Risk Probabilities")
        axes[1].set_ylabel("Probability")
        axes[1].legend(fontsize=8)
    else:
        # 如果没有概率，画 SPC 规则触发情况
        spc_cols = [c for c in df.columns if c.startswith("spc_rule")]
        if spc_cols:
            spc_any = df.get("spc_any_warning", pd.Series(False, index=df.index))
            axes[1].fill_between(dt, 0, spc_any.astype(int), alpha=0.3, color="red")
            axes[1].set_title(f"[{channel}] SPC Rule Triggers")
            axes[1].set_ylabel("SPC Warning Active")

    axes[1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_risk_timeline.png", dpi=200)
    plt.close()


# ============================================================
# 单通道处理流程
# ============================================================

def process_channel(channel: str, data_dir: Path, model_dir: Path):
    """对单个通道执行完整的 Phase 3 训练流程。"""
    csv_path = data_dir / f"channel{channel}" / "data" / f"channel{channel}_labeled.csv"
    if not csv_path.exists():
        print(f"  [!] Labeled data not found: {csv_path}, skipping.")
        return

    print(f"  Loading labeled data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])

    # 输出目录
    plot_dir = data_dir / f"channel{channel}" / "plots"
    report_dir = data_dir / f"channel{channel}" / "reports"
    ch_model_dir = model_dir / f"channel{channel}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ch_model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 创建风险标签
    print(f"  [Step 1] Creating risk labels...")
    df["risk_level"] = create_risk_labels(df)

    label_counts = df["risk_level"].value_counts().sort_index()
    for lbl, cnt in label_counts.items():
        pct = cnt / len(df) * 100
        name = {0: "Normal", 1: "Warning", 2: "Severe"}[lbl]
        print(f"    Level {lbl} ({name}): {cnt} ({pct:.1f}%)")

    # Step 2: SPC 规则
    print(f"  [Step 2] Applying SPC rules...")
    df = apply_spc_rules(df)
    spc_triggered = df["spc_any_warning"].sum() if "spc_any_warning" in df.columns else 0
    print(f"    SPC warnings triggered: {int(spc_triggered)} ({spc_triggered/len(df)*100:.1f}%)")

    # Step 3: 准备训练数据
    print(f"  [Step 3] Preparing training data...")
    available_features = [f for f in MODEL_FEATURES if f in df.columns]
    print(f"    Using {len(available_features)} features: {available_features[:5]}...")

    # 去掉特征中有 NaN 的行
    train_df = df[available_features + ["risk_level"]].dropna()
    if len(train_df) < 100:
        print(f"  [!] Too few valid rows ({len(train_df)}), skipping training.")
        return

    X = train_df[available_features].values
    y = train_df["risk_level"].values

    # 时间顺序切分：前 80% 训练，后 20% 验证（模拟真实时序预测场景）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"    Train: {len(X_train)}, Validation: {len(X_val)}")
    print(f"    Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"    Val   label dist: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # Step 4: 训练模型
    print(f"  [Step 4] Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_val, y_val, channel)

    # Step 5: 评估模型
    print(f"  [Step 5] Evaluating model...")
    evaluate_model(model, X_val, y_val, available_features, channel, plot_dir, report_dir)

    # Step 6: 将预测概率回写到完整数据
    valid_idx = train_df.index
    X_full = train_df[available_features].values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_full)
        for i in range(proba.shape[1]):
            df.loc[valid_idx, f"risk_prob_{i}"] = proba[:, i]

    df.loc[valid_idx, "risk_predicted"] = model.predict(X_full)

    # Step 7: 绘制风险时间线
    print(f"  [Step 6] Generating risk timeline...")
    plot_risk_timeline(df, channel, plot_dir)

    # Step 8: 保存模型
    print(f"  [Step 7] Saving model...")
    import joblib
    model_path = ch_model_dir / f"{channel}_xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"    Model saved to {model_path}")

    # 保存特征列表（推理时需要）
    meta = {
        "channel": channel,
        "features": available_features,
        "n_classes": 3,
        "class_names": ["Normal", "Warning", "Severe"],
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
    }
    meta_path = ch_model_dir / f"{channel}_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"    Model metadata saved to {meta_path}")

    # Step 9: 保存带风险标签的完整数据
    output_path = data_dir / f"channel{channel}" / "data" / f"channel{channel}_risk_labeled.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    Risk-labeled data saved to {output_path}")


# ============================================================
# 入口
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Train early-warning models for all channels")
    parser.add_argument("--data-dir", default=r"d:\Codes\Shangqishixi\data\processed",
                        help="Directory containing Phase 2 labeled data")
    parser.add_argument("--model-dir", default=r"d:\Codes\Shangqishixi\data\models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    channels = ["1", "2_old", "2_new", "3"]

    for ch in channels:
        print(f"\n{'='*50}")
        print(f"[*] Phase 3: Training model for channel '{ch}'")
        print(f"{'='*50}")
        try:
            process_channel(ch, Path(args.data_dir), Path(args.model_dir))
        except Exception as e:
            print(f"[-] Failed to train channel {ch}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] Phase 3 complete for all channels!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
