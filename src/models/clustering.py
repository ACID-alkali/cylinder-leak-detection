"""
Phase 2: 探索性数据分析 (EDA) + 无监督聚类 + 异常检测
=====================================================
本脚本对 Phase 1 清洗后的特征数据进行深入分析：
1. EDA: 分布特征、相关性、时间模式
2. KMeans 聚类: 发现泄漏率行为亚群
3. DBSCAN: 密度聚类，自动发现离群点
4. Isolation Forest: 异常评分
5. 输出带标签数据 + 可视化报告

运行方式：
    python src/models/clustering.py
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
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 配置
# ============================================================

# 用于聚类的核心特征 —— 这些特征从物理和统计两个维度描述了每次测试的状态
CLUSTERING_FEATURES = [
    "Leakrate",                  # 泄漏率原始值
    "leak_roll_mean_20",         # 短期均值趋势
    "leak_roll_std_20",          # 短期波动性
    "leak_roll_slope_20",        # 短期斜率（上升/下降趋势）
    "leak_roll_slope_100",       # 中期斜率
    "leak_spec_position",        # 在上下限中的相对位置 (0~1)
    "leak_rolling_zscore_200",   # 长期 z-score（偏离基线程度）
    "pressure_diff_abs",         # 压降绝对值
]

# EDA 相关性分析的扩展特征列表
EDA_FEATURES = CLUSTERING_FEATURES + [
    "Pressure Measured",
    "Pressure Filled",
    "Temperature Environment",
    "leak_roll_range_20",
    "leak_diff_1",
    "seconds_since_prev",
]


# ============================================================
# EDA 模块
# ============================================================

def run_eda(df: pd.DataFrame, channel: str, plot_dir: Path):
    """为单个通道生成全套 EDA 可视化图表。"""
    print(f"  [EDA] Generating exploratory analysis plots...")
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # ---------- 1. 泄漏率分布 (直方图 + KDE) ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "Leakrate" in df.columns:
        sns.histplot(df["Leakrate"], kde=True, bins=80, color="steelblue", ax=axes[0])
        axes[0].set_title(f"[{channel}] Leakrate Distribution")
        axes[0].set_xlabel("Leakrate (cm³/min)")

        # Box plot
        sns.boxplot(y=df["Leakrate"], color="steelblue", ax=axes[1])
        axes[1].set_title(f"[{channel}] Leakrate Box Plot")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_eda_distribution.png", dpi=200)
    plt.close()

    # ---------- 2. 核心特征相关性热力图 ----------
    available_eda = [c for c in EDA_FEATURES if c in df.columns]
    if len(available_eda) >= 4:
        corr_df = df[available_eda].dropna()
        if len(corr_df) > 100:
            corr_matrix = corr_df.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, square=True,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            )
            ax.set_title(f"[{channel}] Feature Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{channel}_eda_correlation.png", dpi=200)
            plt.close()

    # ---------- 3. 泄漏率时序趋势 + 滚动均值 ----------
    fig, ax = plt.subplots(figsize=(16, 5))
    if "DateTime" in df.columns and "Leakrate" in df.columns:
        ax.scatter(df["DateTime"], df["Leakrate"], alpha=0.15, s=3, color="gray", label="Raw")
        if "leak_roll_mean_20" in df.columns:
            ax.plot(df["DateTime"], df["leak_roll_mean_20"], color="blue",
                    linewidth=0.8, alpha=0.7, label="Rolling Mean (20)")
        if "leak_roll_mean_100" in df.columns:
            ax.plot(df["DateTime"], df["leak_roll_mean_100"], color="red",
                    linewidth=1.0, alpha=0.8, label="Rolling Mean (100)")
        if "Limit1" in df.columns:
            ax.axhline(y=df["Limit1"].iloc[0], color="red", linestyle="--",
                       alpha=0.6, label=f"Upper Limit ({df['Limit1'].iloc[0]})")
        if "Limit3" in df.columns:
            ax.axhline(y=df["Limit3"].iloc[0], color="green", linestyle="--",
                       alpha=0.6, label=f"Lower Limit ({df['Limit3'].iloc[0]})")
        ax.set_title(f"[{channel}] Leakrate Timeline with Rolling Averages")
        ax.set_xlabel("Time")
        ax.set_ylabel("Leakrate (cm³/min)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_eda_timeline.png", dpi=200)
    plt.close()

    # ---------- 4. 班次分布对比 ----------
    if "shift" in df.columns and "Leakrate" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.violinplot(data=df, x="shift", y="Leakrate",
                       order=["day", "middle", "night"], ax=axes[0], inner="box")
        axes[0].set_title(f"[{channel}] Leakrate by Shift (Violin)")

        shift_counts = df["shift"].value_counts()
        axes[1].bar(shift_counts.index, shift_counts.values, color=["#4CAF50", "#FF9800", "#2196F3"])
        axes[1].set_title(f"[{channel}] Test Count by Shift")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(plot_dir / f"{channel}_eda_shift.png", dpi=200)
        plt.close()

    # ---------- 5. 滚动斜率时序图（趋势预警核心） ----------
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    if "DateTime" in df.columns:
        if "leak_roll_slope_20" in df.columns:
            axes[0].plot(df["DateTime"], df["leak_roll_slope_20"],
                         color="orange", linewidth=0.5, alpha=0.7)
            axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[0].set_title(f"[{channel}] Short-term Slope (window=20)")
            axes[0].set_ylabel("Slope")

        if "leak_roll_slope_100" in df.columns:
            axes[1].plot(df["DateTime"], df["leak_roll_slope_100"],
                         color="crimson", linewidth=0.5, alpha=0.7)
            axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[1].set_title(f"[{channel}] Mid-term Slope (window=100)")
            axes[1].set_ylabel("Slope")
            axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_eda_slopes.png", dpi=200)
    plt.close()

    print(f"  [EDA] Saved 5 EDA plots to {plot_dir}")


# ============================================================
# 聚类模块
# ============================================================

def prepare_clustering_data(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    从 DataFrame 中提取聚类特征，去除 NaN，标准化。
    返回: (标准化矩阵, 对应子集 DataFrame, 使用的特征列名列表)
    """
    available = [c for c in CLUSTERING_FEATURES if c in df.columns]
    if len(available) < 4:
        raise ValueError(f"Not enough clustering features available. Found: {available}")

    subset = df[available].dropna().copy()
    if len(subset) < 50:
        raise ValueError(f"Too few valid rows for clustering: {len(subset)}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(subset.values)

    return X, subset, available


def run_kmeans(X: np.ndarray, max_k: int = 8) -> tuple[np.ndarray, int, list[float], list[float]]:
    """
    KMeans 聚类 + Elbow Method 自动选 K。
    返回: (聚类标签, 最优K, 各K的惯性列表, 各K的轮廓系数列表)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(max_k, len(X) - 1)
    if max_k < 2:
        max_k = 2

    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # 安全计算轮廓系数：数据极度集中时 KMeans 可能产生空簇
        try:
            n_unique = len(set(labels))
            if n_unique < 2:
                sil_scores.append(-1.0)
            else:
                sil_scores.append(silhouette_score(X, labels, sample_size=min(5000, len(X))))
        except ValueError:
            sil_scores.append(-1.0)

    # 选择轮廓系数最高的 K
    best_idx = int(np.argmax(sil_scores))
    best_k = list(K_range)[best_idx]

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels = km_final.fit_predict(X)

    print(f"  [KMeans] Best K={best_k} (silhouette={sil_scores[best_idx]:.4f})")
    return labels, best_k, inertias, sil_scores


def run_dbscan(X: np.ndarray) -> np.ndarray:
    """
    DBSCAN 密度聚类。自动检测噪声点（标签 = -1）。
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    # 自适应 eps: 使用 k-距离图的拐点 (Kneedle 几何拐点算法)
    k = min(10, max(2, len(X) // 500))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # 割线最大垂直距离法确定几何拐点 (Kneedle Algorithm)
    x = np.arange(len(k_distances))
    y = k_distances
    p1 = np.array([0, y[0]])
    p2 = np.array([len(y) - 1, y[-1]])
    vec = p2 - p1
    vec_norm_val = np.linalg.norm(vec)
    if vec_norm_val > 1e-12:
        vec_norm = vec / vec_norm_val
        points = np.column_stack((x, y))
        vec_to_points = points - p1
        dists = np.abs(vec_to_points[:, 0] * vec_norm[1] - vec_to_points[:, 1] * vec_norm[0])
        knee_idx = np.argmax(dists)
        eps_val = float(k_distances[knee_idx])
    else:
        eps_val = 0.5

    eps_val = max(eps_val, 0.3)  # 下限保护

    db = DBSCAN(eps=eps_val, min_samples=k)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  [DBSCAN] eps={eps_val:.3f}, clusters={n_clusters}, noise_points={n_noise} ({n_noise/len(X)*100:.1f}%)")

    return labels


def run_isolation_forest(X: np.ndarray, contamination: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest 异常检测。
    返回: (异常标签 1=正常/-1=异常, 异常分数)
    """
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_labels = iso.fit_predict(X)
    iso_scores = iso.decision_function(X)  # 越小越异常

    n_anomaly = int((iso_labels == -1).sum())
    print(f"  [IsoForest] Anomalies detected: {n_anomaly} ({n_anomaly/len(X)*100:.1f}%)")

    return iso_labels, iso_scores


# ============================================================
# 可视化模块
# ============================================================

def plot_elbow(inertias: list, sil_scores: list, best_k: int,
               channel: str, plot_dir: Path):
    """绘制 Elbow + 轮廓系数双轴图。"""
    K_range = range(2, 2 + len(inertias))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "steelblue"
    ax1.plot(K_range, inertias, "o-", color=color1, linewidth=2, markersize=6)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (SSE)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(K_range, sil_scores, "s--", color=color2, linewidth=2, markersize=6)
    ax2.set_ylabel("Silhouette Score", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.axvline(x=best_k, color="green", linestyle=":", linewidth=2,
                label=f"Best K={best_k}")
    ax1.legend(loc="upper right")
    ax1.set_title(f"[{channel}] KMeans Elbow + Silhouette Analysis")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_kmeans_elbow.png", dpi=200)
    plt.close()


def plot_cluster_scatter(X: np.ndarray, labels: np.ndarray,
                         feature_names: list, method: str,
                         channel: str, plot_dir: Path):
    """用 PCA 降维后绘制聚类散点图。"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))

    # 为噪声点 (-1) 使用灰色
    cmap = plt.cm.Set2
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c="gray",
                       alpha=0.3, s=8, label="Noise")
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[cmap(i % 8)], alpha=0.5, s=12,
                       label=f"Cluster {label}")

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title(f"[{channel}] {method} Clusters (PCA projection)")
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_{method.lower()}_scatter.png", dpi=200)
    plt.close()


def plot_anomaly_timeline(df: pd.DataFrame, iso_labels: np.ndarray,
                          iso_scores: np.ndarray, valid_idx: pd.Index,
                          channel: str, plot_dir: Path):
    """在时间线上叠加异常点标记。"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    if "DateTime" not in df.columns or "Leakrate" not in df.columns:
        plt.close()
        return

    dt = df.loc[valid_idx, "DateTime"].values
    leak = df.loc[valid_idx, "Leakrate"].values

    # 上图: 泄漏率 + 异常点高亮
    normal_mask = iso_labels == 1
    anomaly_mask = iso_labels == -1

    axes[0].scatter(dt[normal_mask], leak[normal_mask],
                    c="steelblue", alpha=0.2, s=5, label="Normal")
    axes[0].scatter(dt[anomaly_mask], leak[anomaly_mask],
                    c="red", alpha=0.8, s=20, marker="x", label="Anomaly")

    if "Limit1" in df.columns:
        axes[0].axhline(y=df["Limit1"].iloc[0], color="red", linestyle="--", alpha=0.4)
    if "Limit3" in df.columns:
        axes[0].axhline(y=df["Limit3"].iloc[0], color="green", linestyle="--", alpha=0.4)

    axes[0].set_title(f"[{channel}] Leakrate Timeline with Isolation Forest Anomalies")
    axes[0].set_ylabel("Leakrate (cm³/min)")
    axes[0].legend(fontsize=8)

    # 下图: 异常分数时序
    axes[1].plot(dt, iso_scores, color="purple", linewidth=0.4, alpha=0.6)
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    # 标记阈值线（分数 < 0 通常被视为异常区域）
    axes[1].fill_between(dt, iso_scores, 0,
                         where=(iso_scores < 0), color="red", alpha=0.1)
    axes[1].set_title(f"[{channel}] Isolation Forest Anomaly Score")
    axes[1].set_ylabel("Anomaly Score (lower = more anomalous)")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_anomaly_timeline.png", dpi=200)
    plt.close()


def plot_cluster_profiles(df_subset: pd.DataFrame, labels: np.ndarray,
                          feature_names: list, channel: str, plot_dir: Path):
    """雷达图/箱线图展示各聚类簇的特征差异。"""
    temp = df_subset.copy()
    temp["cluster"] = labels

    # 选取前6个特征做对比箱线图
    plot_feats = feature_names[:min(6, len(feature_names))]
    n_feats = len(plot_feats)
    n_clusters = len(set(labels) - {-1})

    if n_clusters < 2 or n_feats < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(plot_feats):
        if i >= len(axes):
            break
        sns.boxplot(data=temp[temp["cluster"] >= 0],
                    x="cluster", y=feat, hue="cluster", ax=axes[i], palette="Set2", legend=False)
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel("")

    # 隐藏多余的子图
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"[{channel}] KMeans Cluster Feature Profiles", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{channel}_cluster_profiles.png", dpi=200)
    plt.close()


# ============================================================
# 报告生成
# ============================================================

def generate_report(df: pd.DataFrame, km_labels: np.ndarray, db_labels: np.ndarray,
                    iso_labels: np.ndarray, iso_scores: np.ndarray,
                    valid_idx: pd.Index, feature_names: list,
                    best_k: int, channel: str, report_dir: Path):
    """生成文字版分析报告。"""
    report_path = report_dir / f"{channel}_phase2_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Phase 2 Analysis Report - Channel: {channel}\n")
        f.write(f"{'='*60}\n\n")

        # 基础统计
        f.write("1. DATA OVERVIEW\n")
        f.write(f"   Total rows: {len(df)}\n")
        f.write(f"   Valid rows for clustering (no NaN): {len(valid_idx)}\n")
        f.write(f"   Features used: {feature_names}\n")
        if "DateTime" in df.columns:
            f.write(f"   Date range: {df['DateTime'].min()} ~ {df['DateTime'].max()}\n")
        if "Leakrate" in df.columns:
            f.write(f"   Leakrate: mean={df['Leakrate'].mean():.4f}, "
                    f"std={df['Leakrate'].std():.4f}, "
                    f"min={df['Leakrate'].min():.4f}, "
                    f"max={df['Leakrate'].max():.4f}\n")
        f.write("\n")

        # KMeans 结果
        f.write("2. KMEANS CLUSTERING\n")
        f.write(f"   Optimal K: {best_k}\n")
        total_km = len(km_labels)
        for k in range(best_k):
            mask = km_labels == k
            count = int(mask.sum())
            pct = (count / total_km * 100) if total_km > 0 else 0.0
            subset = df.loc[valid_idx[mask]]
            mean_leak = float(subset["Leakrate"].mean()) if ("Leakrate" in subset.columns and not subset.empty) else 0.0
            f.write(f"   Cluster {k}: {count} samples ({pct:.1f}%), avg Leakrate={mean_leak:.4f}\n")
        f.write("\n")

        # DBSCAN 结果
        f.write("3. DBSCAN CLUSTERING\n")
        unique_db = sorted(set(db_labels))
        total_db = len(db_labels)
        for lbl in unique_db:
            mask = db_labels == lbl
            count = int(mask.sum())
            pct = (count / total_db * 100) if total_db > 0 else 0.0
            name = "Noise" if lbl == -1 else f"Cluster {lbl}"
            f.write(f"   {name}: {count} samples ({pct:.1f}%)\n")
        f.write("\n")

        # Isolation Forest 结果
        f.write("4. ISOLATION FOREST ANOMALY DETECTION\n")
        n_normal = int((iso_labels == 1).sum())
        n_anomaly = int((iso_labels == -1).sum())
        total_iso = len(iso_labels)
        norm_pct = (n_normal / total_iso * 100) if total_iso > 0 else 0.0
        anom_pct = (n_anomaly / total_iso * 100) if total_iso > 0 else 0.0
        f.write(f"   Normal: {n_normal} ({norm_pct:.1f}%)\n")
        f.write(f"   Anomaly: {n_anomaly} ({anom_pct:.1f}%)\n")
        f.write(f"   Anomaly score range: [{iso_scores.min():.4f}, {iso_scores.max():.4f}]\n")

        # 异常样本的特征概况
        if n_anomaly > 0:
            anomaly_data = df.loc[valid_idx[iso_labels == -1]]
            f.write(f"\n   Anomaly samples characteristics:\n")
            if "Leakrate" in anomaly_data.columns:
                f.write(f"     Leakrate: mean={anomaly_data['Leakrate'].mean():.4f}, "
                        f"std={anomaly_data['Leakrate'].std():.4f}\n")
            if "leak_spec_position" in anomaly_data.columns:
                f.write(f"     Spec Position: mean={anomaly_data['leak_spec_position'].mean():.4f}\n")
            if "leak_roll_slope_20" in anomaly_data.columns:
                f.write(f"     Short-term Slope: mean={anomaly_data['leak_roll_slope_20'].mean():.6f}\n")

        f.write(f"\n{'='*60}\n")

    print(f"  [Report] Saved to {report_path}")


# ============================================================
# 主流程：单通道处理
# ============================================================

def process_channel(channel: str, data_dir: Path, output_dir: Path):
    """对单个通道执行完整的 Phase 2 分析流程。"""
    csv_path = data_dir / f"channel{channel}" / "data" / f"channel{channel}_clean_features.csv"
    if not csv_path.exists():
        print(f"  [!] CSV not found: {csv_path}, skipping.")
        return

    print(f"  Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])

    # 创建输出目录
    plot_dir = output_dir / f"channel{channel}" / "plots"
    report_dir = output_dir / f"channel{channel}" / "reports"
    data_out_dir = output_dir / f"channel{channel}" / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: EDA
    run_eda(df, channel, plot_dir)

    # Step 2: 准备聚类数据
    try:
        X, df_subset, feature_names = prepare_clustering_data(df)
    except ValueError as e:
        print(f"  [!] Cannot cluster: {e}")
        return

    # 记录有效行的原始索引（用于回写标签）
    valid_idx = df_subset.index

    # Step 3: KMeans
    km_labels, best_k, inertias, sil_scores = run_kmeans(X)
    plot_elbow(inertias, sil_scores, best_k, channel, plot_dir)
    plot_cluster_scatter(X, km_labels, feature_names, "KMeans", channel, plot_dir)
    plot_cluster_profiles(df_subset, km_labels, feature_names, channel, plot_dir)

    # Step 4: DBSCAN
    db_labels = run_dbscan(X)
    plot_cluster_scatter(X, db_labels, feature_names, "DBSCAN", channel, plot_dir)

    # Step 5: Isolation Forest
    iso_labels, iso_scores = run_isolation_forest(X)
    plot_anomaly_timeline(df, iso_labels, iso_scores, valid_idx, channel, plot_dir)

    # Step 6: 将标签回写到原始数据并保存
    df.loc[valid_idx, "kmeans_cluster"] = km_labels
    df.loc[valid_idx, "dbscan_cluster"] = db_labels
    df.loc[valid_idx, "iso_forest_label"] = iso_labels   # 1=正常, -1=异常
    df.loc[valid_idx, "iso_forest_score"] = iso_scores   # 越小越异常

    # 转为 Int64 保持整数格式
    for col in ["kmeans_cluster", "dbscan_cluster", "iso_forest_label"]:
        df[col] = df[col].astype("Int64")

    labeled_path = data_out_dir / f"channel{channel}_labeled.csv"
    df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    print(f"  [Data] Labeled data saved to {labeled_path}")

    # Step 7: 生成文字报告
    generate_report(df, km_labels, db_labels, iso_labels, iso_scores,
                    valid_idx, feature_names, best_k, channel, report_dir)


# 动态定位项目根目录 (src/models/clustering.py -> 项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: EDA + Clustering + Anomaly Detection")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed"),
                        help="Directory containing Phase 1 processed data")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"),
                        help="Output directory for Phase 2 results (plots, reports, labeled data)")
    args = parser.parse_args()

    channels = ["1", "2_old", "2_new", "3"]

    for ch in channels:
        print(f"\n{'='*50}")
        print(f"[*] Phase 2: Processing channel '{ch}'")
        print(f"{'='*50}")
        try:
            process_channel(ch, Path(args.data_dir), Path(args.output_dir))
        except Exception as e:
            print(f"[-] Failed to process channel {ch}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] Phase 2 complete for all channels!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
