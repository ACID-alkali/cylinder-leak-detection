"""
Phase 4: 交互式预警看板 (Streamlit Dashboard)
=============================================
集成 Phase 1~3 的全部成果，提供可视化交互界面。

启动方式：
    streamlit run src/app/dashboard.py
"""

import sys
from pathlib import Path

# 确保 src 目录在搜索路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models.predict import (
    RISK_COLORS,
    RISK_NAMES,
    RiskPredictor,
    get_available_channels,
    load_channel_data,
)

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="缸盖密封测试预警系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHANNEL_DISPLAY_NAMES = {
    "1": "Channel 1 (水道)",
    "2_old": "Channel 2 旧参数 (空腔/F=360)",
    "2_new": "Channel 2 新参数 (空腔/F=60)",
    "3": "Channel 3 (油道)",
}


# ============================================================
# 数据缓存
# ============================================================

@st.cache_data(ttl=600)
def cached_load_data(channel: str) -> pd.DataFrame:
    return load_channel_data(channel)


@st.cache_resource
def cached_load_predictor(channel: str) -> RiskPredictor:
    return RiskPredictor(channel)


# ============================================================
# 侧边栏
# ============================================================

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/000000/engine.png", width=64)
    st.sidebar.title("⚙️ 控制面板")

    channels = get_available_channels()
    if not channels:
        st.sidebar.error("未找到已训练模型，请先运行 Phase 3 训练。")
        st.stop()

    display_options = [CHANNEL_DISPLAY_NAMES.get(ch, ch) for ch in channels]
    selected_display = st.sidebar.selectbox("🔄 选择通道", display_options)
    selected_channel = channels[display_options.index(selected_display)]

    st.sidebar.markdown("---")

    # 预警模式
    mode = st.sidebar.radio(
        "🛡️ 预警模式",
        ["双模联合 (推荐)", "机器学习 (XGBoost)", "统计过程控制 (SPC)"],
        index=0,
    )

    st.sidebar.markdown("---")

    # 页面选择
    page = st.sidebar.radio(
        "📊 分析视图",
        ["概览仪表盘", "SPC 控制图", "单件诊断室", "多通道对比"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("缸盖密封测试数据分析及预警系统 v1.0")

    return selected_channel, mode, page


# ============================================================
# 视图 1: 概览仪表盘
# ============================================================

def render_overview(df: pd.DataFrame, channel: str, predictor: RiskPredictor):
    st.header(f"📈 {CHANNEL_DISPLAY_NAMES.get(channel, channel)} — 概览仪表盘")

    # ---------- KPI 卡片 ----------
    total = len(df)
    n_normal = int((df.get("risk_level", pd.Series(dtype=int)) == 0).sum()) if "risk_level" in df.columns else total
    n_warning = int((df.get("risk_level", pd.Series(dtype=int)) == 1).sum()) if "risk_level" in df.columns else 0
    n_severe = int((df.get("risk_level", pd.Series(dtype=int)) == 2).sum()) if "risk_level" in df.columns else 0
    n_oos = int(df["leak_out_of_spec"].sum()) if "leak_out_of_spec" in df.columns else 0
    pass_rate = (total - n_oos) / total * 100 if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总检测量", f"{total:,}")
    col2.metric("合格率", f"{pass_rate:.2f}%")
    col3.metric("🟢 正常件", f"{n_normal:,}")
    col4.metric("🟡 预警件", f"{n_warning:,}", delta=f"{n_warning/total*100:.1f}%", delta_color="inverse")
    col5.metric("🔴 高危件", f"{n_severe:,}", delta=f"{n_severe/total*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    # ---------- 日期筛选 ----------
    if "DateTime" in df.columns:
        date_min = df["DateTime"].min().date()
        date_max = df["DateTime"].max().date()
        date_range = st.slider(
            "📅 时间范围筛选",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
        )
        mask = (df["DateTime"].dt.date >= date_range[0]) & (df["DateTime"].dt.date <= date_range[1])
        df_filtered = df[mask]
    else:
        df_filtered = df

    # ---------- 泄漏率时序图 + 风险着色 ----------
    if "DateTime" in df_filtered.columns and "Leakrate" in df_filtered.columns:
        fig = go.Figure()

        if "risk_level" in df_filtered.columns:
            for level, name in RISK_NAMES.items():
                subset = df_filtered[df_filtered["risk_level"] == level]
                if len(subset) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset["DateTime"], y=subset["Leakrate"],
                    mode="markers",
                    marker=dict(
                        size=4 if level == 0 else 7,
                        color=RISK_COLORS[level],
                        opacity=0.3 if level == 0 else 0.8,
                    ),
                    name=name,
                    hovertemplate="时间: %{x}<br>泄漏率: %{y:.4f}<extra></extra>",
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df_filtered["DateTime"], y=df_filtered["Leakrate"],
                mode="markers", marker=dict(size=3, color="steelblue", opacity=0.3),
                name="Leakrate",
            ))

        # 上下限线
        if "Limit1" in df_filtered.columns:
            limit1 = df_filtered["Limit1"].iloc[0]
            fig.add_hline(y=limit1, line_dash="dash", line_color="red",
                          annotation_text=f"上限 {limit1}")
        if "Limit3" in df_filtered.columns:
            limit3 = df_filtered["Limit3"].iloc[0]
            fig.add_hline(y=limit3, line_dash="dash", line_color="green",
                          annotation_text=f"下限 {limit3}")

        # 滚动均值
        if "leak_roll_mean_100" in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered["DateTime"], y=df_filtered["leak_roll_mean_100"],
                mode="lines", line=dict(color="navy", width=1.5),
                name="滚动均值 (100)", opacity=0.6,
            ))

        fig.update_layout(
            title="泄漏率时序图 — 多级风险预警标记",
            xaxis_title="时间", yaxis_title="泄漏率 (cm³/min)",
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- 风险概率时序图 ----------
    prob_cols = [c for c in df_filtered.columns if c.startswith("risk_prob_")]
    if prob_cols and "DateTime" in df_filtered.columns:
        fig2 = go.Figure()
        prob_names = ["正常概率", "预警概率", "高危概率"]
        prob_colors = ["#4CAF50", "#FF9800", "#F44336"]
        for i, pc in enumerate(prob_cols):
            fig2.add_trace(go.Scatter(
                x=df_filtered["DateTime"], y=df_filtered[pc],
                mode="lines", line=dict(width=1, color=prob_colors[i]),
                name=prob_names[i], opacity=0.7,
            ))
        fig2.update_layout(
            title="模型预测风险概率时序",
            xaxis_title="时间", yaxis_title="概率",
            height=350, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 视图 2: SPC 控制图
# ============================================================

def render_spc(df: pd.DataFrame, channel: str):
    st.header(f"📏 {CHANNEL_DISPLAY_NAMES.get(channel, channel)} — SPC 统计过程控制图")

    if "Leakrate" not in df.columns or "DateTime" not in df.columns:
        st.warning("数据中缺少必要字段。")
        return

    leak = df["Leakrate"]
    mu = leak.rolling(200, min_periods=50).mean()
    sigma = leak.rolling(200, min_periods=50).std().replace(0, np.nan)

    fig = go.Figure()

    # 原始数据散点
    fig.add_trace(go.Scatter(
        x=df["DateTime"], y=leak,
        mode="markers", marker=dict(size=3, color="gray", opacity=0.3),
        name="Leakrate",
    ))

    # 均值线
    fig.add_trace(go.Scatter(
        x=df["DateTime"], y=mu,
        mode="lines", line=dict(color="blue", width=2),
        name="均值 μ",
    ))

    # 控制线
    for n, label, color, dash in [
        (1, "±1σ", "green", "dot"),
        (2, "±2σ", "orange", "dash"),
        (3, "±3σ", "red", "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["DateTime"], y=mu + n * sigma,
            mode="lines", line=dict(color=color, width=1, dash=dash),
            name=f"+{label}", showlegend=(n == 3),
        ))
        fig.add_trace(go.Scatter(
            x=df["DateTime"], y=mu - n * sigma,
            mode="lines", line=dict(color=color, width=1, dash=dash),
            name=f"-{label}", showlegend=False,
        ))

    # SPC 规则触发点高亮
    spc_cols = [c for c in df.columns if c.startswith("spc_rule") and c != "spc_any_warning"]
    if spc_cols:
        spc_any = df.get("spc_any_warning", pd.Series(False, index=df.index))
        triggered = df[spc_any == True]
        if len(triggered) > 0:
            fig.add_trace(go.Scatter(
                x=triggered["DateTime"], y=triggered["Leakrate"],
                mode="markers",
                marker=dict(size=6, color="red", symbol="x", opacity=0.6),
                name=f"SPC 触发 ({len(triggered)})",
            ))

    fig.update_layout(
        title="SPC 控制图 — Western Electric 规则监控",
        xaxis_title="时间", yaxis_title="泄漏率 (cm³/min)",
        height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # SPC 规则触发统计
    if spc_cols:
        st.subheader("📊 SPC 规则触发统计")
        rule_stats = {}
        rule_descriptions = {
            "spc_rule1_3sigma": "规则1: 单点超出 ±3σ",
            "spc_rule2_run": "规则2: 连续7点在均值同侧",
            "spc_rule3_trend": "规则3: 连续6点单调递增/递减",
            "spc_rule4_2sigma": "规则4: 3点中2点超出 ±2σ",
        }
        for col in spc_cols:
            if col in df.columns:
                count = int(df[col].sum())
                name = rule_descriptions.get(col, col)
                rule_stats[name] = count

        stats_df = pd.DataFrame(
            {"规则": rule_stats.keys(), "触发次数": rule_stats.values()}
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ============================================================
# 视图 3: 单件诊断室
# ============================================================

def render_diagnosis(df: pd.DataFrame, channel: str, predictor: RiskPredictor):
    st.header(f"🔍 {CHANNEL_DISPLAY_NAMES.get(channel, channel)} — 单件实时诊断")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("输入方式")
        input_mode = st.radio("选择输入方式", ["从历史数据随机抽检", "手动输入参数"], horizontal=True)

        if input_mode == "从历史数据随机抽检":
            if st.button("🎲 随机抽取一条数据", use_container_width=True):
                st.session_state["sample_idx"] = int(np.random.randint(0, len(df)))

            idx = st.session_state.get("sample_idx", 0)
            sample = df.iloc[idx]

        else:
            # 手动输入
            leakrate = st.number_input("泄漏率 (Leakrate)", value=0.05, format="%.4f")
            spec_pos = st.number_input("规格位置 (Spec Position)", value=0.6, format="%.4f")
            slope_20 = st.number_input("短期斜率 (Slope 20)", value=0.0, format="%.6f")
            zscore = st.number_input("Z-Score (200)", value=0.0, format="%.4f")
            pressure = st.number_input("测量压力", value=2.0, format="%.4f")

            sample = pd.Series({
                "Leakrate": leakrate,
                "leak_spec_position": spec_pos,
                "leak_roll_slope_20": slope_20,
                "leak_rolling_zscore_200": zscore,
                "Pressure Measured": pressure,
                "leak_roll_mean_5": leakrate,
                "leak_roll_mean_20": leakrate,
                "leak_roll_std_5": 0.01,
                "leak_roll_std_20": 0.01,
                "leak_roll_slope_5": slope_20,
                "leak_roll_slope_50": slope_20 * 0.5,
                "leak_roll_slope_100": slope_20 * 0.3,
                "leak_roll_range_20": 0.05,
                "leak_margin_to_upper": 2.0 - leakrate,
                "leak_nearest_margin": min(2.0 - leakrate, leakrate + 3.0),
                "leak_diff_1": 0.0,
                "leak_diff_5": 0.0,
                "pressure_diff_abs": 15.0,
                "pressure_fill_minus_measure": 0.3,
                "hour": 10,
                "weekday": 2,
                "seconds_since_prev": 5.0,
            })

    with col_right:
        # 执行推理
        result = predictor.predict(sample)

        # 风险等级大字展示
        level = result["risk_level"]
        color = result["risk_color"]
        name = result["risk_name"]

        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 20px; border-radius: 12px;
                        text-align: center; margin-bottom: 20px;">
                <h1 style="color: white; margin: 0;">风险等级: {name}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 概率进度条
        st.subheader("各级别置信度")
        prob_labels = ["🟢 正常", "🟡 预警", "🔴 高危"]
        prob_colors_bar = ["green", "orange", "red"]
        for i, (label, prob) in enumerate(zip(prob_labels, result["probabilities"])):
            st.write(f"{label}: {prob*100:.1f}%")
            st.progress(min(prob, 1.0))

        # 诊断原因
        st.subheader("🩺 AI 诊断原因")
        st.text(result["explanation"])

        # 关键特征贡献
        if result["top_factors"]:
            st.subheader("📊 关键影响因子 (Top 5)")
            factors_data = []
            for fname, fval, fimp in result["top_factors"]:
                factors_data.append({
                    "特征": fname,
                    "当前值": f"{fval:.4f}" if fval is not None else "N/A",
                    "重要性": f"{fimp:.4f}",
                })
            st.dataframe(pd.DataFrame(factors_data), use_container_width=True, hide_index=True)


# ============================================================
# 视图 4: 多通道对比
# ============================================================

def render_comparison():
    st.header("📊 多通道横向对比分析")

    channels = get_available_channels()
    if len(channels) < 2:
        st.warning("至少需要 2 个通道的数据才能进行对比。")
        return

    # 收集各通道统计数据
    stats = []
    all_dfs = {}
    for ch in channels:
        try:
            df = cached_load_data(ch)
            all_dfs[ch] = df
            total = len(df)
            n_oos = int(df["leak_out_of_spec"].sum()) if "leak_out_of_spec" in df.columns else 0
            n_warning = int((df.get("risk_level", pd.Series(dtype=int)) == 1).sum()) if "risk_level" in df.columns else 0
            n_severe = int((df.get("risk_level", pd.Series(dtype=int)) == 2).sum()) if "risk_level" in df.columns else 0
            spc_count = int(df["spc_any_warning"].sum()) if "spc_any_warning" in df.columns else 0

            stats.append({
                "通道": CHANNEL_DISPLAY_NAMES.get(ch, ch),
                "样本量": total,
                "泄漏率均值": f"{df['Leakrate'].mean():.4f}" if "Leakrate" in df.columns else "N/A",
                "泄漏率标准差": f"{df['Leakrate'].std():.4f}" if "Leakrate" in df.columns else "N/A",
                "真实超标数": n_oos,
                "模型预警数": n_warning,
                "模型高危数": n_severe,
                "SPC触发数": spc_count,
                "SPC触发率": f"{spc_count/total*100:.1f}%" if total > 0 else "N/A",
            })
        except Exception:
            continue

    if stats:
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    # 泄漏率分布对比
    st.subheader("泄漏率分布对比 (箱线图)")
    box_data = []
    for ch, df in all_dfs.items():
        if "Leakrate" in df.columns:
            temp = df[["Leakrate"]].copy()
            temp["通道"] = CHANNEL_DISPLAY_NAMES.get(ch, ch)
            box_data.append(temp)

    if box_data:
        combined = pd.concat(box_data, ignore_index=True)
        # 过滤极端值以便可视化
        q99 = combined["Leakrate"].quantile(0.99)
        q01 = combined["Leakrate"].quantile(0.01)
        combined_filtered = combined[
            (combined["Leakrate"] >= q01) & (combined["Leakrate"] <= q99)
        ]
        fig = px.box(combined_filtered, x="通道", y="Leakrate",
                     color="通道", title="各通道泄漏率分布对比 (去除极端 1%)")
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 风险等级分布对比
    st.subheader("风险等级分布对比")
    risk_data = []
    for ch, df in all_dfs.items():
        if "risk_level" in df.columns:
            for level, name in RISK_NAMES.items():
                count = int((df["risk_level"] == level).sum())
                risk_data.append({
                    "通道": CHANNEL_DISPLAY_NAMES.get(ch, ch),
                    "风险等级": name,
                    "数量": count,
                    "占比": count / len(df) * 100,
                })

    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        fig = px.bar(risk_df, x="通道", y="占比", color="风险等级",
                     color_discrete_map={
                         "正常 (Normal)": "#4CAF50",
                         "预警 (Warning)": "#FF9800",
                         "高危 (Severe)": "#F44336",
                     },
                     title="各通道风险等级占比 (%)",
                     barmode="group")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 主程序
# ============================================================

def main():
    selected_channel, mode, page = render_sidebar()

    # 加载数据和模型
    try:
        df = cached_load_data(selected_channel)
        predictor = cached_load_predictor(selected_channel)
    except FileNotFoundError as e:
        st.error(f"加载失败: {e}")
        st.stop()

    # 路由到对应视图
    if page == "概览仪表盘":
        render_overview(df, selected_channel, predictor)
    elif page == "SPC 控制图":
        render_spc(df, selected_channel)
    elif page == "单件诊断室":
        render_diagnosis(df, selected_channel, predictor)
    elif page == "多通道对比":
        render_comparison()


if __name__ == "__main__":
    main()
