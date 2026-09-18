"""
Auto-ABTest Agent: 自动化 A/B 实验分析与统计诊断智能体 Web 交互系统
基于 Streamlit + Plotly 构建，专为数据科学、统计学与大模型 Agent 求职面试展示设计。
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

# 确保能加载 src 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent_workflow import build_ab_test_graph
from src import config

st.set_page_config(
    page_title="Auto-ABTest Agent - 统计学智能体",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Auto-ABTest Agent: 自动化 A/B 实验分析与统计诊断智能体</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">结合统计学严谨推断（SRM卡方检验 / 正态性与方差齐性自适应路由 / Cohen\'s d 效应量）与 LangGraph 状态机的工业级决策系统</div>', unsafe_allow_html=True)

# 侧边栏：参数配置
st.sidebar.header("⚙️ 实验数据与配置")

data_source = st.sidebar.radio(
    "选择数据来源",
    ["内置业务场景样本", "上传本地 CSV 文件"]
)

base_dir = os.path.dirname(os.path.abspath(__file__))
sample_files = {
    "场景 1: 长尾偏态数据（用户单日停留时长）": os.path.join(base_dir, "data", "ab_watch_duration_skewed.csv"),
    "场景 2: 方差不齐正态数据（用户满意度得分）": os.path.join(base_dir, "data", "ab_satisfaction_heteroscedastic.csv"),
    "场景 3: 分流故障样本（SRM 样本比例严重失衡）": os.path.join(base_dir, "data", "ab_srm_anomaly.csv"),
}

selected_csv_path = None
df = None

if data_source == "内置业务场景样本":
    selected_scene = st.sidebar.selectbox("选择测试场景", list(sample_files.keys()))
    selected_csv_path = sample_files[selected_scene]
    if os.path.exists(selected_csv_path):
        df = pd.read_csv(selected_csv_path)
    else:
        st.sidebar.error("样本文件未找到，请先运行 data/generate_sample_data.py 生成数据。")
else:
    uploaded_file = st.sidebar.file_uploader("上传您的 A/B 实验 CSV 数据", type=["csv"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            selected_csv_path = tmp.name
        df = pd.read_csv(selected_csv_path)

if df is not None:
    st.sidebar.subheader("📐 字段映射配置")
    all_cols = list(df.columns)
    
    default_group = "group" if "group" in all_cols else all_cols[0]
    group_col = st.sidebar.selectbox("分组列 (Group Column)", all_cols, index=all_cols.index(default_group))
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        st.error("数据集中未检测到数值型指标列！")
        st.stop()
        
    metric_col = st.sidebar.selectbox("待分析指标 (Metric Column)", numeric_cols, index=0)
    
    unique_groups = list(df[group_col].dropna().unique())
    if len(unique_groups) < 2:
        st.sidebar.error(f"分组列 `{group_col}` 中必须包含至少 2 个不同的实验组！当前仅检测到: {unique_groups}")
        st.stop()
        
    c_val = st.sidebar.selectbox("对照组取值 (Control Group)", unique_groups, index=0)
    remaining_groups = [g for g in unique_groups if g != c_val]
    t_val = st.sidebar.selectbox("实验组取值 (Treatment Group)", remaining_groups, index=0)

    user_query = st.sidebar.text_input("业务问题描述 (可选)", value=f"评估 {t_val} 相比于 {c_val} 在 {metric_col} 指标上的表现")

    st.sidebar.subheader("🤖 大模型润色设置 (可选)")
    api_key_input = st.sidebar.text_input("OpenAI / DeepSeek API Key", type="password", help="若留空则自动启用学术级确定性离线渲染引擎")
    if api_key_input:
        config.OPENAI_API_KEY = api_key_input
        custom_base_url = st.sidebar.text_input("Base URL", value="https://api.deepseek.com/v1")
        custom_model = st.sidebar.text_input("Model Name", value="deepseek-chat")
        config.OPENAI_BASE_URL = custom_base_url
        config.MODEL_NAME = custom_model

    start_btn = st.sidebar.button("🚀 启动智能体诊断分析", type="primary", use_container_width=True)

    # 主区域：数据概览与可视化
    with st.expander("📋 数据预览与描述统计", expanded=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**数据前 5 行预览 (总行数: {len(df):,})**")
            st.dataframe(df.head(5), use_container_width=True)
        with c2:
            st.markdown(f"**指标 `{metric_col}` 描述统计**")
            st.dataframe(df.groupby(group_col)[metric_col].describe().round(3), use_container_width=True)

    # 动态图表展示
    st.subheader("📈 实验组与对照组分布形态对比")
    sub_df = df[df[group_col].isin([c_val, t_val])].copy()
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # 直方图与分布
        fig_hist = px.histogram(
            sub_df, 
            x=metric_col, 
            color=group_col, 
            barmode="overlay",
            marginal="box",
            opacity=0.6,
            title=f"直方图与箱线图分布对比: {metric_col}",
            color_discrete_map={c_val: "#3B82F6", t_val: "#10B981"}
        )
        fig_hist.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with v_col2:
        # 样本量柱状图 (检查分流)
        count_df = sub_df[group_col].value_counts().reset_index()
        count_df.columns = [group_col, "count"]
        count_df["ratio"] = (count_df["count"] / count_df["count"].sum() * 100).round(2)
        count_df["label"] = count_df["count"].astype(str) + " (" + count_df["ratio"].astype(str) + "%)"
        
        fig_bar = px.bar(
            count_df,
            x=group_col,
            y="count",
            text="label",
            color=group_col,
            title="样本入组量与分流比例",
            color_discrete_map={c_val: "#3B82F6", t_val: "#10B981"}
        )
        fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # 点击运行智能体分析
    if start_btn:
        st.markdown("---")
        st.subheader("⚙️ 智能体状态机流转轨迹")
        
        status_box = st.status("正在初始化 LangGraph 状态机...", expanded=True)
        
        app = build_ab_test_graph()
        initial_state = {
            "data_path": selected_csv_path,
            "group_col": group_col,
            "metric_col": metric_col,
            "control_val": c_val,
            "treatment_val": t_val,
            "target_ratio": (0.5, 0.5),
            "user_query": user_query,
            "step_logs": []
        }

        with status_box:
            final_state = app.invoke(initial_state)
            for log in final_state.get("step_logs", []):
                st.write(log)
            status_box.update(label="✅ 智能体状态机执行完毕！", state="complete", expanded=False)

        # 结果报告渲染
        st.markdown("---")
        report_content = final_state.get("final_report", "未生成报告")
        st.markdown(report_content)

        # 下载报告按钮
        st.download_button(
            label="📥 导出完整诊断报告 (Markdown)",
            data=report_content,
            file_name=f"ab_test_report_{metric_col}.md",
            mime="text/markdown"
        )

else:
    st.info("👈 请在左侧侧边栏选择测试数据或上传您自己的实验 CSV 文件。")
