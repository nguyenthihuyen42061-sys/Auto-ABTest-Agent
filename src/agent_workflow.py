"""
LangGraph 状态机智能体工作流 (Agent Workflow)
搭建工业级确定性 A/B 测试诊断状态流，包含条件边动态路由与前置风控拦截
"""

import pandas as pd
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.state import ABTestState
from src.stats_engine import (
    check_srm,
    check_normality,
    check_homoscedasticity,
    run_adaptive_ab_test
)
from src.report_generator import generate_report

def data_ingestion_node(state: ABTestState) -> Dict[str, Any]:
    """
    节点 1: 数据摄入与预处理节点
    """
    logs = list(state.get("step_logs", []))
    logs.append("[1. 数据摄入] 正在加载并校验数据集...")

    data_path = state.get("data_path")
    group_col = state.get("group_col", "group")
    metric_col = state.get("metric_col")
    control_val = state.get("control_val", "control")
    treatment_val = state.get("treatment_val", "treatment")

    df = pd.read_csv(data_path)

    if group_col not in df.columns:
        raise ValueError(f"数据集中未找到分组列 '{group_col}'")
    if metric_col not in df.columns:
        raise ValueError(f"数据集中未找到待分析指标列 '{metric_col}'")

    control_series = df[df[group_col] == control_val][metric_col].dropna()
    treatment_series = df[df[group_col] == treatment_val][metric_col].dropna()

    logs.append(f"[1. 数据摄入完成] 对照组样本数: {len(control_series)}, 实验组样本数: {len(treatment_series)}")

    return {
        "step_logs": logs,
        "n_control": len(control_series),
        "n_treatment": len(treatment_series)
    }

def srm_diagnostic_node(state: ABTestState) -> Dict[str, Any]:
    """
    节点 2: 样本比例失衡 (SRM) 诊断门禁
    """
    logs = list(state.get("step_logs", []))
    logs.append("[2. SRM 诊断] 执行卡方拟合优度检验，审查分流公平性...")

    data_path = state.get("data_path")
    group_col = state.get("group_col", "group")
    metric_col = state.get("metric_col")
    control_val = state.get("control_val", "control")
    treatment_val = state.get("treatment_val", "treatment")
    target_ratio = state.get("target_ratio", (0.5, 0.5))

    df = pd.read_csv(data_path)
    n_c = len(df[df[group_col] == control_val][metric_col].dropna())
    n_t = len(df[df[group_col] == treatment_val][metric_col].dropna())

    srm_res = check_srm(n_c, n_t, target_ratio=target_ratio)
    
    if srm_res["is_srm_detected"]:
        logs.append(f"[2. SRM 严重异常] 🚨 卡方检验 p 值 ({srm_res['p_value']:.4e}) 显著低于阈值！触发前置阻断。")
        return {
            "step_logs": logs,
            "srm_result": srm_res,
            "aborted": True,
            "abort_reason": "检测到严重样本比例失衡 (SRM)"
        }
    else:
        logs.append(f"[2. SRM 检验通过] ✅ p 值为 {srm_res['p_value']:.4f}，分流正常无偏差。")
        return {
            "step_logs": logs,
            "srm_result": srm_res,
            "aborted": False
        }

def route_after_srm(state: ABTestState) -> str:
    """
    条件边: SRM 异常则直接跳到报告节点，正常则进入分布诊断节点
    """
    if state.get("aborted", False):
        return "synthesis_report_node"
    return "distribution_diagnostic_node"

def distribution_diagnostic_node(state: ABTestState) -> Dict[str, Any]:
    """
    节点 3: 分布假设前提自适应验证节点
    """
    logs = list(state.get("step_logs", []))
    logs.append("[3. 分布检验] 正在诊断样本正态性与方差齐性...")

    data_path = state.get("data_path")
    group_col = state.get("group_col", "group")
    metric_col = state.get("metric_col")
    control_val = state.get("control_val", "control")
    treatment_val = state.get("treatment_val", "treatment")

    df = pd.read_csv(data_path)
    control_series = df[df[group_col] == control_val][metric_col].dropna()
    treatment_series = df[df[group_col] == treatment_val][metric_col].dropna()

    norm_c = check_normality(control_series)
    norm_t = check_normality(treatment_series)
    levene_res = check_homoscedasticity(control_series, treatment_series)

    logs.append(
        f"[3. 分布检验完成] 对照组正态性: {norm_c['is_normal']} (p={norm_c['p_value']:.4f}), "
        f"实验组正态性: {norm_t['is_normal']} (p={norm_t['p_value']:.4f}), "
        f"方差齐性: {levene_res['is_homoscedastic']} (p={levene_res['p_value']:.4f})"
    )

    return {
        "step_logs": logs,
        "normality_control": norm_c,
        "normality_treatment": norm_t,
        "homoscedasticity_result": levene_res
    }

def hypothesis_test_node(state: ABTestState) -> Dict[str, Any]:
    """
    节点 4: 自适应假设检验与效应量计算
    """
    logs = list(state.get("step_logs", []))
    logs.append("[4. 统计推断] 智能体自适应选择最佳检验路径并执行推断...")

    data_path = state.get("data_path")
    group_col = state.get("group_col", "group")
    metric_col = state.get("metric_col")
    control_val = state.get("control_val", "control")
    treatment_val = state.get("treatment_val", "treatment")

    df = pd.read_csv(data_path)
    control_series = df[df[group_col] == control_val][metric_col].dropna()
    treatment_series = df[df[group_col] == treatment_val][metric_col].dropna()

    norm_c = state["normality_control"]["is_normal"]
    norm_t = state["normality_treatment"]["is_normal"]
    homo = state["homoscedasticity_result"]["is_homoscedastic"]

    test_res = run_adaptive_ab_test(
        control_series=control_series,
        treatment_series=treatment_series,
        normality_a=norm_c,
        normality_b=norm_t,
        homoscedasticity=homo
    )

    logs.append(
        f"[4. 推断完成] 选定算法: {test_res['method_name']}, "
        f"p值: {test_res['p_value']:.4e}, 统计显著: {test_res['is_significant']}, "
        f"Cohen's d: {test_res['effect_size']['cohens_d']}"
    )

    return {
        "step_logs": logs,
        "test_result": test_res
    }

def synthesis_report_node(state: ABTestState) -> Dict[str, Any]:
    """
    节点 5: 结构化决策报告升华生成
    """
    logs = list(state.get("step_logs", []))
    logs.append("[5. 报告生成] 正在综合数理推导与业务洞察构建专业决策报告...")

    report_markdown = generate_report(state)

    logs.append("[5. 报告完成] 决策报告生成完毕！")
    return {
        "step_logs": logs,
        "final_report": report_markdown
    }

def build_ab_test_graph():
    """
    构建并编译 LangGraph 状态图
    """
    workflow = StateGraph(ABTestState)

    # 添加所有处理节点
    workflow.add_node("data_ingestion_node", data_ingestion_node)
    workflow.add_node("srm_diagnostic_node", srm_diagnostic_node)
    workflow.add_node("distribution_diagnostic_node", distribution_diagnostic_node)
    workflow.add_node("hypothesis_test_node", hypothesis_test_node)
    workflow.add_node("synthesis_report_node", synthesis_report_node)

    # 设定起始入口
    workflow.set_entry_point("data_ingestion_node")

    # 边与拓扑关系
    workflow.add_edge("data_ingestion_node", "srm_diagnostic_node")

    # 核心条件分支：根据 SRM 状态决定继续推断还是阻断告警
    workflow.add_conditional_edges(
        "srm_diagnostic_node",
        route_after_srm,
        {
            "distribution_diagnostic_node": "distribution_diagnostic_node",
            "synthesis_report_node": "synthesis_report_node"
        }
    )

    workflow.add_edge("distribution_diagnostic_node", "hypothesis_test_node")
    workflow.add_edge("hypothesis_test_node", "synthesis_report_node")
    workflow.add_edge("synthesis_report_node", END)

    return workflow.compile()
