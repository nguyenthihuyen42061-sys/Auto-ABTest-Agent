"""
命令行运行与测试脚本 (CLI Runner)
支持一键测试全部业务场景并校验 LangGraph 状态机流转正确性
"""

import os
import sys

# 兼容 Windows 终端 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent_workflow import build_ab_test_graph

def run_single_experiment(data_path: str, group_col: str, metric_col: str, user_query: str):
    print("=" * 70)
    print(f">> 正在启动 A/B 测试诊断智能体: {os.path.basename(data_path)}")
    print(f"   分析指标: {metric_col} | 业务目标: {user_query}")
    print("=" * 70)

    app = build_ab_test_graph()

    initial_state = {
        "data_path": data_path,
        "group_col": group_col,
        "metric_col": metric_col,
        "control_val": "control",
        "treatment_val": "treatment",
        "target_ratio": (0.5, 0.5),
        "user_query": user_query,
        "step_logs": []
    }

    final_state = app.invoke(initial_state)

    print("\n【执行日志轨迹】:")
    for log in final_state.get("step_logs", []):
        print(f"  {log}")

    print("\n【报告预览】:")
    report = final_state.get("final_report", "")
    lines = report.split("\n")
    for line in lines[:25]:
        print(f"  {line}")
    if len(lines) > 25:
        print("  ... (更多内容已收拢)")
    print("\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 场景 1: 长尾偏态观看时长
    path_1 = os.path.join(base_dir, "data", "ab_watch_duration_skewed.csv")
    if os.path.exists(path_1):
        run_single_experiment(
            data_path=path_1,
            group_col="group",
            metric_col="watch_duration",
            user_query="评估推荐流模型升级是否显著提升了用户停留时长"
        )

    # 场景 2: 方差不齐的满意度打分
    path_2 = os.path.join(base_dir, "data", "ab_satisfaction_heteroscedastic.csv")
    if os.path.exists(path_2):
        run_single_experiment(
            data_path=path_2,
            group_col="group",
            metric_col="satisfaction_score",
            user_query="评估 UI 视觉重构对用户满意度评分的影响"
        )

    # 场景 3: SRM 异常分流样本
    path_3 = os.path.join(base_dir, "data", "ab_srm_anomaly.csv")
    if os.path.exists(path_3):
        run_single_experiment(
            data_path=path_3,
            group_col="group",
            metric_col="revenue",
            user_query="评估新结算流程对人均客单价的提升效应"
        )

if __name__ == "__main__":
    main()
