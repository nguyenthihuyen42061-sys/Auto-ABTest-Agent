"""
LangGraph 状态定义模块 (State Contract)
定义在整个 A/B 测试诊断状态流中传递的数据契约 (TypedDict)
"""

from typing import TypedDict, Optional, Dict, Any, List, Tuple

class ABTestState(TypedDict, total=False):
    """
    A/B 测试智能体核心状态
    """
    # 1. 用户输入与数据基础信息
    data_path: str                           # 数据集 CSV 路径
    group_col: str                           # 分组列名 (如 'group')
    metric_col: str                          # 待分析的指标列名 (如 'watch_duration')
    control_val: str                         # 对照组标签值 (如 'control')
    treatment_val: str                       # 实验组标签值 (如 'treatment')
    target_ratio: Tuple[float, float]        # 理论分流比例 (如 (0.5, 0.5))
    user_query: Optional[str]                # 业务问题描述
    n_control: Optional[int]                 # 对照组有效样本数
    n_treatment: Optional[int]               # 实验组有效样本数

    # 2. 中间运行轨迹与过程日志
    step_logs: List[str]                     # 状态流转执行日志列表

    # 3. 统计诊断结果缓存
    srm_result: Optional[Dict[str, Any]]     # SRM 卡方检验结果
    normality_control: Optional[Dict[str, Any]]   # 对照组正态性检验结果
    normality_treatment: Optional[Dict[str, Any]] # 实验组正态性检验结果
    homoscedasticity_result: Optional[Dict[str, Any]] # 方差齐性检验结果
    test_result: Optional[Dict[str, Any]]    # 假设检验最终结果

    # 4. 流程控制与阻断标识
    aborted: bool                            # 是否被前置风控拦截阻断
    abort_reason: Optional[str]              # 阻断原因

    # 5. 产出报告
    final_report: Optional[str]              # 最终生成的 Markdown/LaTeX 决策报告
