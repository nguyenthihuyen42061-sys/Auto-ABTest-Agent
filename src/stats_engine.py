"""
纯统计学推断核心引擎 (Statistical Inference Engine)
严格遵循现代工业界 A/B 实验与数理统计标准：
1. SRM 卡方拟合优度检验
2. 分布正态性诊断（D'Agostino-Pearson / Shapiro-Wilk）
3. 方差齐性检验（Levene-Brown-Forsythe 稳健检验）
4. 自适应检验路由（Student's t-test, Welch's t-test, Mann-Whitney U, Bootstrap）
5. 效应量（Cohen's d、相对提升率、95% 置信区间）
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import ALPHA, SRM_ALPHA, NORMALITY_ALPHA, HOMOSCEDASTICITY_ALPHA, BOOTSTRAP_ROUNDS

def check_srm(
    n_control: int, 
    n_treatment: int, 
    target_ratio: Tuple[float, float] = (0.5, 0.5),
    alpha: float = SRM_ALPHA
) -> Dict[str, Any]:
    """
    样本比例失衡 (Sample Ratio Mismatch, SRM) 卡方检验
    检验实际观察分流样本量与理论设定分流比例是否存在显著性差异
    """
    total = n_control + n_treatment
    if total == 0:
        raise ValueError("总样本量不能为 0")

    p_c, p_t = target_ratio[0] / sum(target_ratio), target_ratio[1] / sum(target_ratio)
    expected_c = total * p_c
    expected_t = total * p_t

    observed = [n_control, n_treatment]
    expected = [expected_c, expected_t]

    # 卡方拟合优度检验 (自由度 df = 2 - 1 = 1)
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    is_srm_detected = bool(p_value < alpha)

    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "total_sample_size": total,
        "target_ratio": [round(p_c, 3), round(p_t, 3)],
        "actual_ratio": [round(n_control / total, 4), round(n_treatment / total, 4)],
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_srm_detected": is_srm_detected,
        "diagnosis": (
            "【严重警告】检测到显著的样本比例失衡 (SRM)！数据可能存在分流服务故障、埋点丢失或系统性崩溃，推断结果不可信！"
            if is_srm_detected else "SRM 检验通过：对照组与实验组样本分流符合预设比例。"
        )
    }

def check_normality(series: pd.Series, alpha: float = NORMALITY_ALPHA) -> Dict[str, Any]:
    """
    正态性诊断
    大样本 (n >= 5000) 采用 D'Agostino-Pearson 综合检验（结合偏度与峰度）
    中轻样本 (n < 5000) 采用 Shapiro-Wilk 检验
    """
    clean_series = series.dropna().to_numpy()
    n = len(clean_series)
    if n < 8:
        return {
            "n": n,
            "method": "Insufficient Sample",
            "stat": 0.0,
            "p_value": 0.0,
            "is_normal": False,
            "skewness": 0.0,
            "kurtosis": 0.0
        }

    skewness = float(stats.skew(clean_series))
    kurtosis = float(stats.kurtosis(clean_series))

    if n >= 5000:
        stat, p_value = stats.normaltest(clean_series)
        method = "D'Agostino-Pearson Omni-Bus Test"
    else:
        stat, p_value = stats.shapiro(clean_series)
        method = "Shapiro-Wilk Test"

    is_normal = bool(p_value >= alpha)
    return {
        "n": n,
        "method": method,
        "stat": float(stat),
        "p_value": float(p_value),
        "is_normal": is_normal,
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
        "alpha": alpha
    }

def check_homoscedasticity(
    control: pd.Series, 
    treatment: pd.Series, 
    alpha: float = HOMOSCEDASTICITY_ALPHA
) -> Dict[str, Any]:
    """
    方差齐性检验 (Levene 检验 - Brown-Forsythe 稳健变体)
    使用中位数作为离差中心，即使数据具有一定偏态，对方差齐性的推断也高度稳健
    """
    c_clean = control.dropna().to_numpy()
    t_clean = treatment.dropna().to_numpy()

    stat, p_value = stats.levene(c_clean, t_clean, center="median")
    is_homoscedastic = bool(p_value >= alpha)

    var_c = float(np.var(c_clean, ddof=1))
    var_t = float(np.var(t_clean, ddof=1))

    return {
        "method": "Levene Test (Brown-Forsythe Robust)",
        "stat": float(stat),
        "p_value": float(p_value),
        "is_homoscedastic": is_homoscedastic,
        "var_control": round(var_c, 4),
        "var_treatment": round(var_t, 4),
        "alpha": alpha
    }

def run_bootstrap_ci(
    control: np.ndarray, 
    treatment: np.ndarray, 
    n_boot: int = BOOTSTRAP_ROUNDS, 
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """
    非参数 Bootstrap 均值差异经验置信区间构建
    用于对经典参数检验结果进行互补印证
    """
    n_c, n_t = len(control), len(treatment)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        sample_c = np.random.choice(control, size=n_c, replace=True)
        sample_t = np.random.choice(treatment, size=n_t, replace=True)
        diffs[i] = np.mean(sample_t) - np.mean(sample_c)

    lower_pct = (1.0 - ci_level) / 2.0 * 100.0
    upper_pct = (1.0 + ci_level) / 2.0 * 100.0

    ci_lower = float(np.percentile(diffs, lower_pct))
    ci_upper = float(np.percentile(diffs, upper_pct))
    # 若 0 不在置信区间内，说明差异显著
    is_significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "method": f"Non-parametric Bootstrap (B={n_boot})",
        "mean_diff": float(np.mean(treatment) - np.mean(control)),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci_level,
        "is_significant": is_significant
    }

def calculate_effect_size(control: np.ndarray, treatment: np.ndarray) -> Dict[str, Any]:
    """
    效应量与业务指标相对提升率计算
    包含 Cohen's d、相对提升率 (Relative Lift) 与均值差标准误
    """
    mean_c, mean_t = float(np.mean(control)), float(np.mean(treatment))
    var_c, var_t = float(np.var(control, ddof=1)), float(np.var(treatment, ddof=1))
    n_c, n_t = len(control), len(treatment)

    mean_diff = mean_t - mean_c
    relative_lift = (mean_diff / mean_c * 100.0) if mean_c != 0 else 0.0

    # 合并方差 (Pooled Standard Deviation)
    pooled_sd = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    # 效应量量级解释 (Cohen, 1988 标准)
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "微弱效应 (Negligible / Small effect, |d| < 0.2)"
    elif abs_d < 0.5:
        interpretation = "小效应 (Small effect, 0.2 <= |d| < 0.5)"
    elif abs_d < 0.8:
        interpretation = "中等效应 (Medium effect, 0.5 <= |d| < 0.8)"
    else:
        interpretation = "强效应 (Large effect, |d| >= 0.8)"

    return {
        "mean_control": round(mean_c, 4),
        "mean_treatment": round(mean_t, 4),
        "mean_diff": round(mean_diff, 4),
        "relative_lift_pct": round(relative_lift, 3),
        "cohens_d": round(float(cohens_d), 4),
        "effect_interpretation": interpretation
    }

def run_adaptive_ab_test(
    control_series: pd.Series,
    treatment_series: pd.Series,
    normality_a: bool,
    normality_b: bool,
    homoscedasticity: bool,
    alpha: float = ALPHA
) -> Dict[str, Any]:
    """
    自适应假设检验决策与执行总路由：
    - 若两组均满足正态性且方差齐 -> Student's t-test
    - 若两组均满足正态性但方差不齐 -> Welch's t-test (Satterthwaite 近似)
    - 若任一组呈现偏态/非正态 -> Mann-Whitney U 秩和检验
    同时自动运行 Bootstrap 经验置信区间对比印证
    """
    c_arr = control_series.dropna().to_numpy()
    t_arr = treatment_series.dropna().to_numpy()

    both_normal = normality_a and normality_b

    if both_normal and homoscedasticity:
        # 标准两独立样本 t 检验
        stat, p_val = stats.ttest_ind(t_arr, c_arr, equal_var=True, alternative="two-sided")
        method_name = "独立双样本 Student's t-test (等方差假设)"
        reason = "两组样本经诊断均符合正态分布，且 Levene 方差齐性检验无显著差异。"
    elif both_normal and not homoscedasticity:
        # Welch's t 检验
        stat, p_val = stats.ttest_ind(t_arr, c_arr, equal_var=False, alternative="two-sided")
        method_name = "Welch's t-test (异方差校正 / Satterthwaite 近似)"
        reason = "两组样本满足正态性，但方差齐性未通过，采用 Welch 方差校正消除第一类错误膨胀。"
    else:
        # Mann-Whitney U 秩和非参数检验
        stat, p_val = stats.mannwhitneyu(t_arr, c_arr, alternative="two-sided")
        method_name = "Mann-Whitney U 非参数秩和检验"
        reason = "样本检验出显著的偏态或长尾特征，违背经典参数检验正态性前提，路由至秩和非参数检验以保证结论稳健性。"

    effect_metrics = calculate_effect_size(c_arr, t_arr)
    bootstrap_res = run_bootstrap_ci(c_arr, t_arr, n_boot=BOOTSTRAP_ROUNDS, ci_level=1.0 - alpha)

    is_significant = bool(p_val < alpha)

    return {
        "method_name": method_name,
        "routing_reason": reason,
        "test_statistic": float(stat),
        "p_value": float(p_val),
        "alpha": alpha,
        "is_significant": is_significant,
        "effect_size": effect_metrics,
        "bootstrap_validation": bootstrap_res
    }
