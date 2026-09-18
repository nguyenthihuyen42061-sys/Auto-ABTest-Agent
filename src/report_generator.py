"""
决策报告生成引擎 (Report Synthesis Engine)
将底层 Python 确定的统计学事实升华构建为带有严谨数理推导 (LaTeX) 与商业决策建议的专业分析报告。
支持在线大模型调用与离线学术级报告模板两种模式（零报错兜底）。
"""

from typing import Dict, Any
import json
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

SYSTEM_PROMPT = """你是一位资深互联网数据科学专家与统计学导师。
你的任务是根据传入的确定性 A/B 测试统计检验结果（JSON），撰写一份学术严谨且契合高管业务决策的《A/B 实验统计诊断与商业推断报告》。

要求：
1. 恪守“统计数据绝对真实”原则，报告中的所有样本量、检验量、p 值、效应量必须严格以输入的数据为准，严禁捏造或二次计算；
2. 涉及数学公式、假设检验符号时，必须使用规范的 LaTeX 格式（如 $p < 0.05$，$H_0$，$t = \\dots$）；
3. 结构清晰，必须包含以下板块：
   - 【实验结论摘要 (Executive Summary)】：一句话给出明确决策结论（全量发布 / 放弃 / 迭代优化）。
   - 【数据质量与分流诊断 (SRM Check)】：阐述分流检验情况与卡方值。
   - 【分布特征与检验方法路由依据】：说明为什么选择该检验方法（正态性、方差齐性结果）。
   - 【核心推断与效应量分析】：详细呈现统计量、p值、Cohen's d、相对提升率及 95% 置信区间。
   - 【业务落地建议与风险提示】：给出产品/算法团队具体的落地建议与潜在长期效应跟踪建议。
4. 全文使用中文。
"""

def generate_offline_report(state: Dict[str, Any]) -> str:
    """
    离线学术级报告模板生成器（无需任何 API Key，本地直接渲染高标准 Markdown + LaTeX 报告）
    """
    if state.get("aborted", False):
        srm = state.get("srm_result", {})
        return f"""# ⚠️ A/B 实验分流系统故障告警与推断阻断报告

## 1. 核心判定与风险阻断 (Execution Aborted)
> [!CAUTION]
> **【紧急风控拦截】** 系统检测到严重的 **样本比例失衡 (Sample Ratio Mismatch, SRM)**，已主动阻断后续假设检验与指标比较流程！
> **根本原因**：当入组样本比例偏离理论设计时，指标均值差异极大概率源于系统性采样偏差（如特定版本崩溃、分流哈希漏洞），强行推断将导致严重的**辛普森悖论 (Simpson's Paradox)**。

## 2. SRM 统计检验指标明细
* **预设理论比例**：对照组 : 实验组 = ${srm.get('target_ratio', [0.5, 0.5])[0] * 100:.1f}\\% : {srm.get('target_ratio', [0.5, 0.5])[1] * 100:.1f}\\%$
* **实际入组样本**：对照组 $n_c = {srm.get('n_control', 0)}$，实验组 $n_t = {srm.get('n_treatment', 0)}$（实际比例 ${srm.get('actual_ratio', [0, 0])[0]*100:.2f}\\% : {srm.get('actual_ratio', [0, 0])[1]*100:.2f}\\%$）
* **卡方拟合优度检验**：
  $$\\chi^2 = {srm.get('chi2_stat', 0.0):.4f}, \\quad p\\text{{-value}} = {srm.get('p_value', 0.0):.4e}$$
* **统计学判定**：在显著性水准 $\\alpha = {srm.get('alpha', 0.01)}$ 下，卡方检验 $p < \\alpha$，拒绝“分流比例符合理论设计”的原假设 $H_0$。

## 3. 产研团队排查指引 (Action Items)
1. **客户端崩溃排查**：检查实验组版本是否存在低端设备 Crash 导致日志未上报；
2. **埋点触发时机比对**：确认分流标记曝光埋点与指标触发埋点是否时序倒挂；
3. **哈希分流服务审计**：检查分流网关的 MurmurHash / MD5 盐值分配机制。
"""

    srm = state.get("srm_result", {})
    norm_c = state.get("normality_control", {})
    norm_t = state.get("normality_treatment", {})
    levene = state.get("homoscedasticity_result", {})
    test_res = state.get("test_result", {})
    eff = test_res.get("effect_size", {})
    boot = test_res.get("bootstrap_validation", {})

    sig_badge = "✅ 差异具有统计显著性 ($p < \\alpha$)" if test_res.get("is_significant") else "❌ 差异未达到统计显著 ($p \\ge \\alpha$)"

    return f"""# 📊 A/B 实验统计推断与决策诊断报告

**分析指标**：`{state.get('metric_col')}` | **分组依据**：`{state.get('group_col')}`  
**业务意图**：{state.get('user_query', '评估实验组相较于对照组的业务表现差异')}

---

## 一、核心实验结论 (Executive Summary)
* **推断结论**：{sig_badge}
* **核心指标**：对照组均值 $\\bar{{X}}_c = {eff.get('mean_control', 0.0)}$，实验组均值 $\\bar{{X}}_t = {eff.get('mean_treatment', 0.0)}$。
* **相对提升**：实验组相对提升 **{eff.get('relative_lift_pct', 0.0):+.2f}\\%**（Cohen's $d = {eff.get('cohens_d', 0.0):.4f}$，属于 **{eff.get('effect_interpretation', '无')}**）。
* **决策建议**：{'【建议全量上线】实验组指标表现出统计显著且正向的业务效应，可推进该版本在全量用户推广。' if test_res.get('is_significant') and eff.get('mean_diff', 0) > 0 else '【建议保持观望或迭代】指标提升未达显著水平或具有回撤风险，暂不建议盲目全量。'}

---

## 二、数据质量与分流健全度 (SRM Check)
* **样本入组量**：对照组 $n_c = {srm.get('n_control', 0)}$，实验组 $n_t = {srm.get('n_treatment', 0)}$（总样本量 $N = {srm.get('total_sample_size', 0)}$）。
* **拟合优度卡方检验**：
  $$\\chi^2 = {srm.get('chi2_stat', 0.0):.4f}, \\quad p\\text{{-value}} = {srm.get('p_value', 0.0):.4f}$$
* **诊断结论**：$p \\ge {srm.get('alpha', 0.01)}$，未观察到样本比例失衡（SRM），流量分配无系统性偏差，实验具有因果可比性。

---

## 三、分布特征与自适应推断路由依据
* **正态性诊断**：
  * 对照组：$p = {norm_c.get('p_value', 0.0):.4f}$，偏度 $= {norm_c.get('skewness', 0.0)}$（{'符合正态' if norm_c.get('is_normal') else '非正态/偏态'}）；
  * 实验组：$p = {norm_t.get('p_value', 0.0):.4f}$，偏度 $= {norm_t.get('skewness', 0.0)}$（{'符合正态' if norm_t.get('is_normal') else '非正态/偏态'}）。
* **方差齐性诊断 (Levene 检验)**：
  $$W = {levene.get('stat', 0.0):.4f}, \\quad p\\text{{-value}} = {levene.get('p_value', 0.0):.4f} \\quad (\\sigma_c^2 = {levene.get('var_control', 0.0)}, \\sigma_t^2 = {levene.get('var_treatment', 0.0)})$$
* **智能体路由选择**：系统自动选定 **{test_res.get('method_name')}**。
  * *选型依据*：{test_res.get('routing_reason')}

---

## 四、核心统计推断与效应量评估
* **统计检验量**：统计量统计值 $= {test_res.get('test_statistic', 0.0):.4f}$，检验 $p\\text{{-value}} = {test_res.get('p_value', 0.0):.4e}$（设定显著性水平 $\\alpha = {test_res.get('alpha', 0.05)}$）。
* **效应量评估 (Effect Size)**：
  * 无量纲效应量 Cohen's $d$：
    $$d = \\frac{{\\bar{{X}}_t - \\bar{{X}}_c}}{{S_{{\\text{{pooled}}}}}} = {eff.get('cohens_d', 0.0):.4f}$$
  * 均值绝对差异：$\\Delta = {eff.get('mean_diff', 0.0):+.4f}$
* **非参数 Bootstrap 经验验证 ($B = 2000$)**：
  * 差异 95% 经验置信区间为 **$[{boot.get('ci_lower', 0.0)}, \\; {boot.get('ci_upper', 0.0)}]$**。
  * 该置信区间{'不包含 0，双重佐证了差异显著性' if boot.get('is_significant') else '包含 0，印证了统计不显著'}。

---

## 五、产研决策建议 (Next Steps)
1. **上线评估**：{'推进灰度放量至 100%' if test_res.get('is_significant') and eff.get('mean_diff', 0) > 0 else '暂停放量，深入下钻不同用户群体（如新老用户）的表现异质性'}；
2. **长期因果效应监测**：上线后需防范“新奇效应 (Novelty Effect)”，建议在全量 14 天后再次评估留存衰减情况。
"""

def generate_report(state: Dict[str, Any]) -> str:
    """
    根据配置决定调用大模型或生成高质量离线报告
    """
    # 若被 SRM 阻断，优先输出离线标准审计报告（严谨、明确）
    if state.get("aborted", False):
        return generate_offline_report(state)

    if not OPENAI_API_KEY:
        # 离线模式
        return generate_offline_report(state)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.3
        )

        context_data = {
            "group_col": state.get("group_col"),
            "metric_col": state.get("metric_col"),
            "user_query": state.get("user_query"),
            "srm_result": state.get("srm_result"),
            "normality_control": state.get("normality_control"),
            "normality_treatment": state.get("normality_treatment"),
            "homoscedasticity": state.get("homoscedasticity_result"),
            "test_result": state.get("test_result")
        }

        user_prompt = f"以下是系统计算的完整统计学真实数据（请在此事实基础上生成完整报告）：\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```"

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        return str(response.content)

    except Exception as e:
        # 网络或 Key 异常时自动无缝降级为离线学术模板
        offline_content = generate_offline_report(state)
        return f"{offline_content}\n\n> *(提示: 大模型在线润色连接异常 [{str(e)}]，系统已无缝降级为内置学术级确定性渲染引擎)*"
