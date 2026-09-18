"""
A/B 测试模拟数据生成脚本
用于模拟互联网工业界常见的业务场景：
1. 场景 A（停留时长/收益指标）：长尾偏态对数正态分布 (Log-normal)，模拟留存、观看时长或消费金额。
2. 场景 B（常规业务指标）：近似正态分布指标，模拟人均浏览 PV 或打分。
3. 场景 C（分流异常场景）：样本比例失衡 (Sample Ratio Mismatch, SRM) 模拟分流故障。
"""

import os
import numpy as np
import pandas as pd

def generate_datasets(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 场景 1: 长尾偏态数据（用户单日停留时长，单位：分钟）
    # 模拟对照组与实验组，实验组人均时长略有提升，呈现右偏长尾
    # -------------------------------------------------------------
    n_control_1 = 5000
    n_test_1 = 5000
    
    # 对照组: mu=2.5, sigma=0.8 (中位数约12.18分钟，有长尾极值)
    control_duration = np.random.lognormal(mean=2.5, sigma=0.8, size=n_control_1)
    # 实验组: 提升算法推荐精度，分布右移 (mu=2.58, sigma=0.8)
    test_duration = np.random.lognormal(mean=2.58, sigma=0.8, size=n_test_1)
    
    df_scene_1 = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_control_1 + n_test_1)],
        "group": ["control"] * n_control_1 + ["treatment"] * n_test_1,
        "watch_duration": np.concatenate([control_duration, test_duration]).round(2)
    })
    path_1 = os.path.join(output_dir, "ab_watch_duration_skewed.csv")
    df_scene_1.to_csv(path_1, index=False, encoding="utf-8")
    print(f"[生成完成] 场景1（长尾偏态数据）: {path_1}，样本量: {len(df_scene_1)}")

    # -------------------------------------------------------------
    # 场景 2: 近似正态分布指标（页面加载打分/人均活跃度得分）
    # 模拟对照组与实验组方差不齐（Heteroscedasticity）
    # -------------------------------------------------------------
    n_control_2 = 4000
    n_test_2 = 4000
    
    control_score = np.random.normal(loc=75.0, scale=10.0, size=n_control_2)
    test_score = np.random.normal(loc=77.2, scale=12.5, size=n_test_2) # 均值提升但方差增大
    
    df_scene_2 = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_control_2 + n_test_2)],
        "group": ["control"] * n_control_2 + ["treatment"] * n_test_2,
        "satisfaction_score": np.concatenate([control_score, test_score]).round(2)
    })
    path_2 = os.path.join(output_dir, "ab_satisfaction_heteroscedastic.csv")
    df_scene_2.to_csv(path_2, index=False, encoding="utf-8")
    print(f"[生成完成] 场景2（方差不齐近似正态数据）: {path_2}，样本量: {len(df_scene_2)}")

    # -------------------------------------------------------------
    # 场景 3: 分流故障样本（SRM 样本比例严重失衡）
    # 理论设计 1:1 分流，实际由于某机型 crash 导致实验组样本严重缺失 (53% vs 47%)
    # -------------------------------------------------------------
    n_control_3 = 10600
    n_test_3 = 9400  # 卡方检验 p 值将远小于 0.001
    
    control_conv = np.random.normal(loc=50.0, scale=8.0, size=n_control_3)
    test_conv = np.random.normal(loc=52.0, scale=8.0, size=n_test_3)
    
    df_scene_3 = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_control_3 + n_test_3)],
        "group": ["control"] * n_control_3 + ["treatment"] * n_test_3,
        "revenue": np.concatenate([control_conv, test_conv]).round(2)
    })
    path_3 = os.path.join(output_dir, "ab_srm_anomaly.csv")
    df_scene_3.to_csv(path_3, index=False, encoding="utf-8")
    print(f"[生成完成] 场景3（SRM 分流异常数据）: {path_3}，样本量: {len(df_scene_3)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_datasets(current_dir)
