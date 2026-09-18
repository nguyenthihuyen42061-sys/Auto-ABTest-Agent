"""
项目全局配置与超参数管理
支持通过环境变量读取 API Key 与服务地址，并提供统计学默认检验水准
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 大模型配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# 统计学超参数
ALPHA = float(os.getenv("ALPHA", "0.05"))             # 假设检验显著性水平 (默认 0.05)
SRM_ALPHA = float(os.getenv("SRM_ALPHA", "0.01"))     # SRM 卡方检验显著性水平 (更严格，通常设为 0.01 或 0.001)
BOOTSTRAP_ROUNDS = int(os.getenv("BOOTSTRAP_ROUNDS", "2000")) # Bootstrap 重抽样次数
NORMALITY_ALPHA = 0.05                               # 正态性检验显著性水平
HOMOSCEDASTICITY_ALPHA = 0.05                        # 方差齐性检验显著性水平
