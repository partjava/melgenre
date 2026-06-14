"""
工具函数模块
"""

import os
import json
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.config import MODELS_DIR, RESULTS_DIR, DL_CONFIG


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[OK] 随机种子已设置: {seed}")


def get_device() -> torch.device:
    """获取计算设备（GPU/CPU）"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[OK] 使用CPU")
    return device


def save_model(model, filepath: str) -> None:
    """保存PyTorch模型"""
    torch.save(model.state_dict(), filepath)
    print(f"[OK] 模型已保存: {filepath}")


def load_model(model, filepath: str, device: torch.device = None) -> None:
    """加载PyTorch模型"""
    if device is None:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"[OK] 模型已加载: {filepath}")


def save_sklearn_model(model, filepath: str) -> None:
    """保存sklearn模型"""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] 模型已保存: {filepath}")


def load_sklearn_model(filepath: str):
    """加载sklearn模型"""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] 模型已加载: {filepath}")
    return model


def save_results(results: dict, filename: str) -> None:
    """保存实验结果为JSON"""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OK] 结果已保存: {filepath}")


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"
