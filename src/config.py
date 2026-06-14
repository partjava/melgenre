"""
全局配置文件（FMA-medium版本）
集中管理所有路径、音频参数、模型超参数
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FMA_DIR = DATA_DIR / "fma_small"           # FMA音频文件目录
FMA_METADATA = DATA_DIR / "fma_metadata"   # FMA元数据目录
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models_saved"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# 确保输出目录存在
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ==================== FMA-medium数据集配置 ====================
# 8个顶级流派（FMA-small，每类1000首，完全平衡）
GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]
NUM_GENRES = len(GENRES)
GENRE_TO_LABEL = {genre: idx for idx, genre in enumerate(GENRES)}
LABEL_TO_GENRE = {idx: genre for genre, idx in GENRE_TO_LABEL.items()}

# ==================== 音频参数 ====================
SAMPLE_RATE = 22050       # 采样率
DURATION = 30             # 音频时长（秒）
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_FFT = 2048              # FFT窗口大小
HOP_LENGTH = 512          # 帧移
N_MELS = 128              # 梅尔滤波器组数量
N_MFCC = 20               # MFCC系数数量

# ==================== 特征提取配置 ====================
FEATURE_PARAMS = {
    "n_mfcc": N_MFCC,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "n_chroma": 12,
    "n_contrast": 7,
    "n_tonnetz": 6,
}

# ==================== 深度学习配置 ====================
SPECTROGRAM_SHAPE = (128, 128)

DL_CONFIG = {
    "batch_size": 64,         # 数据量大了，batch可以开大
    "num_epochs": 150,
    "learning_rate": 0.0005,
    "weight_decay": 5e-4,
    "patience": 25,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
}

# 数据增强参数
AUGMENTATION = {
    "time_stretch_range": (0.7, 1.3),
    "pitch_shift_range": (-3, 3),
    "noise_level": 0.01,
}

# ==================== 传统ML配置 ====================
ML_CONFIG = {
    "cv_folds": 5,
    "random_state": 42,
}

# ==================== 可视化配置 ====================
FIGURE_DPI = 150
FIGURE_SIZE = (12, 6)

# 中文字体配置（Windows）
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
