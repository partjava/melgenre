"""
音频特征提取模块
提供完整的音频特征提取功能，覆盖时域、频域、时频域三个维度
"""

import numpy as np
import librosa
import librosa.display
from typing import Dict, Optional, Tuple

from src.config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC,
    FEATURE_PARAMS
)


def load_audio(
    filepath: str,
    sr: int = SAMPLE_RATE,
    duration: Optional[float] = 30,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件

    参数:
        filepath: 音频文件路径
        sr: 目标采样率
        duration: 截取时长（秒），None表示全部加载
        offset: 起始偏移（秒）

    返回:
        (音频信号数组, 采样率)
    """
    y, sr = librosa.load(filepath, sr=sr, duration=duration, offset=offset)
    return y, sr


# ==================== 时域特征 ====================

def extract_zero_crossing_rate(y: np.ndarray) -> np.ndarray:
    """提取过零率（时域特征）"""
    return librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]


def extract_rms_energy(y: np.ndarray) -> np.ndarray:
    """提取短时能量/RMS（时域特征）"""
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]


def extract_tempo(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """提取节拍速度BPM（时域特征）"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return float(tempo[0]) if len(tempo) > 0 else 0.0


# ==================== 频域特征 ====================

def extract_spectral_centroid(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """提取频谱质心（频域特征）"""
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]


def extract_spectral_bandwidth(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """提取频谱带宽（频域特征）"""
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]


def extract_spectral_rolloff(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """提取频谱滚降点（频域特征）"""
    return librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]


# ==================== 时频域特征 ====================

def extract_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """
    提取MFCC特征（时频域特征）

    返回: shape (n_mfcc, time_steps)
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # 加上一阶和二阶差分（delta特征）
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta_mfcc, delta2_mfcc])


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    提取梅尔频谱图（时频域特征）

    返回: shape (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # 转换为对数刻度（dB）
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_chroma(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """提取色度特征（时频域特征，反映音高/和弦信息）"""
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_spectral_contrast(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """提取频谱对比度（反映音色差异）"""
    return librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_tonnetz(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """提取调性网络特征（反映和声信息）"""
    y_harmonic = librosa.effects.harmonic(y)
    return librosa.feature.tonnetz(y=y_harmonic, sr=sr)


# ==================== 综合特征提取 ====================

def extract_statistical_features(feature: np.ndarray) -> np.ndarray:
    """
    对时间序列特征计算统计量，将其压缩为固定长度向量

    参数:
        feature: shape (n_features, time_steps)

    返回:
        统计特征向量: [mean, std, min, max, median] × n_features
    """
    return np.concatenate([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
        np.min(feature, axis=1),
        np.max(feature, axis=1),
        np.median(feature, axis=1),
    ])


def extract_all_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Dict[str, np.ndarray]:
    """
    提取所有音频特征

    参数:
        y: 音频信号
        sr: 采样率

    返回:
        字典，键为特征名称，值为特征数组
    """
    features = {}

    # 时域特征
    features["zcr"] = extract_zero_crossing_rate(y)
    features["rms"] = extract_rms_energy(y)
    features["tempo"] = np.array([extract_tempo(y, sr)])

    # 频域特征
    features["spectral_centroid"] = extract_spectral_centroid(y, sr)
    features["spectral_bandwidth"] = extract_spectral_bandwidth(y, sr)
    features["spectral_rolloff"] = extract_spectral_rolloff(y, sr)

    # 时频域特征
    features["mfcc"] = extract_mfcc(y, sr)
    features["mel_spectrogram"] = extract_mel_spectrogram(y, sr)
    features["chroma"] = extract_chroma(y, sr)
    features["spectral_contrast"] = extract_spectral_contrast(y, sr)
    features["tonnetz"] = extract_tonnetz(y, sr)

    return features


def extract_feature_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    提取固定长度的特征向量（用于传统ML模型）

    将各特征的统计量拼接为一维向量

    返回:
        一维特征向量
    """
    features = extract_all_features(y, sr)

    feature_vector = []
    # 跳过mel_spectrogram（维度太大，不适合传统ML）
    for name, feat in features.items():
        if name == "mel_spectrogram":
            continue
        if feat.ndim == 1:
            # 一维特征直接取统计量
            feature_vector.extend([
                np.mean(feat), np.std(feat),
                np.min(feat), np.max(feat), np.median(feat)
            ])
        else:
            # 二维特征对每个维度取统计量
            feature_vector.extend(extract_statistical_features(feat).tolist())

    return np.array(feature_vector, dtype=np.float32)


def get_feature_names() -> list:
    """返回特征向量中每个维度的名称（用于特征重要性分析）"""
    names = []

    # 时域
    for base in ["zcr", "rms"]:
        for stat in ["mean", "std", "min", "max", "median"]:
            names.append(f"{base}_{stat}")
    names.append("tempo")

    # 频域
    for base in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
        for stat in ["mean", "std", "min", "max", "median"]:
            names.append(f"{base}_{stat}")

    # MFCC (20个系数 × 3组 × 5个统计量)
    for prefix in ["mfcc", "delta_mfcc", "delta2_mfcc"]:
        for i in range(N_MFCC):
            for stat in ["mean", "std", "min", "max", "median"]:
                names.append(f"{prefix}_{i}_{stat}")

    # Chroma (12个音级)
    for i in range(12):
        for stat in ["mean", "std", "min", "max", "median"]:
            names.append(f"chroma_{i}_{stat}")

    # Spectral Contrast (7个频带)
    for i in range(7):
        for stat in ["mean", "std", "min", "max", "median"]:
            names.append(f"spectral_contrast_{i}_{stat}")

    # Tonnetz (6个维度)
    for i in range(6):
        for stat in ["mean", "std", "min", "max", "median"]:
            names.append(f"tonnetz_{i}_{stat}")

    return names
