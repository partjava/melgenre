"""
可视化模块
提供音频信号、特征、实验结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, List, Dict

from src.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS,
    GENRES, FIGURE_DPI, FIGURE_SIZE, RESULTS_DIR
)

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

plt.switch_backend("Agg")  # 非交互式后端


def plot_waveform(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "音频波形",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制音频时域波形

    参数:
        y: 音频信号
        sr: 采样率
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=FIGURE_DPI, dpi=FIGURE_DPI)
    time = np.arange(len(y)) / sr
    ax.plot(time, y, color="#2196F3", linewidth=0.5)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title(title)
    ax.set_xlim(0, len(y) / sr)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_spectrum(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "频谱图",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制音频频域频谱

    参数:
        y: 音频信号
        sr: 采样率
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=FIGURE_DPI, dpi=FIGURE_DPI)

    # 计算FFT
    fft = np.fft.fft(y)
    magnitude = np.abs(fft[:len(fft)//2])
    freq = np.fft.fftfreq(len(y), 1/sr)[:len(fft)//2]

    ax.plot(freq, magnitude, color="#FF5722", linewidth=0.5)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    ax.set_xlim(0, sr/2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    title: str = "梅尔频谱图",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制梅尔频谱图（时频域）

    参数:
        y: 音频信号
        sr: 采样率
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=FIGURE_DPI, dpi=FIGURE_DPI)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_three_domain_analysis(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    genre: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    三域分析图：时域波形 + 频域频谱 + 时频域梅尔频谱图

    这是课程报告要求的核心可视化
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=FIGURE_DPI)

    # 时域 - 波形
    time = np.arange(len(y)) / sr
    axes[0].plot(time, y, color="#2196F3", linewidth=0.5)
    axes[0].set_xlabel("时间 (秒)")
    axes[0].set_ylabel("振幅")
    axes[0].set_title(f"[时域] 音频波形 - {genre}")
    axes[0].set_xlim(0, len(y) / sr)
    axes[0].grid(True, alpha=0.3)

    # 频域 - 频谱
    fft = np.fft.fft(y)
    magnitude = np.abs(fft[:len(fft)//2])
    freq = np.fft.fftfreq(len(y), 1/sr)[:len(fft)//2]
    axes[1].plot(freq, magnitude, color="#FF5722", linewidth=0.5)
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("幅度")
    axes[1].set_title(f"[频域] 频谱 - {genre}")
    axes[1].set_xlim(0, sr/2)
    axes[1].grid(True, alpha=0.3)

    # 时频域 - 梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=axes[2], cmap="magma"
    )
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")
    axes[2].set_title(f"[时频域] 梅尔频谱图 - {genre}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_feature_comparison(
    features_dict: Dict[str, List[np.ndarray]],
    feature_name: str,
    sr: int = SAMPLE_RATE,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    对比不同流派的某个特征

    参数:
        features_dict: {genre_name: [feature_array_1, feature_array_2, ...]}
        feature_name: 特征名称
        save_path: 保存路径
    """
    n_genres = len(features_dict)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=FIGURE_DPI)
    axes = axes.flatten()

    for idx, (genre, features) in enumerate(features_dict.items()):
        if idx >= 10:
            break
        ax = axes[idx]
        # 取平均特征
        mean_feat = np.mean(features, axis=0)
        if mean_feat.ndim == 2:
            img = ax.imshow(mean_feat, aspect="auto", origin="lower", cmap="magma")
            fig.colorbar(img, ax=ax)
        else:
            ax.plot(mean_feat)
        ax.set_title(genre, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"各流派 {feature_name} 对比", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list = None,
    title: str = "混淆矩阵",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    绘制混淆矩阵热力图

    参数:
        cm: 混淆矩阵 (num_classes, num_classes)
        labels: 标签名称列表
        title: 标题
        save_path: 保存路径
        normalize: 是否归一化
    """
    if labels is None:
        labels = GENRES

    fig, ax = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        vmin=0, vmax=1 if normalize else None,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制多模型对比柱状图

    参数:
        results: {model_name: {"accuracy": 0.8, "f1": 0.75, ...}}
        save_path: 保存路径
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
        # 在柱子上方显示数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("评估指标")
    ax.set_ylabel("分数")
    ax.set_title("模型性能对比")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(["准确率", "精确率", "召回率", "F1分数"])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制训练曲线（Loss和Accuracy）

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=FIGURE_DPI)

    epochs = range(1, len(train_losses) + 1)

    # Loss曲线
    ax1.plot(epochs, train_losses, "b-", label="训练损失")
    ax1.plot(epochs, val_losses, "r-", label="验证损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失")
    ax1.set_title("训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy曲线
    ax2.plot(epochs, train_accs, "b-", label="训练准确率")
    ax2.plot(epochs, val_accs, "r-", label="验证准确率")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率")
    ax2.set_title("训练/验证准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制特征重要性条形图

    参数:
        importances: 特征重要性数组
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        save_path: 保存路径
    """
    # 取前N个
    indices = np.argsort(importances)[-top_n:]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    ax.barh(range(len(top_names)), top_importances, color="#2196F3")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("特征重要性")
    ax.set_title(f"Top {top_n} 重要特征")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_genre_distribution(
    counts: Dict[str, int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制数据集流派分布柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)

    genres = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))

    bars = ax.bar(genres, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=10
        )

    ax.set_xlabel("流派")
    ax.set_ylabel("样本数量")
    ax.set_title("GTZAN数据集流派分布")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig
