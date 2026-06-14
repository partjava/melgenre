"""
生成8个流派的三域分析对比图
每个流派一行：时域波形 | 频域频谱 | 梅尔频谱图
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

from src.config import GENRES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, RESULTS_DIR, DATA_DIR
from src.data_loader import scan_dataset

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_all_genres_three_domain(save_path=None):
    """
    生成8个流派的三域分析对比图

    返回:
        matplotlib Figure对象
    """
    # 扫描数据集
    df = scan_dataset()

    # 每个流派选一首
    samples = {}
    for genre in GENRES:
        genre_df = df[df["genre"] == genre]
        if len(genre_df) > 0:
            samples[genre] = genre_df.iloc[0]["filepath"]

    n_genres = len(samples)
    fig, axes = plt.subplots(n_genres, 3, figsize=(18, 4 * n_genres), dpi=100)

    for row_idx, (genre, filepath) in enumerate(samples.items()):
        print(f"处理 {genre}: {Path(filepath).name}")

        # 加载音频
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=15)

        # 时域波形
        ax = axes[row_idx, 0]
        time = np.arange(len(y)) / sr
        ax.plot(time, y, color="#2196F3", linewidth=0.5)
        ax.set_ylabel(genre, fontsize=11, fontweight="bold", rotation=0, labelpad=60)
        if row_idx == 0:
            ax.set_title("时域波形", fontsize=12, fontweight="bold")
        ax.set_xlim(0, len(y) / sr)
        ax.grid(True, alpha=0.3)
        if row_idx == n_genres - 1:
            ax.set_xlabel("时间 (秒)")

        # 频域频谱
        ax = axes[row_idx, 1]
        fft = np.fft.fft(y)
        magnitude = np.abs(fft[:len(fft)//2])
        freq = np.fft.fftfreq(len(y), 1/sr)[:len(fft)//2]
        ax.plot(freq, magnitude, color="#FF5722", linewidth=0.5)
        if row_idx == 0:
            ax.set_title("频域频谱", fontsize=12, fontweight="bold")
        ax.set_xlim(0, sr/2)
        ax.grid(True, alpha=0.3)
        if row_idx == n_genres - 1:
            ax.set_xlabel("频率 (Hz)")

        # 梅尔频谱图
        ax = axes[row_idx, 2]
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax, cmap="magma"
        )
        if row_idx == 0:
            ax.set_title("梅尔频谱图", fontsize=12, fontweight="bold")
        if row_idx == n_genres - 1:
            ax.set_xlabel("时间 (秒)")

    fig.suptitle("8个流派三域分析对比", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")

    return fig


if __name__ == "__main__":
    save_path = RESULTS_DIR / "all_genres_three_domain.png"
    plot_all_genres_three_domain(save_path)
    print("完成!")
