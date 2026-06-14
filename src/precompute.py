"""
预计算梅尔频谱图并保存为.npy文件
避免训练时反复从音频文件加载和计算
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS,
    SPECTROGRAM_SHAPE, DURATION, DATA_DIR
)
from src.data_loader import scan_dataset


def precompute_spectrograms(output_dir: str = None):
    """
    预计算所有音频的梅尔频谱图并保存

    参数:
        output_dir: 输出目录，默认为 data/spectrograms
    """
    if output_dir is None:
        output_dir = DATA_DIR / "spectrograms"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # 扫描数据集
    df = scan_dataset()
    print(f"\n开始预计算 {len(df)} 个音频的梅尔频谱图...")
    print(f"输出目录: {output_dir}")

    # 进度条
    success = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="预计算频谱图"):
        filepath = row["filepath"]
        genre = row["genre"]
        filename = Path(filepath).stem

        # 输出路径
        genre_dir = output_dir / genre
        genre_dir.mkdir(exist_ok=True)
        npy_path = genre_dir / f"{filename}.npy"

        # 如果已存在则跳过
        if npy_path.exists():
            success += 1
            continue

        try:
            # 加载音频
            y, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

            # 如果音频长度不足，补零
            target_len = int(SAMPLE_RATE * DURATION)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))

            # 提取梅尔频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Resize到固定尺寸
            from scipy.ndimage import zoom
            target_h, target_w = SPECTROGRAM_SHAPE
            h, w = mel_spec_db.shape
            zoom_h = target_h / h
            zoom_w = target_w / w
            mel_resized = zoom(mel_spec_db, (zoom_h, zoom_w))

            # 归一化到[0, 1]
            mel_min, mel_max = mel_resized.min(), mel_resized.max()
            if mel_max - mel_min > 0:
                mel_resized = (mel_resized - mel_min) / (mel_max - mel_min)

            # 保存
            np.save(str(npy_path), mel_resized.astype(np.float32))
            success += 1

        except Exception as e:
            print(f"\n  [!] 跳过: {filepath} ({e})")
            failed += 1

    print(f"\n[OK] 预计算完成: 成功 {success}, 失败 {failed}")
    print(f"  保存位置: {output_dir}")


if __name__ == "__main__":
    precompute_spectrograms()
