"""
PyTorch Dataset模块
支持两种模式：
1. 预计算模式（快速）：从.npy文件加载预计算的梅尔频谱图
2. 在线计算模式（慢）：实时从WAV文件计算
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable
from pathlib import Path

from src.config import (
    SPECTROGRAM_SHAPE, DATA_DIR
)


class PrecomputedGenreDataset(Dataset):
    """
    预计算频谱图数据集（快速版）

    从预计算的.npy文件加载梅尔频谱图
    """

    def __init__(
        self,
        filepaths: list,
        labels: list,
        genres: list = None,
        spectrogram_shape: Tuple[int, int] = SPECTROGRAM_SHAPE,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.filepaths = list(filepaths)
        self.labels = list(labels)
        self.genres = list(genres) if genres is not None else None
        self.spectrogram_shape = spectrogram_shape
        self.augment = augment
        self.transform = transform

        # 预计算文件的根目录
        self.spec_dir = DATA_DIR / "spectrograms"

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        # 从.npy文件加载
        npy_path = self._get_npy_path(filepath, idx)
        if npy_path.exists():
            mel_spec = np.load(str(npy_path))
        else:
            # fallback: 如果没有预计算文件，返回零
            mel_spec = np.zeros(self.spectrogram_shape, dtype=np.float32)

        # 数据增强（简单的数组级增强）
        if self.augment:
            mel_spec = self._augment_spectrogram(mel_spec)

        # 转换为tensor，添加通道维度 [1, H, W]
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label

    def _get_npy_path(self, filepath: str, idx: int = 0) -> Path:
        """获取对应的.npy文件路径"""
        filepath = Path(filepath)
        filename = filepath.stem
        # 使用genre构造路径（预计算时按genre分目录保存）
        if self.genres is not None:
            genre = self.genres[idx]
        else:
            genre = filepath.parent.name
        return self.spec_dir / genre / f"{filename}.npy"

    def _augment_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        频谱图级数据增强（比音频级更快）
        - 随机时间偏移
        - 随机频率遮蔽
        - 添加噪声
        """
        import random

        # 随机时间偏移
        if random.random() < 0.5:
            shift = random.randint(-5, 5)
            spec = np.roll(spec, shift, axis=1)

        # 随机频率遮蔽 (SpecAugment)
        if random.random() < 0.3:
            f = random.randint(0, spec.shape[0] - 10)
            w = random.randint(1, 8)
            spec[f:f+w, :] = 0

        # 添加噪声
        if random.random() < 0.3:
            noise = np.random.randn(*spec.shape).astype(np.float32) * 0.01
            spec = spec + noise

        return spec


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    创建训练、验证、测试的DataLoader

    优先使用预计算的频谱图，如果没有则回退到在线计算

    返回:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    # 检查是否有预计算的频谱图
    spec_dir = DATA_DIR / "spectrograms"
    use_precomputed = spec_dir.exists() and any(spec_dir.rglob("*.npy"))

    if use_precomputed:
        print("[OK] 使用预计算频谱图（快速模式）")
        train_dataset = PrecomputedGenreDataset(
            train_df["filepath"].values,
            train_df["label"].values,
            genres=train_df["genre"].values,
            augment=False,
        )
        val_dataset = PrecomputedGenreDataset(
            val_df["filepath"].values,
            val_df["label"].values,
            genres=val_df["genre"].values,
            augment=False,
        )
        test_dataset = PrecomputedGenreDataset(
            test_df["filepath"].values,
            test_df["label"].values,
            genres=test_df["genre"].values,
            augment=False,
        )
    else:
        print("[!] 未找到预计算频谱图，使用在线计算模式（较慢）")
        print("  建议先运行: python -m src.precompute")
        # 回退到在线计算
        from src.dataset import OnlineGenreDataset
        train_dataset = OnlineGenreDataset(
            train_df["filepath"].values,
            train_df["label"].values,
            augment=True,
        )
        val_dataset = OnlineGenreDataset(
            val_df["filepath"].values,
            val_df["label"].values,
            augment=False,
        )
        test_dataset = OnlineGenreDataset(
            test_df["filepath"].values,
            test_df["label"].values,
            augment=False,
        )

    # 均衡采样：每个batch中各类别出现概率相等
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"[OK] DataLoader创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")

    return train_loader, val_loader, test_loader


class OnlineGenreDataset(Dataset):
    """
    在线计算频谱图数据集（慢速版，作为fallback）

    实时从WAV文件加载并计算梅尔频谱图
    """

    def __init__(
        self,
        filepaths: list,
        labels: list,
        spectrogram_shape: Tuple[int, int] = SPECTROGRAM_SHAPE,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        import librosa
        self.librosa = librosa

        self.filepaths = list(filepaths)
        self.labels = list(labels)
        self.spectrogram_shape = spectrogram_shape
        self.augment = augment
        self.transform = transform

        from src.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, DURATION
        self.sr = SAMPLE_RATE
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS
        self.duration = DURATION

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            y, _ = self.librosa.load(filepath, sr=self.sr, duration=self.duration)
        except Exception:
            y = np.zeros(int(self.sr * self.duration))

        if self.augment:
            y = self._augment_audio(y)

        mel_spec = self.librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = self.librosa.power_to_db(mel_spec, ref=np.max)

        from scipy.ndimage import zoom
        target_h, target_w = self.spectrogram_shape
        h, w = mel_spec_db.shape
        mel_resized = zoom(mel_spec_db, (target_h / h, target_w / w))

        mel_min, mel_max = mel_resized.min(), mel_resized.max()
        if mel_max - mel_min > 0:
            mel_resized = (mel_resized - mel_min) / (mel_max - mel_min)

        mel_tensor = torch.FloatTensor(mel_resized).unsqueeze(0)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label

    def _augment_audio(self, y: np.ndarray) -> np.ndarray:
        import random
        if random.random() < 0.5:
            rate = random.uniform(0.8, 1.2)
            y = self.librosa.effects.time_stretch(y, rate=rate)
            target_len = int(self.sr * self.duration)
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, max(0, target_len - len(y))))
        if random.random() < 0.5:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise
        return y
