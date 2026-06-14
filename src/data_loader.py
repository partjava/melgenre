"""
数据加载模块（FMA-medium版本）
负责解析FMA元数据，构建文件路径与标签列表
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from src.config import (
    FMA_DIR, FMA_METADATA, GENRES, GENRE_TO_LABEL, LABEL_TO_GENRE,
    DL_CONFIG
)


def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """
    根据track_id获取音频文件路径

    FMA的目录结构:
        fma_medium/000/000002.mp3
        fma_medium/001/000034.mp3
        fma_medium/155/155000.mp3

    规则: track_id // 1000 为子目录名，文件名为6位补零的track_id
    """
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def load_fma_metadata(metadata_dir: Path = None) -> pd.DataFrame:
    """
    加载FMA的tracks.csv元数据

    返回:
        DataFrame，列: [track_id, genre_top, subset]
    """
    if metadata_dir is None:
        metadata_dir = FMA_METADATA

    tracks_path = metadata_dir / "tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"未找到 {tracks_path}\n"
            f"请下载fma_metadata.zip并解压到 {metadata_dir}\n"
            f"下载地址: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
        )

    # FMA的tracks.csv有两行header，需要特殊处理
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # 提取关键列
    df = pd.DataFrame({
        "track_id": tracks.index,
        "genre_top": tracks[("track", "genre_top")],
        "subset": tracks[("set", "subset")],
        "split": tracks[("set", "split")],
    })

    # 只保留small子集
    df = df[df["subset"] == "small"].copy()

    # 去掉没有流派标签的
    df = df.dropna(subset=["genre_top"]).copy()

    # 修正流派名称（斜杠在Windows路径中不允许）
    df["genre_top"] = df["genre_top"].str.replace(" / ", "-", regex=False)

    print(f"[OK] FMA-medium元数据加载完成:")
    print(f"  总曲目: {len(df)}")
    print(f"  流派数: {df['genre_top'].nunique()}")
    print(f"  流派列表: {sorted(df['genre_top'].unique())}")

    return df


def scan_dataset(audio_dir: Path = None, metadata_dir: Path = None) -> pd.DataFrame:
    """
    扫描FMA-medium数据集，返回包含文件路径和标签的DataFrame

    返回:
        DataFrame，列: [filepath, genre, label, track_id]
    """
    if audio_dir is None:
        audio_dir = FMA_DIR

    # 加载元数据
    df = load_fma_metadata(metadata_dir)

    # 构建文件路径
    records = []
    for _, row in df.iterrows():
        track_id = int(row["track_id"])
        genre = row["genre_top"]
        filepath = get_audio_path(audio_dir, track_id)

        # 检查文件是否存在
        if filepath.exists():
            records.append({
                "track_id": track_id,
                "filepath": str(filepath),
                "genre": genre,
                "label": GENRE_TO_LABEL.get(genre, -1),
            })

    result_df = pd.DataFrame(records)

    # 过滤掉未知流派
    result_df = result_df[result_df["label"] >= 0].reset_index(drop=True)

    if len(result_df) == 0:
        raise FileNotFoundError(
            f"未找到任何音频文件，请检查数据集路径: {audio_dir}\n"
            f"请下载FMA-medium数据集: https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
        )

    print(f"[OK] 扫描完成: 共找到 {len(result_df)} 个音频文件，{result_df['genre'].nunique()} 个流派")
    return result_df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_state: int = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据集按比例划分为训练集、验证集、测试集（分层抽样）
    """
    train_ratio = train_ratio or DL_CONFIG["train_ratio"]
    val_ratio = val_ratio or DL_CONFIG["val_ratio"]
    test_ratio = test_ratio or DL_CONFIG["test_ratio"]
    random_state = random_state or DL_CONFIG["random_state"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    from sklearn.model_selection import train_test_split

    stratify_col = df["label"] if stratify else None

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        random_state=random_state, stratify=stratify_col
    )

    stratify_temp = temp_df["label"] if stratify else None
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_adjusted),
        random_state=random_state, stratify=stratify_temp
    )

    print(f"[OK] 数据集划分完成:")
    print(f"  训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def get_genre_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """获取各流派的样本数量分布"""
    dist = df.groupby("genre").size().reset_index(name="count")
    dist = dist.sort_values("count", ascending=False).reset_index(drop=True)
    return dist


def print_dataset_summary(df: pd.DataFrame) -> None:
    """打印数据集摘要信息"""
    print("=" * 50)
    print("数据集摘要")
    print("=" * 50)
    print(f"总样本数: {len(df)}")
    print(f"流派数量: {df['genre'].nunique()}")
    print(f"流派列表: {', '.join(sorted(df['genre'].unique()))}")
    print(f"\n各流派样本分布:")
    dist = get_genre_distribution(df)
    for _, row in dist.iterrows():
        bar = "█" * (row["count"] // 100)
        print(f"  {row['genre']:>20s}: {row['count']:5d} {bar}")
    print("=" * 50)
