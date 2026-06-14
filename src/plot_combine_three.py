"""
合并时域波形图、频域频谱图、梅尔频谱图为一张图
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from src.config import RESULTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def combine_three(save_path=None):
    """将三张中文命名的图片合并为一张横排图"""
    img_paths = [
        RESULTS_DIR / "时域波形图.png",
        RESULTS_DIR / "频域频谱图.png",
        RESULTS_DIR / "梅尔频谱图.png",
    ]
    titles = ["时域波形", "频域频谱", "梅尔频谱图"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    for ax, path, title in zip(axes, img_paths, titles):
        if path.exists():
            img = mpimg.imread(str(path))
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle("音频三域分析", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")

    return fig


if __name__ == "__main__":
    save_path = RESULTS_DIR / "三域分析.png"
    combine_three(save_path)
    print("完成!")
