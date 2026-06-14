"""
合并CNN和CRNN的训练曲线图片到一张图
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from src.config import RESULTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def combine_training_curves(save_path=None):
    """将CNN和CRNN的训练曲线图片合并为一张图"""
    cnn_path = RESULTS_DIR / "training_curves_CNN.png"
    crnn_path = RESULTS_DIR / "training_curves_CRNN.png"

    if not cnn_path.exists() or not crnn_path.exists():
        print(f"未找到训练曲线图片")
        return

    # 读取图片
    cnn_img = mpimg.imread(str(cnn_path))
    crnn_img = mpimg.imread(str(crnn_path))

    # 上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

    ax1.imshow(cnn_img)
    ax1.set_title("CNN 训练曲线", fontsize=13, fontweight="bold", pad=10)
    ax1.axis("off")

    ax2.imshow(crnn_img)
    ax2.set_title("CRNN 训练曲线", fontsize=13, fontweight="bold", pad=10)
    ax2.axis("off")

    fig.suptitle("深度学习模型训练曲线对比", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")

    return fig


if __name__ == "__main__":
    save_path = RESULTS_DIR / "training_curves_all.png"
    combine_training_curves(save_path)
    print("完成!")
