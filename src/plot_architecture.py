"""
生成系统架构流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.config import RESULTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_architecture(save_path=None):
    """绘制系统架构流程图"""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 定义颜色
    colors = {
        "data": "#E3F2FD",
        "feature": "#FFF3E0",
        "ml": "#E8F5E9",
        "dl": "#F3E5F5",
        "output": "#FFEBEE",
        "border": {
            "data": "#1565C0",
            "feature": "#E65100",
            "ml": "#2E7D32",
            "dl": "#6A1B9A",
            "output": "#C62828",
        }
    }

    def draw_box(x, y, w, h, text, color_key, fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=colors[color_key],
            edgecolor=colors["border"][color_key],
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, color="#333333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # 第一行：数据层
    draw_box(0.5, 6.5, 2.5, 1, "FMA-small\n8000首/8类", "data")
    draw_box(3.5, 6.5, 2.5, 1, "预计算\n频谱图", "data")
    draw_box(6.5, 6.5, 2.5, 1, "传统特征\n提取(455维)", "feature")
    draw_box(9.5, 6.5, 2.5, 1, "均衡采样\nWeightedSampler", "data")

    # 第二行：模型层
    draw_box(0.5, 4.2, 2, 1, "SVM", "ml")
    draw_box(2.8, 4.2, 2, 1, "RF", "ml")
    draw_box(5.1, 4.2, 2, 1, "KNN", "ml")
    draw_box(7.4, 4.2, 2, 1, "XGBoost", "ml")
    draw_box(10, 4.2, 1.5, 1, "CNN", "dl")
    draw_box(11.8, 4.2, 1.5, 1, "CRNN", "dl")

    # 第三行：输出层
    draw_box(3, 2, 3, 1, "模型对比\n混淆矩阵", "output")
    draw_box(7, 2, 3, 1, "训练曲线\n特征分析", "output")
    draw_box(11, 2, 2.5, 1, "Web Demo\nStreamlit", "output")

    # 标题
    ax.text(7, 7.9, "系统架构流程图", ha="center", va="center",
            fontsize=16, fontweight="bold")

    # 箭头连接
    # 数据 → 特征
    draw_arrow(3.0, 7.0, 3.5, 7.0)
    draw_arrow(6.0, 7.0, 6.5, 7.0)
    draw_arrow(9.0, 7.0, 9.5, 7.0)

    # 数据 → 模型
    draw_arrow(1.75, 6.5, 1.75, 5.2)  # 特征 → SVM
    draw_arrow(4.5, 6.5, 4.5, 5.2)    # 特征 → KNN (approx)
    draw_arrow(7.75, 6.5, 7.75, 5.2)  # XGBoost

    # 传统ML → 输出
    draw_arrow(2.5, 4.2, 3.5, 3.0)
    draw_arrow(5, 4.2, 4.5, 3.0)

    # DL → 输出
    draw_arrow(10.75, 4.2, 8.5, 3.0)
    draw_arrow(12.5, 4.2, 12.25, 3.0)

    # 输出 → Web
    draw_arrow(10, 2.5, 11, 2.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")

    return fig


if __name__ == "__main__":
    save_path = RESULTS_DIR / "architecture.png"
    plot_architecture(save_path)
    print("完成!")
