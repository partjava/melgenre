"""
合并6个模型的混淆矩阵到一张图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from src.config import GENRES, RESULTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_confusion_matrices():
    """从结果文件中加载所有模型的混淆矩阵"""
    # 找到最新的结果文件
    result_files = list(RESULTS_DIR.glob("full_report_*.json"))
    if not result_files:
        print("未找到结果文件")
        return {}

    latest = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"加载结果: {latest}")

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    cms = {}
    for model_name, metrics in data.get("models", {}).items():
        cm = metrics.get("confusion_matrix")
        if cm:
            cms[model_name] = np.array(cm)

    return cms


def plot_combined_confusion_matrices(cms, save_path=None):
    """
    绘制6个模型的混淆矩阵合并图

    参数:
        cms: {model_name: confusion_matrix}
        save_path: 保存路径
    """
    models = list(cms.keys())
    n = len(models)

    # 2行3列布局
    rows = (n + 2) // 3
    cols = min(n, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 12), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 根据模型的标签数量动态获取genre名称
    for idx, (model_name, cm) in enumerate(cms.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        n_classes = cm.shape[0]
        # 根据类别数选择标签
        if n_classes == len(GENRES):
            labels = GENRES
        else:
            labels = [f"Class {i}" for i in range(n_classes)]

        # 归一化
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        # 绘制热力图
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, vmin=0, vmax=1,
            cbar=False,
            annot_kws={"size": 6},
        )

        # 计算准确率
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{model_name} (Acc: {acc:.1%})", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # 旋转标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # 隐藏多余的子图
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    fig.suptitle("各模型混淆矩阵对比", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")

    return fig


if __name__ == "__main__":
    cms = load_confusion_matrices()
    if cms:
        save_path = RESULTS_DIR / "confusion_matrix_all.png"
        plot_combined_confusion_matrices(cms, save_path)
        print("完成!")
    else:
        print("没有找到混淆矩阵数据")
