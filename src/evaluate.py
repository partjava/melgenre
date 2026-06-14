"""
统一评估模块
汇总传统ML和深度学习的评估结果，生成对比报告
"""

import numpy as np
import json
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.config import GENRES, RESULTS_DIR
from src.visualize import (
    plot_confusion_matrix, plot_model_comparison,
    plot_training_curves, plot_feature_importance
)
from src.utils import save_results, get_timestamp


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算完整评估指标

    返回:
        {accuracy, precision, recall, f1, confusion_matrix, classification_report}
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class_report": classification_report(
            y_true, y_pred, target_names=GENRES, output_dict=True, zero_division=0
        ),
    }


def generate_full_report(
    ml_results: Optional[Dict[str, Dict]] = None,
    dl_results: Optional[Dict[str, Dict]] = None,
    save: bool = True,
) -> Dict:
    """
    生成完整的实验评估报告

    参数:
        ml_results: 传统ML结果 {model_name: {"y_pred": ..., "metrics": ...}}
        dl_results: 深度学习结果 {model_name: {"y_pred": ..., "metrics": ..., "history": ...}}
        save: 是否保存结果

    返回:
        完整报告字典
    """
    report = {
        "timestamp": get_timestamp(),
        "models": {},
    }

    all_metrics = {}

    # 汇总传统ML结果
    if ml_results:
        print("=" * 60)
        print("传统机器学习模型评估结果")
        print("=" * 60)
        for name, result in ml_results.items():
            metrics = result.get("metrics", {})
            all_metrics[name] = metrics
            print(f"\n{name}:")
            print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
            print(f"  精确率: {metrics.get('precision', 0):.4f}")
            print(f"  召回率: {metrics.get('recall', 0):.4f}")
            print(f"  F1分数: {metrics.get('f1', 0):.4f}")

    # 汇总深度学习结果
    if dl_results:
        print("\n" + "=" * 60)
        print("深度学习模型评估结果")
        print("=" * 60)
        for name, result in dl_results.items():
            metrics = result.get("metrics", {})
            all_metrics[name] = metrics
            print(f"\n{name}:")
            print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
            print(f"  精确率: {metrics.get('precision', 0):.4f}")
            print(f"  召回率: {metrics.get('recall', 0):.4f}")
            print(f"  F1分数: {metrics.get('f1', 0):.4f}")

    report["models"] = all_metrics

    # 生成可视化
    print("\n生成可视化图表...")

    # 1. 模型对比图
    if len(all_metrics) > 1:
        comparison_metrics = {
            name: {k: v for k, v in m.items() if k in ["accuracy", "precision", "recall", "f1"]}
            for name, m in all_metrics.items()
        }
        plot_model_comparison(
            comparison_metrics,
            save_path=str(RESULTS_DIR / "model_comparison.png")
        )
        print("  [OK] 模型对比图已保存")

    # 2. 各模型混淆矩阵
    for name, result in {**(ml_results or {}), **(dl_results or {})}.items():
        metrics = result.get("metrics", {})
        cm = metrics.get("confusion_matrix")
        if cm:
            cm_array = np.array(cm)
            plot_confusion_matrix(
                cm_array,
                title=f"{name} 混淆矩阵",
                save_path=str(RESULTS_DIR / f"confusion_matrix_{name}.png")
            )
            print(f"  [OK] {name} 混淆矩阵已保存")

    # 3. 深度学习训练曲线
    if dl_results:
        for name, result in dl_results.items():
            history = result.get("history", {})
            if history:
                plot_training_curves(
                    history.get("train_losses", []),
                    history.get("val_losses", []),
                    history.get("train_accs", []),
                    history.get("val_accs", []),
                    save_path=str(RESULTS_DIR / f"training_curves_{name}.png")
                )
                print(f"  [OK] {name} 训练曲线已保存")

    # 4. 特征重要性（如果有RandomForest结果）
    if ml_results and "RandomForest" in ml_results:
        rf_result = ml_results["RandomForest"]
        if "feature_importances" in rf_result:
            from src.feature_extraction import get_feature_names
            feature_names = get_feature_names()
            importances = rf_result["feature_importances"]
            if len(feature_names) == len(importances):
                plot_feature_importance(
                    importances, feature_names,
                    save_path=str(RESULTS_DIR / "feature_importance.png")
                )
                print("  [OK] 特征重要性图已保存")

    # 保存报告
    if save:
        # 移除不可序列化的对象
        serializable_report = {
            "timestamp": report["timestamp"],
            "models": {}
        }
        for name, metrics in report["models"].items():
            serializable_report["models"][name] = {
                k: v for k, v in metrics.items()
                if k != "per_class_report" or isinstance(v, (dict, list, str, int, float))
            }
        save_results(serializable_report, f"full_report_{get_timestamp()}.json")

    return report


def print_per_class_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> None:
    """打印每个流派的详细分类结果"""
    print(f"\n{model_name} 各流派分类详情:")
    print(f"{'流派':>10s} | {'精确率':>8s} | {'召回率':>8s} | {'F1':>8s} | {'支持数':>6s}")
    print("-" * 50)

    report = classification_report(
        y_true, y_pred, target_names=GENRES, output_dict=True, zero_division=0
    )

    for genre in GENRES:
        if genre in report:
            r = report[genre]
            print(f"{genre:>10s} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | "
                  f"{r['f1-score']:>8.4f} | {int(r['support']):>6d}")


def find_confused_pairs(cm: np.ndarray, top_n: int = 5) -> List[Dict]:
    """
    找出最容易混淆的流派对

    返回:
        [{"pair": ("rock", "pop"), "confusion_rate": 0.35}, ...]
    """
    n = cm.shape[0]
    pairs = []

    for i in range(n):
        for j in range(n):
            if i != j:
                total_i = cm[i].sum()
                if total_i > 0:
                    rate = cm[i][j] / total_i
                    if rate > 0.05:  # 只关注混淆率>5%的
                        pairs.append({
                            "true_genre": GENRES[i],
                            "pred_genre": GENRES[j],
                            "confusion_rate": float(rate),
                            "count": int(cm[i][j]),
                        })

    # 按混淆率排序
    pairs.sort(key=lambda x: x["confusion_rate"], reverse=True)
    return pairs[:top_n]
