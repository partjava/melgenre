"""
传统机器学习模型训练模块
实现SVM、随机森林、KNN、XGBoost四个模型的训练与评估
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

from src.config import ML_CONFIG, GENRES, RESULTS_DIR, MODELS_DIR
from src.data_loader import scan_dataset, split_dataset
from src.feature_extraction import extract_feature_vector, get_feature_names, load_audio
from src.utils import set_seed, save_sklearn_model, save_results, get_timestamp


def extract_features_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    从DataFrame中批量提取特征向量

    参数:
        df: 包含filepath和label列的DataFrame

    返回:
        (特征矩阵 X, 标签数组 y)
    """
    features = []
    labels = []
    total = len(df)

    for idx, row in df.iterrows():
        try:
            y_audio, sr = load_audio(row["filepath"])
            feat = extract_feature_vector(y_audio, sr)
            features.append(feat)
            labels.append(row["label"])
        except Exception as e:
            print(f"  [!] 跳过文件 {row['filepath']}: {e}")
            continue

        # 进度显示
        processed = len(features)
        if processed % 50 == 0 or processed == total:
            print(f"  特征提取进度: {processed}/{total} ({processed/total*100:.1f}%)")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"[OK] 特征提取完成: X shape={X.shape}, y shape={y.shape}")
    return X, y


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """训练SVM模型"""
    print("\n--- 训练SVM ---")
    start = time.time()

    model = SVC(
        kernel="rbf", C=10, gamma="scale",
        random_state=ML_CONFIG["random_state"],
        probability=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start

    print(f"  SVM 准确率: {acc:.4f} (耗时: {elapsed:.1f}秒)")
    save_sklearn_model(model, MODELS_DIR / "svm_model.pkl")

    return {
        "model_name": "SVM",
        "model": model,
        "y_pred": y_pred,
        "elapsed": elapsed,
    }


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """训练随机森林模型"""
    print("\n--- 训练随机森林 ---")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=ML_CONFIG["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start

    print(f"  随机森林 准确率: {acc:.4f} (耗时: {elapsed:.1f}秒)")
    save_sklearn_model(model, MODELS_DIR / "rf_model.pkl")

    return {
        "model_name": "RandomForest",
        "model": model,
        "y_pred": y_pred,
        "elapsed": elapsed,
        "feature_importances": model.feature_importances_,
    }


def train_knn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """训练KNN模型"""
    print("\n--- 训练KNN ---")
    start = time.time()

    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start

    print(f"  KNN 准确率: {acc:.4f} (耗时: {elapsed:.1f}秒)")
    save_sklearn_model(model, MODELS_DIR / "knn_model.pkl")

    return {
        "model_name": "KNN",
        "model": model,
        "y_pred": y_pred,
        "elapsed": elapsed,
    }


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """训练XGBoost模型"""
    print("\n--- 训练XGBoost ---")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=ML_CONFIG["random_state"],
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start

    print(f"  XGBoost 准确率: {acc:.4f} (耗时: {elapsed:.1f}秒)")
    save_sklearn_model(model, MODELS_DIR / "xgb_model.pkl")

    return {
        "model_name": "XGBoost",
        "model": model,
        "y_pred": y_pred,
        "elapsed": elapsed,
        "feature_importances": model.feature_importances_,
    }


def train_all_ml_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    训练所有传统ML模型并返回结果

    返回:
        {model_name: {"model": ..., "y_pred": ..., "metrics": ...}}
    """
    print("=" * 60)
    print("传统机器学习模型训练")
    print("=" * 60)

    # 提取特征
    print("\n[1/5] 提取训练集特征...")
    X_train, y_train = extract_features_from_df(train_df)

    print("\n[2/5] 提取测试集特征...")
    X_test, y_test = extract_features_from_df(test_df)

    # 特征标准化
    print("\n[3/5] 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_sklearn_model(scaler, MODELS_DIR / "scaler.pkl")

    # 训练各模型
    print("\n[4/5] 训练模型...")
    all_results = {}

    # SVM
    svm_result = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    svm_result["metrics"] = compute_metrics(y_test, svm_result["y_pred"])
    all_results["SVM"] = svm_result

    # Random Forest（使用原始特征，不需要标准化）
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    rf_result["metrics"] = compute_metrics(y_test, rf_result["y_pred"])
    all_results["RandomForest"] = rf_result

    # KNN
    knn_result = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    knn_result["metrics"] = compute_metrics(y_test, knn_result["y_pred"])
    all_results["KNN"] = knn_result

    # XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_result["metrics"] = compute_metrics(y_test, xgb_result["y_pred"])
    all_results["XGBoost"] = xgb_result

    # 汇总
    print("\n[5/5] 结果汇总:")
    print(f"{'模型':>12s} | {'准确率':>8s} | {'精确率':>8s} | {'召回率':>8s} | {'F1':>8s} | {'耗时':>8s}")
    print("-" * 65)
    for name, result in all_results.items():
        m = result["metrics"]
        print(f"{name:>12s} | {m['accuracy']:>8.4f} | {m['precision']:>8.4f} | "
              f"{m['recall']:>8.4f} | {m['f1']:>8.4f} | {result['elapsed']:>7.1f}s")

    return all_results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算分类评估指标"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = None,
) -> Dict[str, Dict]:
    """
    对各模型进行交叉验证

    返回:
        {model_name: {"mean_accuracy": ..., "std_accuracy": ..., "scores": [...]}}
    """
    n_folds = n_folds or ML_CONFIG["cv_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=ML_CONFIG["random_state"])

    models = {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    }

    cv_results = {}
    for name, model in models.items():
        print(f"  {name} 交叉验证中...")
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_results[name] = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "scores": scores.tolist(),
        }
        print(f"    {n_folds}折CV准确率: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_results


if __name__ == "__main__":
    set_seed(42)

    # 扫描数据集
    df = scan_dataset()

    # 划分数据集
    train_df, val_df, test_df = split_dataset(df)

    # 训练所有模型
    results = train_all_ml_models(train_df, test_df)

    # 保存结果
    save_results(
        {name: {k: v for k, v in r.items() if k not in ["model"]}
         for name, r in results.items()},
        f"ml_results_{get_timestamp()}.json"
    )
