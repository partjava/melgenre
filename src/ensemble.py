"""
模型集成模块
将多个模型的预测结果进行投票/平均，提升分类准确率
"""

import numpy as np
import torch
from typing import Dict, List

from src.config import GENRES, MODELS_DIR, SPECTROGRAM_SHAPE
from src.models import get_model
from src.utils import load_sklearn_model
from src.feature_extraction import extract_feature_vector, load_audio


def load_all_models(device: torch.device = None):
    """加载所有训练好的模型"""
    if device is None:
        device = torch.device("cpu")

    models = {}

    # 传统ML模型
    for name in ["svm", "rf", "knn", "xgb"]:
        path = MODELS_DIR / f"{name}_model.pkl"
        if path.exists():
            models[name.upper()] = load_sklearn_model(str(path))

    # 加载scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        scaler = load_sklearn_model(str(scaler_path))

    # 深度学习模型
    for name in ["cnn", "crnn"]:
        path = MODELS_DIR / f"{name}_best.pth"
        if path.exists():
            model = get_model(name)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            model.to(device)
            models[name.upper()] = model

    return models, scaler


def predict_with_model(
    model,
    filepath: str,
    scaler=None,
    device: torch.device = None,
) -> np.ndarray:
    """
    用单个模型预测，返回概率分布

    返回: shape (num_classes,) 的概率数组
    """
    if device is None:
        device = torch.device("cpu")

    model_name = type(model).__name__

    # 深度学习模型
    if hasattr(model, 'conv_layers') or hasattr(model, 'cnn'):
        from src.dataset import PrecomputedGenreDataset
        from pathlib import Path
        import pandas as pd

        # 获取genre名
        track_id = Path(filepath).stem
        genre_dirs = list(Path(filepath).parent.parent.iterdir())
        genre = None
        for d in genre_dirs:
            if d.is_dir() and (Path(d) / f"{Path(filepath).stem}.npy").exists():
                genre = d.name
                break

        if genre is None:
            # 在线计算
            from scipy.ndimage import zoom
            y, _ = load_audio(filepath)
            import librosa
            mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, n_fft=2048, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            h, w = mel_db.shape
            mel_resized = zoom(mel_db, (SPECTROGRAM_SHAPE[0]/h, SPECTROGRAM_SHAPE[1]/w))
            mel_min, mel_max = mel_resized.min(), mel_resized.max()
            if mel_max - mel_min > 0:
                mel_resized = (mel_resized - mel_min) / (mel_max - mel_min)
            tensor = torch.FloatTensor(mel_resized).unsqueeze(0).unsqueeze(0).to(device)
        else:
            from src.config import DATA_DIR
            npy_path = DATA_DIR / "spectrograms" / genre / f"{track_id}.npy"
            if npy_path.exists():
                spec = np.load(str(npy_path))
                tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
            else:
                return np.zeros(len(GENRES))

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        return probs

    # 传统ML模型
    else:
        y, sr = load_audio(filepath)
        feat = extract_feature_vector(y, sr).reshape(1, -1)
        if scaler is not None:
            feat = scaler.transform(feat)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(feat)[0]
        else:
            pred = model.predict(feat)[0]
            probs = np.zeros(len(GENRES))
            probs[pred] = 1.0
            return probs


def ensemble_predict(
    filepath: str,
    models: Dict,
    scaler=None,
    device: torch.device = None,
    method: str = "average",
) -> tuple:
    """
    集成预测

    参数:
        filepath: 音频文件路径
        models: 模型字典
        scaler: 特征标准化器
        device: 计算设备
        method: "average" (概率平均) 或 "vote" (投票)

    返回:
        (预测标签, 各流派概率)
    """
    all_probs = {}

    for name, model in models.items():
        try:
            probs = predict_with_model(model, filepath, scaler, device)
            all_probs[name] = probs
        except Exception as e:
            print(f"  {name} 预测失败: {e}")
            continue

    if not all_probs:
        return -1, np.zeros(len(GENRES))

    if method == "average":
        # 概率平均
        avg_probs = np.mean(list(all_probs.values()), axis=0)
        pred = np.argmax(avg_probs)
        return pred, avg_probs

    elif method == "vote":
        # 多数投票
        votes = np.zeros(len(GENRES))
        for name, probs in all_probs.items():
            votes[np.argmax(probs)] += 1
        pred = np.argmax(votes)
        # 投票结果转为概率
        vote_probs = votes / votes.sum()
        return pred, vote_probs


def evaluate_ensemble(
    test_df,
    models: Dict,
    scaler=None,
    device: torch.device = None,
    method: str = "average",
) -> Dict:
    """
    在测试集上评估集成模型

    返回:
        评估指标字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    y_true = []
    y_pred = []

    total = len(test_df)
    for idx, row in test_df.iterrows():
        try:
            pred, _ = ensemble_predict(row["filepath"], models, scaler, device, method)
            y_true.append(row["label"])
            y_pred.append(pred)
        except Exception:
            continue

        if (len(y_true)) % 100 == 0:
            print(f"  进度: {len(y_true)}/{total}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


if __name__ == "__main__":
    import torch
    from src.data_loader import scan_dataset, split_dataset
    from src.utils import set_seed
    from src.visualize import plot_confusion_matrix, plot_model_comparison
    from src.config import RESULTS_DIR

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    df = scan_dataset()
    _, _, test_df = split_dataset(df)

    # 加载模型
    print("加载模型...")
    models, scaler = load_all_models(device)
    print(f"已加载 {len(models)} 个模型: {list(models.keys())}")

    # 评估集成模型
    print("\n评估集成模型（概率平均法）...")
    avg_metrics = evaluate_ensemble(test_df, models, scaler, device, method="average")
    print(f"  准确率: {avg_metrics['accuracy']:.4f}")
    print(f"  F1分数: {avg_metrics['f1']:.4f}")

    print("\n评估集成模型（投票法）...")
    vote_metrics = evaluate_ensemble(test_df, models, scaler, device, method="vote")
    print(f"  准确率: {vote_metrics['accuracy']:.4f}")
    print(f"  F1分数: {vote_metrics['f1']:.4f}")

    # 保存结果
    from src.utils import save_results, get_timestamp
    save_results({
        "ensemble_average": avg_metrics,
        "ensemble_vote": vote_metrics,
    }, f"ensemble_results_{get_timestamp()}.json")

    # 生成混淆矩阵
    plot_confusion_matrix(
        np.array(avg_metrics["confusion_matrix"]),
        title="集成模型混淆矩阵（概率平均）",
        save_path=str(RESULTS_DIR / "confusion_matrix_ensemble_avg.png")
    )
    plot_confusion_matrix(
        np.array(vote_metrics["confusion_matrix"]),
        title="集成模型混淆矩阵（投票法）",
        save_path=str(RESULTS_DIR / "confusion_matrix_ensemble_vote.png")
    )
    print("\n图表已保存到 results/ 目录")
