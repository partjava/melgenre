"""
深度学习模型训练模块
实现CNN和CRNN模型的训练，包含早停、学习率调度、数据增强
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from src.config import DL_CONFIG, NUM_GENRES, RESULTS_DIR, MODELS_DIR
from src.models import get_model, GenreCNN, GenreCRNN
from src.dataset import create_dataloaders
from src.utils import set_seed, get_device, save_model, count_parameters, format_time


class EarlyStopping:
    """早停机制，防止过拟合"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    训练一个epoch

    返回:
        (平均损失, 准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    验证集评估

    返回:
        (平均损失, 准确率)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = None,
    learning_rate: float = None,
    patience: int = None,
    class_weights: torch.Tensor = None,
) -> Dict:
    """
    训练深度学习模型

    参数:
        model_name: 模型名称 ("cnn" 或 "crnn")
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        num_epochs: 训练轮数
        learning_rate: 学习率
        patience: 早停耐心值

    返回:
        训练结果字典
    """
    num_epochs = num_epochs or DL_CONFIG["num_epochs"]
    learning_rate = learning_rate or DL_CONFIG["learning_rate"]
    patience = patience or DL_CONFIG["patience"]

    # 创建模型
    model = get_model(model_name).to(device)
    print(f"\n{'='*60}")
    print(f"训练 {model_name.upper()} 模型")
    print(f"{'='*60}")
    print(f"参数量: {count_parameters(model):,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DL_CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience)

    # 训练记录
    history = {
        "train_losses": [], "val_losses": [],
        "train_accs": [], "val_accs": [],
        "learning_rates": [],
    }
    best_val_acc = 0.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)
        history["learning_rates"].append(current_lr)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model(model, MODELS_DIR / f"{model_name}_best.pth")
            best_marker = " *"
        else:
            best_marker = ""

        # 打印进度（每轮都打印）
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (num_epochs - epoch)
        print(f"  Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s | ETA: {eta:.0f}s{best_marker}")

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n  早停触发于 Epoch {epoch}，最佳Epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time

    # 加载最佳模型
    model.load_state_dict(
        torch.load(MODELS_DIR / f"{model_name}_best.pth", map_location=device)
    )

    print(f"\n  训练完成! 耗时: {format_time(elapsed)}")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "elapsed": elapsed,
    }


def evaluate_dl_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    在测试集上评估深度学习模型

    返回:
        (真实标签数组, 预测标签数组, 指标字典)
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return y_true, y_pred, metrics


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """根据训练集分布计算类别权重（反比例权重，小类权重高）"""
    from src.config import NUM_GENRES
    label_counts = torch.zeros(NUM_GENRES)
    for _, labels in train_loader:
        for label in labels:
            label_counts[label] += 1
    # 反比例权重：样本越少权重越高
    weights = 1.0 / (label_counts + 1e-6)
    weights = weights / weights.sum() * NUM_GENRES  # 归一化
    return weights


def train_all_dl_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    训练所有深度学习模型

    返回:
        {model_name: {"model": ..., "history": ..., "metrics": ...}}
    """
    all_results = {}

    # 训练CNN
    cnn_result = train_model("cnn", train_loader, val_loader, device)
    y_true, y_pred, metrics = evaluate_dl_model(cnn_result["model"], test_loader, device)
    cnn_result["metrics"] = metrics
    cnn_result["y_true"] = y_true
    cnn_result["y_pred"] = y_pred
    all_results["CNN"] = cnn_result

    # 训练CRNN
    crnn_result = train_model("crnn", train_loader, val_loader, device)
    y_true, y_pred, metrics = evaluate_dl_model(crnn_result["model"], test_loader, device)
    crnn_result["metrics"] = metrics
    crnn_result["y_true"] = y_true
    crnn_result["y_pred"] = y_pred
    all_results["CRNN"] = crnn_result

    # 汇总
    print(f"\n{'='*60}")
    print("深度学习模型测试结果汇总:")
    print(f"{'='*60}")
    print(f"{'模型':>8s} | {'准确率':>8s} | {'精确率':>8s} | {'召回率':>8s} | {'F1':>8s}")
    print("-" * 50)
    for name, result in all_results.items():
        m = result["metrics"]
        print(f"{name:>8s} | {m['accuracy']:>8.4f} | {m['precision']:>8.4f} | "
              f"{m['recall']:>8.4f} | {m['f1']:>8.4f}")

    return all_results


if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    from src.data_loader import scan_dataset, split_dataset

    df = scan_dataset()
    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=DL_CONFIG["batch_size"]
    )

    results = train_all_dl_models(train_loader, val_loader, test_loader, device)
