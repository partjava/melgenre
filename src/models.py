"""
深度学习模型定义模块
定义CNN和CRNN两种音乐流派分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.config import NUM_GENRES, SPECTROGRAM_SHAPE


class GenreCNN(nn.Module):
    """
    卷积神经网络音乐流派分类器

    针对FMA-medium (17000首) 优化：约300万参数

    输入: 梅尔频谱图 (batch, 1, 128, 128)
    输出: 流派类别概率 (batch, num_classes)
    """

    def __init__(self, num_classes: int = NUM_GENRES, dropout: float = 0.5):
        super(GenreCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第1层: 1 → 32通道
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # 第2层: 32 → 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # 第3层: 64 → 128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # 第4层: 128 → 256通道
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 (batch, 1, 128, 128)

        返回:
            logits (batch, num_classes)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class GenreCRNN(nn.Module):
    """
    CNN + LSTM 混合模型

    利用CNN提取空间特征，LSTM捕捉时序依赖关系
    比纯CNN更适合处理音频这种时序信号

    输入: 梅尔频谱图 (batch, 1, 128, 128)
    输出: 流派类别概率 (batch, 10)
    """

    def __init__(
        self,
        num_classes: int = NUM_GENRES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(GenreCRNN, self).__init__()

        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(0.3),
        )

        # 计算CNN输出的时间步维度
        self.rnn_input_size = self._get_rnn_input_size()

        # 双向LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),  # *2 因为双向
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _get_rnn_input_size(self) -> int:
        """计算CNN输出后送入RNN的特征维度"""
        dummy = torch.zeros(1, 1, *SPECTROGRAM_SHAPE)
        cnn_out = self.cnn(dummy)
        # cnn_out shape: (batch, channels, freq, time)
        b, c, f, t = cnn_out.shape
        return c * f  # 将通道和频率维度合并为RNN输入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量 (batch, 1, 128, 128)

        返回:
            logits (batch, num_classes)
        """
        # CNN特征提取
        cnn_out = self.cnn(x)  # (batch, channels, freq, time)
        b, c, f, t = cnn_out.shape

        # 重排为序列: (batch, time, channels*freq)
        rnn_in = cnn_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # LSTM时序建模
        rnn_out, _ = self.rnn(rnn_in)  # (batch, time, hidden*2)

        # 取最后时间步的输出
        rnn_out = rnn_out[:, -1, :]  # (batch, hidden*2)

        # 分类
        output = self.classifier(rnn_out)
        return output


def get_model(
    model_name: str,
    num_classes: int = NUM_GENRES,
    **kwargs
) -> nn.Module:
    """
    模型工厂函数

    参数:
        model_name: 模型名称 ("cnn" 或 "crnn")
        num_classes: 分类数量

    返回:
        PyTorch模型实例
    """
    models = {
        "cnn": GenreCNN,
        "crnn": GenreCRNN,
    }

    if model_name not in models:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(models.keys())}")

    model = models[model_name](num_classes=num_classes, **kwargs)
    return model
