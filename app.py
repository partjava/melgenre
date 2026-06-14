"""
Streamlit Web Demo
音乐流派分类系统交互式演示

运行: streamlit run app.py
"""

import sys
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    GENRES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC,
    SPECTROGRAM_SHAPE, MODELS_DIR, DURATION
)
from src.models import get_model
from src.feature_extraction import extract_all_features
from src.utils import load_sklearn_model

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="[Music] 音乐流派智能分类系统",
    page_icon="[Music]",
    layout="wide",
)

st.title("[Music] 基于梅尔频谱与深度学习的音乐流派智能分类系统")
st.markdown("---")


# ==================== 模型加载 ====================
@st.cache_resource
def load_cnn_model():
    """加载CNN模型"""
    model_path = MODELS_DIR / "cnn_best.pth"
    if not model_path.exists():
        return None
    model = get_model("cnn")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_crnn_model():
    """加载CRNN模型"""
    model_path = MODELS_DIR / "crnn_best.pth"
    if not model_path.exists():
        return None
    model = get_model("crnn")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_sklearn_models():
    """加载sklearn模型"""
    models = {}
    model_files = {
        "SVM": "svm_model.pkl",
        "RandomForest": "rf_model.pkl",
        "KNN": "knn_model.pkl",
        "XGBoost": "xgb_model.pkl",
    }
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = load_sklearn_model(str(path))
    # 加载scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        scaler = load_sklearn_model(str(scaler_path))
    return models, scaler


def audio_to_melspec_tensor(y: np.ndarray) -> torch.Tensor:
    """将音频转换为梅尔频谱图tensor"""
    from scipy.ndimage import zoom

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize
    h, w = mel_spec_db.shape
    zoom_h = SPECTROGRAM_SHAPE[0] / h
    zoom_w = SPECTROGRAM_SHAPE[1] / w
    mel_resized = zoom(mel_spec_db, (zoom_h, zoom_w))

    # 归一化
    mel_min, mel_max = mel_resized.min(), mel_resized.max()
    if mel_max - mel_min > 0:
        mel_resized = (mel_resized - mel_min) / (mel_max - mel_min)

    tensor = torch.FloatTensor(mel_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


# ==================== 侧边栏 ====================
st.sidebar.header("[File] 上传音频")
uploaded_file = st.sidebar.file_uploader(
    "选择一个音频文件",
    type=["wav", "mp3", "ogg", "flac", "au"],
    help="支持 WAV, MP3, OGG, FLAC 等常见音频格式"
)

st.sidebar.markdown("---")
st.sidebar.header("[Settings] 设置")

model_choice = st.sidebar.selectbox(
    "选择分类模型",
    ["CNN", "CRNN", "SVM", "RandomForest", "KNN", "XGBoost"],
    index=0,
)

show_features = st.sidebar.checkbox("显示详细特征分析", value=True)
duration = st.sidebar.slider("分析时长 (秒)", 5, 30, 10)

# ==================== 主页面 ====================
if uploaded_file is not None:
    # 保存上传文件到临时位置
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 加载音频
    y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=duration)

    # ---- 音频信息 ----
    col1, col2, col3 = st.columns(3)
    col1.metric("采样率", f"{sr} Hz")
    col2.metric("时长", f"{len(y)/sr:.2f} 秒")
    col3.metric("采样点数", f"{len(y):,}")

    # ---- 三域分析 ----
    st.subheader("[Chart] 三域分析")

    tab1, tab2, tab3 = st.tabs(["[Time] 时域波形", "[Freq] 频域频谱", "[Mel] 时频域梅尔频谱图"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 3))
        time_axis = np.arange(len(y)) / sr
        ax.plot(time_axis, y, color="#2196F3", linewidth=0.5)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("振幅")
        ax.set_title("音频波形")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 3))
        fft = np.fft.fft(y)
        magnitude = np.abs(fft[:len(fft)//2])
        freq = np.fft.fftfreq(len(y), 1/sr)[:len(fft)//2]
        ax.plot(freq, magnitude, color="#FF5722", linewidth=0.5)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("幅度")
        ax.set_title("频谱")
        ax.set_xlim(0, sr/2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4))
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax, cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("梅尔频谱图")
        st.pyplot(fig)
        plt.close()

    # ---- 分类预测 ----
    st.subheader("[Target] 流派分类结果")

    # 提取特征用于sklearn模型
    if model_choice in ["CNN", "CRNN"]:
        model = load_cnn_model() if model_choice == "CNN" else load_crnn_model()
        if model is None:
            st.error(f"[X] {model_choice} 模型未找到，请先运行训练脚本")
        else:
            tensor = audio_to_melspec_tensor(y)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()

            # 显示结果
            sorted_idx = np.argsort(probs)[::-1]
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**预测流派: [Genre] {GENRES[sorted_idx[0]]}**")
                st.markdown(f"置信度: **{probs[sorted_idx[0]]*100:.1f}%**")

                # 各流派概率
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(GENRES))]
                bars = ax.barh(
                    [GENRES[i] for i in sorted_idx],
                    [probs[i] for i in sorted_idx],
                    color=colors
                )
                ax.set_xlabel("置信度")
                ax.set_title(f"{model_choice} 分类结果")
                for bar, idx in zip(bars, sorted_idx):
                    ax.text(
                        bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{probs[idx]*100:.1f}%", va="center", fontsize=9
                    )
                ax.set_xlim(0, 1.1)
                st.pyplot(fig)
                plt.close()

            with col2:
                # 概率分布饼图
                fig, ax = plt.subplots(figsize=(6, 6))
                top_n = 5
                top_idx = sorted_idx[:top_n]
                top_probs = probs[top_idx]
                other_prob = 1 - top_probs.sum()
                labels = [GENRES[i] for i in top_idx] + ["其他"]
                sizes = list(top_probs) + [other_prob]
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie)
                ax.set_title("流派置信度分布")
                st.pyplot(fig)
                plt.close()

    else:
        # sklearn模型
        models, scaler = load_sklearn_models()
        if model_choice not in models or models[model_choice] is None:
            st.error(f"[X] {model_choice} 模型未找到，请先运行训练脚本")
        else:
            features = extract_all_features(y, sr)
            from src.feature_extraction import extract_feature_vector
            feat_vector = extract_feature_vector(y, sr).reshape(1, -1)

            if model_choice in ["SVM", "KNN"] and scaler is not None:
                feat_vector = scaler.transform(feat_vector)

            model = models[model_choice]
            y_pred = model.predict(feat_vector)[0]

            # 如果模型支持概率预测
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_vector)[0]
                sorted_idx = np.argsort(probs)[::-1]

                st.markdown(f"**预测流派: [Genre] {GENRES[y_pred]}**")
                st.markdown(f"置信度: **{probs[y_pred]*100:.1f}%**")

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["#4CAF50" if i == y_pred else "#2196F3" for i in range(len(GENRES))]
                bars = ax.barh(
                    [GENRES[i] for i in sorted_idx],
                    [probs[i] for i in sorted_idx],
                    color=colors
                )
                ax.set_xlabel("置信度")
                ax.set_title(f"{model_choice} 分类结果")
                ax.set_xlim(0, 1.1)
                st.pyplot(fig)
                plt.close()
            else:
                st.markdown(f"**预测流派: [Genre] {GENRES[y_pred]}**")

    # ---- 详细特征分析 ----
    if show_features:
        st.subheader("[Analysis] 详细特征分析")
        features = extract_all_features(y, sr)

        col1, col2 = st.columns(2)
        with col1:
            # MFCC
            fig, ax = plt.subplots(figsize=(8, 3))
            img = ax.imshow(features["mfcc"][:20], aspect="auto", origin="lower", cmap="coolwarm")
            fig.colorbar(img, ax=ax)
            ax.set_title("MFCC特征")
            ax.set_xlabel("时间帧")
            ax.set_ylabel("MFCC系数")
            st.pyplot(fig)
            plt.close()

            # 色度特征
            fig, ax = plt.subplots(figsize=(8, 3))
            img = ax.imshow(features["chroma"], aspect="auto", origin="lower", cmap="coolwarm")
            fig.colorbar(img, ax=ax)
            ax.set_title("色度特征 (Chroma)")
            st.pyplot(fig)
            plt.close()

        with col2:
            # 过零率
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(features["zcr"], color="#9C27B0")
            ax.set_title(f"过零率 (均值: {np.mean(features['zcr']):.4f})")
            ax.set_xlabel("时间帧")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # RMS能量
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(features["rms"], color="#FF9800")
            ax.set_title(f"RMS能量 (均值: {np.mean(features['rms']):.4f})")
            ax.set_xlabel("时间帧")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # 清理临时文件
    os.unlink(tmp_path)

else:
    # 未上传文件时显示说明
    st.info("<<< 请在左侧侧边栏上传音频文件开始分析")

    st.markdown("""
    ### 系统功能

    本系统实现了基于梅尔频谱与深度学习的音乐流派智能分类，支持：

    - **6种分类模型**: CNN、CRNN、SVM、随机森林、KNN、XGBoost
    - **三域分析**: 时域波形、频域频谱、时频域梅尔频谱图
    - **多维特征**: MFCC、色度特征、频谱对比度、过零率、RMS能量等
    - **实时预测**: 上传音频即可获得分类结果与置信度

    ### 使用方法

    1. 在左侧上传音频文件（支持 WAV/MP3/OGG 等格式）
    2. 选择分类模型
    3. 查看三域分析结果和分类预测

    ### 技术栈

    | 模块 | 技术 |
    |------|------|
    | 音频处理 | librosa, soundfile |
    | 机器学习 | scikit-learn, xgboost |
    | 深度学习 | PyTorch (CNN/CRNN) |
    | Web应用 | Streamlit |
    """)

# 底部信息
st.markdown("---")
st.markdown(
    "<center>武汉晴川学院 · 音频大数据分析与应用 · 基于梅尔频谱与深度学习的音乐流派智能分类系统</center>",
    unsafe_allow_html=True
)
