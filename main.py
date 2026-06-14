"""
主运行脚本
一键运行完整的音乐流派分类实验流程

使用方法:
    python main.py              # 运行全部实验
    python main.py --ml-only    # 只运行传统ML
    python main.py --dl-only    # 只运行深度学习
    python main.py --demo       # 启动Web Demo
"""

import sys
import argparse
import time

from src.utils import set_seed, get_device, format_time, get_timestamp
from src.config import DL_CONFIG, RESULTS_DIR


def run_ml_experiment():
    """运行传统机器学习实验"""
    from src.data_loader import scan_dataset, split_dataset, print_dataset_summary
    from src.train_ml import train_all_ml_models
    from src.evaluate import generate_full_report

    print("\n" + "=" * 60)
    print("阶段1: 传统机器学习模型训练")
    print("=" * 60)

    # 数据准备
    df = scan_dataset()
    print_dataset_summary(df)
    train_df, val_df, test_df = split_dataset(df)

    # 训练
    ml_results = train_all_ml_models(train_df, test_df)

    return ml_results


def run_dl_experiment():
    """运行深度学习实验"""
    from src.data_loader import scan_dataset, split_dataset
    from src.dataset import create_dataloaders
    from src.train_dl import train_all_dl_models

    print("\n" + "=" * 60)
    print("阶段2: 深度学习模型训练")
    print("=" * 60)

    device = get_device()

    # 数据准备
    df = scan_dataset()
    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=DL_CONFIG["batch_size"]
    )

    # 训练
    dl_results = train_all_dl_models(train_loader, val_loader, test_loader, device)

    return dl_results


def run_full_experiment():
    """运行完整实验流程"""
    set_seed(42)
    start_time = time.time()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  基于梅尔频谱与深度学习的音乐流派智能分类系统              ║")
    print("║  完整实验流程                                              ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # 阶段1: 传统ML
    ml_results = run_ml_experiment()

    # 阶段2: 深度学习
    dl_results = run_dl_experiment()

    # 阶段3: 生成完整报告
    print("\n" + "=" * 60)
    print("阶段3: 生成实验报告")
    print("=" * 60)

    from src.evaluate import generate_full_report
    report = generate_full_report(ml_results, dl_results)

    # 完成
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[OK] 全部实验完成! 总耗时: {format_time(total_time)}")
    print(f"[OK] 结果保存在: {RESULTS_DIR}")
    print(f"[OK] 可视化图表已保存到 results/ 目录")
    print(f"\n启动Web Demo: streamlit run app.py")
    print(f"{'='*60}")

    return report


def launch_demo():
    """启动Streamlit Web Demo"""
    import subprocess
    print("启动 Streamlit Web Demo...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    parser = argparse.ArgumentParser(description="音乐流派智能分类系统")
    parser.add_argument("--ml-only", action="store_true", help="只运行传统ML实验")
    parser.add_argument("--dl-only", action="store_true", help="只运行深度学习实验")
    parser.add_argument("--demo", action="store_true", help="启动Web Demo")
    args = parser.parse_args()

    if args.demo:
        launch_demo()
    elif args.ml_only:
        set_seed(42)
        run_ml_experiment()
    elif args.dl_only:
        set_seed(42)
        run_dl_experiment()
    else:
        run_full_experiment()


if __name__ == "__main__":
    main()
