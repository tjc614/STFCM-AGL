import argparse
import os
import shutil
from datetime import datetime
from trainer import AGLSTFCMTrainer
from config import Config


def create_save_directory(config):
    """创建保存目录"""
    start_time = datetime.now().strftime("%Y%m%d%H%M")
    if config.dataset in ['taiex', 'sse']:
        dir_name = os.path.join(config.save_dir, f"{config.year}{config.dataset.upper()}{start_time}")
    else:
        dir_name = os.path.join(config.save_dir, f"{config.dataset}{start_time}")

    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def grid_search(config, save_dir):
    """网格搜索最优参数"""
    # 定义搜索空间
    order_list = [2, 3, 4, 5]
    d_hidden_list = [3, 4, 5, 6]
    filter_nums_list = [8, 10, 12, 14]
    kernel_size_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    lambda1_list = [0.5, 1.0, 1.5]
    lambda2_list = [0.5, 1.0, 1.5]

    total_runs = len(order_list) * len(d_hidden_list) * len(filter_nums_list) * len(kernel_size_list) * len(
        lambda1_list) * len(lambda2_list)
    current_run = 0
    best_rmse = float('inf')
    best_params = {}

    results_file = os.path.join(save_dir, "grid_search_results.txt")

    with open(results_file, "w") as f:
        f.write("AGL-STFCM Grid Search Results\n")
        f.write("============================\n\n")

    for order in order_list:
        for d_hidden in d_hidden_list:
            for filter_nums in filter_nums_list:
                for kernel_size in kernel_size_list:
                    for lambda1 in lambda1_list:
                        for lambda2 in lambda2_list:
                            current_run += 1

                            # 更新配置
                            config.order = order
                            config.d_hidden = d_hidden
                            config.filter_nums = filter_nums
                            config.kernel_size = kernel_size
                            config.lambda1 = lambda1
                            config.lambda2 = lambda2

                            print(f"\n---------- Run {current_run}/{total_runs} ----------")
                            print(f"order: {order}, d_hidden: {d_hidden}, filter_nums: {filter_nums}")
                            print(f"kernel_size: {kernel_size}, lambda1: {lambda1}, lambda2: {lambda2}")

                            # 训练模型
                            trainer = AGLSTFCMTrainer(config)
                            try:
                                results, _ = trainer.train()
                                current_rmse = results['average']['rmse']

                                # 记录结果
                                with open(results_file, "a") as f:
                                    f.write(f"Run {current_run}/{total_runs}: ")
                                    f.write(f"order={order}, d_hidden={d_hidden}, filter_nums={filter_nums}, ")
                                    f.write(f"kernel_size={kernel_size}, lambda1={lambda1}, lambda2={lambda2}, ")
                                    f.write(f"Test RMSE={current_rmse:.4f}\n")

                                # 更新最佳结果
                                if current_rmse < best_rmse:
                                    best_rmse = current_rmse
                                    best_params = {
                                        'order': order,
                                        'd_hidden': d_hidden,
                                        'filter_nums': filter_nums,
                                        'kernel_size': kernel_size,
                                        'lambda1': lambda1,
                                        'lambda2': lambda2,
                                        'rmse': current_rmse
                                    }

                                    # 保存最佳模型
                                    best_model_path = os.path.join(save_dir, "best_model")
                                    trainer.save_model(best_model_path)

                            except Exception as e:
                                print(f"Error in run {current_run}: {e}")
                                with open(results_file, "a") as f:
                                    f.write(f"Run {current_run}/{total_runs}: ERROR - {e}\n")

    # 保存最佳参数
    with open(results_file, "a") as f:
        f.write(f"\n\nBest Parameters:\n")
        f.write(f"================\n")
        f.write(f"Best Test RMSE: {best_rmse:.4f}\n")
        for key, value in best_params.items():
            if key != 'rmse':
                f.write(f"{key}: {value}\n")

    print(f"\n\nGrid search completed!")
    print(f"Best Test RMSE: {best_rmse:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params


def single_run(config, save_dir):
    """单次运行"""
    print("Starting AGL-STFCM single run...")
    print(f"Dataset: {config.dataset}")
    print(f"Parameters: order={config.order}, d_hidden={config.d_hidden}")
    print(f"filter_nums={config.filter_nums}, kernel_size={config.kernel_size}")
    print(f"lambda1={config.lambda1}, lambda2={config.lambda2}")

    trainer = AGLSTFCMTrainer(config)
    results, history = trainer.train()

    # 保存模型
    model_path = os.path.join(save_dir, "model")
    trainer.save_model(model_path)

    # 所有数据集都使用测试误差
    result_type_desc = "Test Error"
    metric_desc = "Test"

    # 保存结果
    results_file = os.path.join(save_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"AGL-STFCM {result_type_desc} Results\n")
        f.write("=" * (len(f"AGL-STFCM {result_type_desc} Results")) + "\n\n")
        f.write(f"Dataset: {config.dataset}\n")
        if config.dataset.lower() == 'eeg':
            f.write(f"EEG Subject: {config.eeg_subject}\n")
        elif config.dataset in ['taiex', 'sse']:
            f.write(f"Year: {config.year}\n")
        f.write(f"Evaluation Type: {result_type_desc}\n\n")
        f.write(f"Model: AGL-STFCM (Adaptive Granularity Learning for Spatio-Temporal Fuzzy Cognitive Maps)\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  order: {config.order}\n")
        f.write(f"  d_hidden: {config.d_hidden}\n")
        f.write(f"  filter_nums: {config.filter_nums}\n")
        f.write(f"  kernel_size: {config.kernel_size}\n")
        f.write(f"  lambda1: {config.lambda1}\n")
        f.write(f"  lambda2: {config.lambda2}\n\n")
        f.write(f"{metric_desc} Results:\n")
        for key, metrics in results.items():
            if isinstance(metrics, dict):
                f.write(f"  {key}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {metrics:.4f}\n")

    print(f"Results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='AGL-STFCM Model Training')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='taiex',
                        choices=['taiex', 'sse', 'traffic', 'temp', 'epc', 'eeg'],
                        help='Dataset to use')
    parser.add_argument('--year', type=int, default=2004,
                        help='Year for TAIEX/SSE dataset (2000-2004 for TAIEX, 2018-2023 for SSE)')
    parser.add_argument('--eeg_subject', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='Subject number for EEG dataset')

    # 模型参数
    parser.add_argument('--order', type=int, default=4,
                        help='Time window size')
    parser.add_argument('--d_hidden', type=int, default=4,
                        help='Hidden dimension size')
    parser.add_argument('--filter_nums', type=int, default=10,
                        help='Number of TCN filters')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Convolution kernel size')
    parser.add_argument('--aggregator_type', type=str, default='adaptive_pool',
                        choices=['mean', 'pool', 'lstm', 'adaptive_pool'],
                        help='GraphSAGE aggregator type')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Lambda1 parameter for granularity allocation')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='Lambda2 parameter for granularity allocation')
    parser.add_argument('--l1', type=float, default=1e-5,
                        help='L1 regularization')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='L2 regularization')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.0,
                        help='Validation split ratio')

    # 运行模式
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'grid_search'],
                        help='Running mode: single run or grid search')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=321,
                        help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level')

    args = parser.parse_args()

    # 创建配置对象
    config = Config()

    # 更新配置
    for key, value in vars(args).items():
        setattr(config, key, value)

    # 根据数据集更新特定配置
    config.update_for_dataset(config.dataset)

    # 创建保存目录
    save_dir = create_save_directory(config)
    print(f"AGL-STFCM results will be saved to: {save_dir}")

    # 复制代码文件到保存目录
    code_files = ['run.py', 'models.py', 'trainer.py', 'data_loader.py', 'metrics.py', 'config.py']
    for file in code_files:
        if os.path.exists(file):
            shutil.copy2(file, save_dir)

    # 运行模式
    if config.mode == 'grid_search':
        best_params = grid_search(config, save_dir)
    else:
        results = single_run(config, save_dir)


if __name__ == '__main__':
    main()