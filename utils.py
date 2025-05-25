import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime


def plot_predictions(true_values, predictions, save_path=None, title="Predictions vs True Values"):
    """绘制预测结果对比图"""
    plt.figure(figsize=(12, 8))

    if len(true_values.shape) > 1 and true_values.shape[1] > 1:
        # 多变量情况，绘制前4个变量
        n_vars = min(4, true_values.shape[1])
        for i in range(n_vars):
            plt.subplot(2, 2, i + 1)
            plt.plot(true_values[:, i], label='True', alpha=0.7)
            plt.plot(predictions[:, i], label='Predicted', alpha=0.7)
            plt.title(f'Variable {i + 1}')
            plt.legend()
            plt.grid(True, alpha=0.3)
    else:
        # 单变量情况
        if len(true_values.shape) > 1:
            true_values = true_values[:, 0]
            predictions = predictions[:, 0]

        plt.plot(true_values, label='True Values', alpha=0.7)
        plt.plot(predictions, label='Predictions', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制MSE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    if 'val_mse' in history.history:
        plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def save_config(config, save_path):
    """保存配置到文件"""
    with open(save_path, 'w') as f:
        f.write("Configuration\n")
        f.write("=============\n\n")
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")


def load_config_from_file(config_path):
    """从文件加载配置"""
    config_dict = {}
    with open(config_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line and not line.startswith('=') and not line.startswith('Configuration'):
                key, value = line.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                # 尝试转换数据类型
                try:
                    if value.isdigit():
                        config_dict[key] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        config_dict[key] = float(value)
                    elif value.lower() in ['true', 'false']:
                        config_dict[key] = value.lower() == 'true'
                    else:
                        config_dict[key] = value
                except:
                    config_dict[key] = value
    return config_dict


def create_experiment_name(config):
    """创建实验名称"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.dataset}_{config.aggregator_type}_o{config.order}_h{config.d_hidden}_{timestamp}"
    return exp_name


def print_model_summary(model, input_shape):
    """打印模型摘要"""
    print("\nModel Summary:")
    print("==============")
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(var) for var in model.trainable_variables]):,}")


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory