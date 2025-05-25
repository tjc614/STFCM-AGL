# metrics.py - 评价指标文件

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(true, prediction):
    """计算RMSE"""
    return np.sqrt(mean_squared_error(true, prediction))


def mae(true, prediction):
    """计算MAE"""
    return mean_absolute_error(true, prediction)


def mape(true, prediction):
    """计算MAPE"""
    return np.mean(np.abs((true - prediction) / true)) * 100


def calculate_metrics(true, prediction, verbose=True):
    """计算所有评价指标"""
    rmse_val = rmse(true, prediction)
    mae_val = mae(true, prediction)
    mape_val = mape(true, prediction)

    if verbose:
        print(f'RMSE: {rmse_val:.4f}')
        print(f'MAE: {mae_val:.4f}')
        print(f'MAPE: {mape_val:.4f}%')

    return {
        'rmse': rmse_val,
        'mae': mae_val,
        'mape': mape_val
    }


def evaluate_model(true_values, predictions, output_nodes, scaler=None, dataset_name=None):
    """评估模型在多个输出节点上的性能"""
    if scaler is not None:
        true_values = scaler.inverse_transform(true_values)
        predictions = scaler.inverse_transform(predictions)

    results = {}
    rmse_list = []
    mae_list = []

    for i, node in enumerate(output_nodes):
        node_metrics = calculate_metrics(true_values[:, node], predictions[:, node], verbose=False)
        results[f'node_{node}'] = node_metrics
        rmse_list.append(node_metrics['rmse'])
        mae_list.append(node_metrics['mae'])

    # 计算平均指标
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)

    results['average'] = {
        'rmse': avg_rmse,
        'mae': avg_mae
    }

    # 所有数据集都显示为test error
    print(f'Average RMSE: {avg_rmse:.4f}')
    print(f'Average MAE: {avg_mae:.4f}')

    return results


def evaluate_training_error(model, train_X, train_label, output_nodes, scaler=None):
    """专门用于计算训练误差的函数（保留但不使用）"""
    train_predictions = model.predict(train_X)

    if scaler is not None:
        train_predictions = scaler.inverse_transform(train_predictions)
        train_true = scaler.inverse_transform(train_label)
    else:
        train_true = train_label

    results = {}
    rmse_list = []
    mae_list = []

    for i, node in enumerate(output_nodes):
        node_metrics = calculate_metrics(train_true[:, node], train_predictions[:, node], verbose=False)
        results[f'node_{node}'] = node_metrics
        rmse_list.append(node_metrics['rmse'])
        mae_list.append(node_metrics['mae'])

    # 计算平均指标
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)

    results['average'] = {
        'rmse': avg_rmse,
        'mae': avg_mae
    }

    print(f'Average Training RMSE: {avg_rmse:.4f}')
    print(f'Average Training MAE: {avg_mae:.4f}')

    return results