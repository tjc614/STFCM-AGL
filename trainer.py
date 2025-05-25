import tensorflow as tf
from models import AGLSTFCM
from data_loader import get_data_loader
from metrics import evaluate_model
import os
import shutil
from datetime import datetime


class AGLSTFCMTrainer:
    """AGL-STFCM模型训练器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None

        # 设置随机种子
        tf.random.set_seed(config.seed)

        # 设置日志级别
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def build_model(self):
        """构建模型"""
        self.model = AGLSTFCM(
            order=self.config.order,
            node_num=self.config.node_num,
            d_hidden=self.config.d_hidden,
            filter_nums=self.config.filter_nums,
            kernel_size=self.config.kernel_size,
            l1=self.config.l1,
            l2=self.config.l2,
            aggregator_type=self.config.aggregator_type,
            lambda1=self.config.lambda1,
            lambda2=self.config.lambda2
        )

        optimizer = tf.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

        return self.model

    def load_data(self):
        """加载数据"""
        data_loader = get_data_loader(self.config.dataset)

        if self.config.dataset in ['taiex', 'sse']:
            return data_loader(self.config.order, self.config.year)
        elif self.config.dataset == 'eeg':
            return data_loader(self.config.order, self.config.eeg_subject)
        else:
            return data_loader(self.config.order)

    def train(self):
        """训练模型"""
        # 加载数据
        train_X, train_label, test_X, test_label, scaler = self.load_data()

        # 构建模型
        if self.model is None:
            self.build_model()

        # 训练模型
        self.history = self.model.fit(
            x=train_X,
            y=train_label,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            validation_split=self.config.validation_split
        )

        # 所有数据集都计算测试误差
        print(f"\n=== Test Error Evaluation ({self.config.dataset.upper()} Dataset) ===")
        predictions = self.model.predict(test_X)
        results = evaluate_model(test_label, predictions, self.config.output_nodes,
                                 scaler, self.config.dataset)

        return results, self.history

    def save_model(self, save_path):
        """保存模型"""
        if self.model is not None:
            self.model.save_weights(save_path)
            print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """加载模型"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(load_path)
        print(f"Model loaded from {load_path}")