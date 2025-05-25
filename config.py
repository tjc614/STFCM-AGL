class Config:
    """配置类"""

    def __init__(self):
        # 数据集配置
        self.dataset = 'taiex'
        self.year = 2004  # 适用于taiex和sse
        self.eeg_subject = 1  # 适用于eeg数据集

        # 模型参数
        self.order = 4
        self.node_num = 4  # 根据数据集调整
        self.d_hidden = 4
        self.filter_nums = 10
        self.kernel_size = 3
        self.aggregator_type = 'adaptive_pool'
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.l1 = 1e-5
        self.l2 = 1e-5

        # 训练参数
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        self.validation_split = 0.0
        self.verbose = 1

        # 输出节点
        self.output_nodes = [0]  # 根据数据集调整

        # 其他参数
        self.seed = 321
        self.save_dir = './save'

    def update_for_dataset(self, dataset_name):
        """根据数据集更新配置"""
        dataset_configs = {
            'taiex': {'node_num': 4, 'output_nodes': [0]},
            'sse': {'node_num': 4, 'output_nodes': [0]},
            'traffic': {'node_num': 6, 'output_nodes': list(range(6))},
            'temp': {'node_num': 24, 'output_nodes': [2, 3]},
            'epc': {'node_num': 7, 'output_nodes': [4, 5, 6]},
            'eeg': {'node_num': 32, 'output_nodes': list(range(32))}
        }

        if dataset_name in dataset_configs:
            for key, value in dataset_configs[dataset_name].items():
                setattr(self, key, value)