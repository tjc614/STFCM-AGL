import tensorflow as tf
from tensorflow import keras
from tcn import TCN


def allocate_granularity(data, lambda1, lambda2):
    """粒度分配策略函数"""
    lower_bounds = tf.sigmoid(lambda1 * (data - 0.05 * tf.abs(data)))
    upper_bounds = tf.sigmoid(lambda2 * (data + 0.05 * tf.abs(data)))
    return lower_bounds, upper_bounds


class GraphSAGELayer(keras.layers.Layer):
    """GraphSAGE层实现"""

    def __init__(self, units, activation=None, aggregator_type='', lambda1=1.0, lambda2=1.0):
        super(GraphSAGELayer, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.aggregator_type = aggregator_type
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dense = keras.layers.Dense(units)

        if aggregator_type == 'lstm':
            self.lstm_layer = keras.layers.LSTM(units)
        if aggregator_type == 'adaptive_pool':
            self.attn_dense = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        lower_bounds, upper_bounds = allocate_granularity(node_features, self.lambda1, self.lambda2)

        if self.aggregator_type == 'adaptive_pool':
            max_pool = tf.reduce_max(tf.matmul(adjacency_matrix, node_features), axis=1, keepdims=True)
            avg_pool = tf.reduce_mean(tf.matmul(adjacency_matrix, node_features), axis=1, keepdims=True)
            pooled_features = tf.concat([max_pool, avg_pool], axis=-1)
            attn_scores = self.attn_dense(pooled_features)
            attn_weights = tf.nn.softmax(attn_scores, axis=1)
            neighbor_features = tf.reduce_sum(attn_weights * pooled_features, axis=1, keepdims=True)
            neighbor_features = tf.tile(neighbor_features, [1, adjacency_matrix.shape[1], 1])

        combined_features = tf.concat([node_features, neighbor_features], axis=-1)
        combined_features = tf.concat([lower_bounds, combined_features, upper_bounds], axis=-1)
        output = self.dense(combined_features)

        if self.activation is not None:
            output = self.activation(output)
        return output


class GraphSAGEModel(keras.Model):
    """GraphSAGE模型实现"""

    def __init__(self, hidden_units, output_units, aggregator_type='mean', lambda1=1.0, lambda2=1.0):
        super(GraphSAGEModel, self).__init__()
        self.sage1 = GraphSAGELayer(hidden_units, activation='relu',
                                    aggregator_type=aggregator_type, lambda1=lambda1, lambda2=lambda2)
        self.sage2 = GraphSAGELayer(output_units, aggregator_type=aggregator_type,
                                    lambda1=lambda1, lambda2=lambda2)

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        x = self.sage1([adjacency_matrix, node_features])
        x = self.sage2([adjacency_matrix, x])
        return x


class AGLSTFCM(keras.Model):
    """AGL-STFCM主模型 (Adaptive Granularity Learning for Spatio-Temporal Fuzzy Cognitive Maps)"""

    def __init__(self, order, node_num, d_hidden, filter_nums, kernel_size,
                 l1=1e-5, l2=1e-5, aggregator_type='adaptive_pool', lambda1=1.0, lambda2=1.0):
        super(AGLSTFCM, self).__init__()
        self.order = order
        self.node_num = node_num
        self.d_hidden = d_hidden
        self.filter_nums = filter_nums
        self.kernel_size = kernel_size
        self.l1 = l1
        self.l2 = l2

        # 初始化组件
        self.node_embedding_list = []
        self.tcn_list = []
        self.dense_list = []

        # 创建node embedding模块
        for i in range(self.order):
            self.node_embedding_list.append(
                GraphSAGEModel(self.d_hidden, self.node_num,
                               aggregator_type=aggregator_type, lambda1=lambda1, lambda2=lambda2))

        # 创建TCN和Dense模块
        for i in range(self.node_num):
            self.tcn_list.append(TCN(nb_filters=filter_nums, kernel_size=kernel_size, dilations=[1, 2, 4]))
            self.dense_list.append(keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        node_embed_order = []

        # 对每个时间步进行node embedding
        for i in range(self.order):
            adjacency_matrix = tf.eye(self.node_num, batch_shape=[batch_size])
            node_features = tf.expand_dims(inputs[:, :, i], axis=-1)
            node_embedding = self.node_embedding_list[i]([adjacency_matrix, node_features])

            expected_shape = [batch_size, self.node_num, self.d_hidden]
            actual_shape = tf.shape(node_embedding)
            if tf.reduce_prod(actual_shape) == tf.reduce_prod(expected_shape):
                node_embedding = tf.reshape(node_embedding, expected_shape)
            node_embed_order.append(node_embedding)

        node_embed_concat = tf.stack(node_embed_order, axis=2)

        # 对每个节点进行TCN和Dense处理
        outputs_list = []
        for i in range(self.node_num):
            node_embed = tf.transpose(node_embed_concat[:, i, :, :], perm=[0, 2, 1])
            tcn_outputs = self.tcn_list[i](node_embed)
            outputs = self.dense_list[i](tcn_outputs)
            outputs_list.append(outputs)

        fcm_outputs = tf.concat(outputs_list, axis=-1)
        return fcm_outputs