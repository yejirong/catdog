# 模型保存与加载——张量方式：保存网络张量参数到文件系统上是最轻量级的一种方式
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 数据据预处理
# 对每个样本的数据进行处理，x代表输入图片，y代表图片对应标签
def preprocess(x, y):
    # 将x∈（0，255）的灰度图片，转化成0-1的灰度范围
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 将图片转化成28*28的宽和高
    x = tf.reshape(x, [28 * 28])
    # 转化y的数据类型
    y = tf.cast(y, dtype=tf.int32)
    # 将y值进行one_hot转化，如1->[0,1,0,0,0,0,0,0,0,0],9->[0,0,0,0,0,0,0,0,1,0]
    y = tf.one_hot(y, depth=10)
    return x, y


# 设置batch大小
batchsz = 128
# 加载数据集
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# 打印数据集形状
print('datasets:', x.shape, y.shape, x.min(), x.max())

# 转换成 Dataset 对象，才能利用 TensorFlow 提供的各种便捷功能
# 通过 Dataset.from_tensor_slices 可以将训练部分的数据图片 x 和标签 y 都转换成Dataset 对象
db = tf.data.Dataset.from_tensor_slices((x, y))
# 将数据转换成 Dataset 对象后，可以添加一系列的数据集标准处理步骤，如随机打散shuffle、 预处理preprocess、 按批装载batch
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

# 构建网络结构，通过 Sequential 容器封装成一个网络大类对象
network = Sequential([layers.Dense(256, activation='relu'),  # 全连接层，第一层节点为256
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])  # 输出层，10分类
# 调用 build 函数或者直接进行一次前向计算，才能完成网络参数的创建
network.build(input_shape=(None, 28 * 28))
# 打印网络结构
network.summary()

# 模型装配
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

# 模型训练
network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)
# 模型测试
network.evaluate(ds_val)
# 保存模型权重，将 network 模型保存到 weights.ckpt 文件上
network.save_weights('weights.ckpt')
print('saved weights.')
# 删除网络对象
del network
##############################################################################
# 先创建好网络对象
# 然后调用网络对象的 load_weights(path)方法
# 即可将指定的模型文件中保存的张量数值写入到当前网络参数中去
############################################################################

# 需要先建立模型，且与原模型必须完全一致
network_1 = Sequential([layers.Dense(256, activation='relu'),
                        layers.Dense(128, activation='relu'),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(10)])
# 装配新的模型
network_1.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
# 加载已保存的网络参数
network_1.load_weights('weights.ckpt')
print('loaded weights!')
# 在已有参数情况进行模型测试
network_1.evaluate(ds_val)
