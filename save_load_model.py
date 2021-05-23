# 模型保存与加载——网络方式：可以将模型的结构以及模型的参数保存.h到文件上
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

# 保存
network.save('model.h5')
print('saved total model.')
# 删除网络对象
del network

print('loaded model from file.')
# 恢复网络结构和网络参数
network_1 = tf.keras.models.load_model('model.h5', compile=False)
# 装配新的模型
network_1.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )
# 在已有参数情况进行模型测试
network_1.evaluate(ds_val)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',    #配置模型
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

rain_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)  #测试集不能增强！

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_2.h5') #保存第二个模型