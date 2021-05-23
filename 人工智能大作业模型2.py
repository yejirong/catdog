from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# 引用vgg16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_dir = 'F:/大三上/大作业/人工智能/cat_dog'  # 数据路径
# 训练，验证和测试的目录
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 构建keras中的Sequential模型
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())  # 采用Flatten层，用来将输入“压平”
model.add(layers.Dropout(0.5))  # 防止过拟合采用Dropout
model.add(layers.Dense(256, activation='relu'))  # 使用relu函数
model.add(layers.Dense(1, activation='sigmoid'))  # 使用sigmoid函数

# 图片归一化，数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)  # 图片归一化
# 使用keras数据生成器，图片大小，batch大小，"binary"返回1D的二值标签
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
# vgg16判断，加快训练速度
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 使用函数binary_crossentropy，使用RMSprop优化器
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
# 训练模型，生成器返回次数：steps_per_epoch，数据迭代的轮数：epochs
# 生成验证集的生成器：validation_data，指定验证集的生成器返回次数： validation_steps
history = model.fit_generator(train_generator,
                              steps_per_epoch=30,
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=20)
model.save('model2.h5')

# 画出相关的loss等曲线
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='训练acc')
plt.plot(epochs, val_acc, 'b', label='验证acc')
plt.title('验证准确性acc')
plt.legend()
plt.figure()
plt.savefig('./acc.png')
# plt.show()

plt.plot(epochs, loss, 'bo', label='训练loss')
plt.plot(epochs, val_loss, 'b', label='验证loss')
plt.title('验证准确性loss')
plt.legend()
plt.savefig('./loss.png')
plt.show()
