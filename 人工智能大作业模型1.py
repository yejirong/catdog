'''
班级：大数据1801
姓名：叶际荣
学号: 201806140014
'''
import keras
from keras import layers
import numpy as np
import os
import shutil
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import models


def get_deal():
    # 数据集处理
    base_dir = 'F:/大三上/大作业/人工智能/cat_dog'  # 数据集保存的路径
    train_dir = os.path.join(base_dir, 'train')  # 训练集保存的文件夹
    train_dir_dog = os.path.join(train_dir, 'dog')  # 训练集保存的狗的图片
    train_dir_cat = os.path.join(train_dir, 'cat')  # 训练集保存的猫的图片

    test_dir = os.path.join(base_dir, 'test')  # 测试集保存的文件夹
    test_dir_dog = os.path.join(test_dir, 'dog')  # 狗
    test_dir_cat = os.path.join(test_dir, 'cat')  # 猫

    test_va = os.path.join(base_dir, 'test_va')  # 验证集集保存的文件夹
    test_va_dog = os.path.join(test_va, 'dog')
    test_va_cat = os.path.join(test_va, 'cat')
    # 创建相关文件夹
    os.mkdir(base_dir)  # 数据集文件夹
    os.mkdir(train_dir)  # 测试集文件夹
    os.mkdir(train_dir_dog)
    os.mkdir(train_dir_cat)
    os.mkdir(test_dir)  # 训练集文件夹
    os.mkdir(test_dir_dog)
    os.mkdir(test_dir_cat)
    os.mkdir(test_va)  # 验证集文件夹
    os.mkdir(test_va_dog)
    os.mkdir(test_va_cat)
    # 原数据集的路径
    dc_dir = 'C:/Users/叶际荣/Desktop/data/train'
    # 猫的1000张训练图片
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]  # 文件名称
    for fname in fnames:
        s = os.path.join(dc_dir, fname)  # 源地址
        d = os.path.join(train_dir_cat, fname)  # 保存在目标地址
        shutil.copyfile(s, d)

    # 猫的1000张测试图片
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 2000)]  # 文件名称
    for fname in fnames:
        s = os.path.join(dc_dir, fname)  # 源地址
        d = os.path.join(test_dir_cat, fname)  # 保存在目标地址
        shutil.copyfile(s, d)

    # 猫的1000张验证图片
    fnames = ['cat.{}.jpg'.format(i) for i in range(2000, 3000)]  # 文件名称
    for fname in fnames:
        src = os.path.join(dc_dir, fname)
        dst = os.path.join(test_va_cat, fname)  # 保存在目标地址
        shutil.copyfile(src, dst)

    # 狗的1000张训练图片
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]  # 文件名称
    for fname in fnames:
        s = os.path.join(dc_dir, fname)  # 源地址
        d = os.path.join(train_dir_dog, fname)  # 保存在目标地址
        shutil.copyfile(s, d)

    # 狗的1000张测试图片
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 2000)]
    for fname in fnames:
        s = os.path.join(dc_dir, fname)  # 源地址
        d = os.path.join(test_dir_dog, fname)  # 保存在目标地址
        shutil.copyfile(s, d)

    # 狗的1000张验证图片
    fnames = ['dog.{}.jpg'.format(i) for i in range(2000, 3000)]  # 文件名称
    for fname in fnames:
        src = os.path.join(dc_dir, fname)  # 源地址
        dst = os.path.join(test_va_dog, fname)  # 保存在目标地址
        shutil.copyfile(src, dst)


# 训练
def get_training():
    model = models.Sequential()  # 构建keras中的Sequential模型
    # 采用TensorFlow的卷积函数conv2d，卷积核数量，kernel_size参数，图片大小，使用relu函数
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    # 指定池窗口大小
    model.add(layers.MaxPool2D((2, 2)))
    # model.add(layers.Dropout(0.5))  # 防止过拟合采用Dropout
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    # model.add(layers.Dropout(0.5))  # 防止过拟合采用Dropout
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    # model.add(layers.Dropout(0.5))  # 防止过拟合采用Dropout
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    # model.add(layers.Dropout(0.5))  # 防止过拟合采用Dropout
    # 采用Flatten层，用来将输入“压平”
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid函数
    # 使用函数binary_crossentropy，使用RMSprop优化器
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    train_dir = 'F:/大三上/大作业/人工智能/cat_dog/train'  # 训练集路径
    test_dir = 'F:/大三上/大作业/人工智能/cat_dog/test'  # 测试集路径
    # 图片归一化
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # 使用keras数据生成器，图片大小，batch大小，"binary"返回1D的二值标签
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    # 训练模型，生成器返回次数：steps_per_epoch，数据迭代的轮数：epochs
    # 生成验证集的生成器：validation_data，指定验证集的生成器返回次数： validation_steps
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=test_generator,
        validation_steps=50)
    model.save('model.h5')  # 保存模型
    return history


if __name__ == '__main__':
    # get_deal()  # 数据处理
    get_training()  # 训练模型
