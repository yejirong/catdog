from keras.models import load_model
import cv2

# model = load_model('F:/大三上/大作业/人工智能/代码/model.h5')  # 引入模型
model = load_model('F:/大三上/大作业/人工智能/代码/model2.h5')  # 引入模型
model.summary()  # 显示一下模型构成

img = cv2.imread('35.jpg')  # 图片名
image = cv2.resize(img, (150, 150))  # 设置图片大小
image = image.reshape(1, 150, 150, 3)
print('识别为:')
predict = model.predict_classes(image)
if (predict[0] == 0):
    print('A cat')
else:
    print('A dog')

# 显示图片
cv2.imshow('img', img)
cv2.waitKey(0)
