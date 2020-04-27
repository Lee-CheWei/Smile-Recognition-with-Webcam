# 載入需要的套件
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np
import os
import cv2

# 變數宣告
data_type = "Training"
emotion_class = ["Happy", "Neutral"]  # 兩種情緒類別
img_num = 100  # 一種情緒有 100 張training data
img_size = 160  # training data 統一resize到160x160的影像大小

# 讀圖和存值到變數裡面
x_train = np.zeros((len(emotion_class) * img_num, img_size, img_size, 3))  # 建立大小為(200, 160, 160, 3)的陣列, 用來存放影像資料
y_train = np.zeros((len(emotion_class) * img_num, 2))  # 建立大小為(200, 2)的陣列, 用來存放標記資料

s = 0
for i in range(len(emotion_class)):  # len(emotion_class) 兩類情緒
    for j in range(img_num):  # 一種情緒有 100 張training data
        imagePath = (os.getcwd() + "\\" + emotion_class[i] + "\\" + data_type + "\\img (" + str(
            j + 1) + ").jpg")  # 影像載入路徑: os.getcwd()存取目前這支程式的資料夾路徑, 接著根據資料夾的放置結構設置欲存取影像之路徑
        image = cv2.imread(imagePath)  # 用cv2.imread來讀取圖片
        resized_image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # resize影像大小
        x_train[s][:][:][:] = resized_image  # 將resize後的image依序存入(200, 160, 160, 3)陣列中
        y_train[s][i] = 1  # s=0~99 (前100 張)將陣列索引為 i=0 (Happy)的值設為1, 也就是Happy標記為1 0, Neutral標記為0 1
        s += 1  # 將Happy和Neutral一起擺放之陣列索引
x_train /= 255  # 像素值0~255正規化至0~1

# CNN
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
                 input_shape=(img_size, img_size, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())

# 進行神經網路模型訓練
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=20, batch_size=125, verbose=2)

# 模型儲存
model.save('model.h5')
