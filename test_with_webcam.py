import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 在準備擷取攝影機的影像之前, 要先建立一個VideoCapture物件
# VideoCapture(0)代表使用第一隻攝影機
video_capture = cv2.VideoCapture(0)

# 調用一個很基本的opencv的臉部偵測工具
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

anterior = 0
x_test = np.zeros((1, 160, 160, 3))  # 宣告一個矩陣大小 96x96x3(3因為是rgb) np.zeros這個矩陣初始值=0
font = cv2.FONT_HERSHEY_SIMPLEX  # 為了將人臉辨識結果寫在畫面中的宣告

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(96, 96, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # 攤平接DNN
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.load_weights("accuracy80_weights.h5")  # load先前訓練好的CNN model的weights

while True:
    # 如果攝影機沒有啟動, 印出'Unable to load camera.'
    if not video_capture.isOpened():
        print('Unable to load camera.')

    # 從攝影機擷取一張影像, 第一個回傳值ret代表成功與否
    # 第二個回傳值frame就是攝影機的單張畫面
    ret, frame = video_capture.read()

    # 因為需要進行多次的縮放, scaleFactor用來控制每次圖片縮小的比例
    # minNeighbors表示每次偵測時, 同時偵測周圍多少點
    # minSize表示偵測點的最小值
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 把偵測到的人臉區域框出來
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        im = frame[y:y + h, x:x + w]  # 存偵測到的每個人臉的影像
        resize_image = cv2.resize(im, (160, 160), interpolation=cv2.INTER_CUBIC)  # 將抓到的人臉resize到一樣大小
        x_test[0][:][:][:] = resize_image
        x_test = x_test / 255  # 下面同#位置的幾行縮排到這可以執行多人情緒辨識
        predictions = model.predict(x_test)
        print(predictions)

        # 判斷辨識結果並加文字到影像裡面
        if np.argmax(predictions[0]) == 0:
            cv2.putText(frame, 'Neutral', (x, y), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'happy', (x, y), font, 1, (0, 0, 255), 2)

        if len(faces) == 1:
            cv2.putText(frame, str(len(faces)) + "face detected", (10, 40), font, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, str(len(faces)) + "faces detected", (10, 40), font, 1, (255, 0, 0), 2)
        print("偵測到 ", len(faces), "個人臉")

    # 在螢幕上show出攝影機拍到的畫面
    cv2.imshow('Video', frame)

    # 若按下q鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
video_capture.release()
# 關閉所有opencv視窗
cv2.destroyAllWindows()
