from keras.models import load_model
import numpy as np
import cv2

model = load_model("model.h5")
data_type = "Testing"
emotion_class = ["Happy", "Neutral"]  # 兩類情緒(Happy and Neutral)
img_num = 10  # 測試資料有10張
img_size = 160  # 影像大小160x160

# np.zeros 初始一個全為0的矩陣, 大小為兩類情緒, 影像大小160x160, RGB 3個色彩通道
x_test = np.zeros((1, img_size, img_size, 3))
correct = 0

for j in range(10):  # 測試資料有10張, 迴圈表示變數j從0到9

    imagePath = (data_type + "/img (" + str(j + 1) + ").jpg")  # 影像路徑
    print(imagePath)

    image = cv2.imread(imagePath)  # 用cv2套件來讀取影像
    resized_image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    x_test[0][:][:][:] = resized_image  # 把縮小後的影像存到變數x_test裡面
    x_test /= 255  # 把灰階值0~255正規化0~1之間

    predictions = model.predict(x_test)  # 對測試影像進行模型預測
    print(predictions)  # 印出預測的機率結果
    print(str(emotion_class[np.argmax(predictions)]))  # 印出文字結果

    if np.argmax(predictions) == 0:
        # cv2.putText 在圖片上用寫字的方式show出結果, 裡面參數依次是：原始影像,添加的文字,左上角坐標,字體,字體大小,顏色,字體粗細
        cv2.putText(image, emotion_class[np.argmax(predictions)], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    else:
        cv2.putText(image, emotion_class[np.argmax(predictions)], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("result", image)  # 把圖和結果的字一起顯示在視窗中
    cv2.waitKey(2000)  # 等待2 秒

    if emotion_class[np.argmax(predictions)] == "Happy":  # 如果是開心就加一
        correct += 1

accuracy = correct / img_num * 100  # 辨識率
print('辨識率: ', accuracy, '%')
