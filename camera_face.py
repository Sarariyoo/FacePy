import cv2
import glob
import time
import sys
import numpy as np
from datetime import datetime

#コンピュータに接続されているカメラが1台なら0
cap = cv2.VideoCapture(0)

#顔を判定するために使う
cascade_path = r'C:\Users\Ryoma\dev\FacePy\opencv-master\haarcascades\haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)

dir = r'C:\Users\Ryoma\dev\FacePy\img\\'

#写真を撮る枚数。多ければ多いほどいいが重くなる
num = 300

label = str(input("あなたを判別するための名前を半角英数字4文字で入力してください ex.slf:"))
file_number = len(glob.glob(r'C:\Users\Ryoma\dev\FacePy\img\*'))
count = 0

if not len(label) == 4:
    print("半角英数字4文字で入力してください")
    sys.exit()

while True:
    if count < num:
        time.sleep(0.01)
        print("あと{}枚です".format(num-count))
        datenow = datetime.now()
        r, img = cap.read()
        #imgUMat = np.float32(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 1, minSize = (100, 100))


        #顔を判定した場合
        if len(faces) > 0:
            for face in faces:
                #□の左上のx座標
                x = face[0]
                #同じくy座標
                y = face[1]
                #□の長さ
                width = face[2]
                #同じく高さ
                height = face[3]
                #リサイズ
                roi = cv2.resize(img[y:y + height, x:x + width], (50, 50), interpolation = cv2.INTER_LINEAR)
                cv2.imwrite(dir + label+ "_" + str(count) + '.jpg', roi)
        
        
        count = len(glob.glob(r'C:\Users\Ryoma\dev\FacePy\img\*')) - file_number
    
    else:
        break
    
cap.release()