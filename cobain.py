import cv2 
import os
import time
from PIL import Image
from keras.models import load_model
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

path = 'cascade.xml'
camera = 1
objectName = 'TOMAT'
frameWidth = 640
frameHeight = 480
colors = (255, 0, 255)

# cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(camera)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass


cv2.namedWindow('Hasil')
cv2.createTrackbar('Scale', 'Hasil',400,100,empty)
cv2.createTrackbar('Neig', 'Hasil',8,20,empty)

cascade = cv2.CascadeClassifier(path)

while True:
    success, img = cap.read()
    # Resize the raw image into (224-height,224-width) pixels
    image_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_array = np.asarray(image_resize, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_normal = (image_array / 127.5) - 1
    
    data[0] = image_normal

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    indeks = int(index)
    nama = str(class_name[2:])
    # konfidens = str(np.round(confidence_score * 100))[:-2] + '%'
    konfidens = int(confidence_score * 100)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleVal = 1 + (cv2.getTrackbarPos('Scale', 'Hasil')/1000)
    neig = cv2.getTrackbarPos('Neig', 'Hasil')
    # objects = cascade.detectMultiScale(gray, scaleVal, neig)
    objects = cascade.detectMultiScale(gray, 1.2, 80)
    for (x, y, w, h) in objects :
        # cv2.rectangle(img, (x, y),(x+w, y+h),colors,3)
        # cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,colors,2)
        # roi_color = img[y:y+h, x:x+w]
        if indeks == 0:
            # cv2.putText(image, "Tomat Sudah Matang", (10, 50), font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
            # cv2.rectangle(img, (120, 360), (480,120), (0,255,0),2)
            # cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0),3)
            cv2.putText(img,"No Data",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            roi_color = img[y:y+h, x:x+w]
        elif indeks == 1:
            # cv2.putText(image, "Tomat Setengah Matang", (10, 50), font, fontScale, (0,255,255), thickness, cv2.LINE_AA)
            cv2.rectangle(img, (120, 360), (480,120), (0,255,255),2)
            cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,255),3)
            cv2.putText(img,"Matang",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            roi_color = img[y:y+h, x:x+w]
        elif indeks == 2:
            # cv2.putText(image, "Tomat Setengah Matang", (10, 50), font, fontScale, (0,255,255), thickness, cv2.LINE_AA)
            cv2.rectangle(img, (120, 360), (480,120), (0,255,255),2)
            cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,255),3)
            cv2.putText(img,"Setengah Matang",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            roi_color = img[y:y+h, x:x+w]
        elif indeks == 3:
            # cv2.putText(image, "Tomat Masih Mentah", (10, 50), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
            cv2.rectangle(img, (120, 360), (480,120), (0,0,255),2)
            cv2.rectangle(img, (x, y),(x+w, y+h),(0,0,255),3)
            cv2.putText(img,"Masih Mentah",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            roi_color = img[y:y+h, x:x+w]
    
    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    print("Confidence Score:", konfidens)

    cv2.imshow("Hasil", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("EXIT")
cap.release()
cv2.destroyAllWindows()
