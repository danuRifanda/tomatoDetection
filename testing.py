# from keras.models import load_model  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the model
# model = load_model("keras_Model.h5", compile=False)

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(1)

# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()

#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)

#     # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#     # Normalize the image array
#     image = (image / 127.5) - 1

#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)

#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt

def size(img,ratio):
    img = cv2.resize(img,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    return img

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray,3)
    cv2.imshow('blur',blur)
    r_cscd = cv2.CascadeClassifier('cascade.xml')
    human = r_cscd.detectMultiScale(blur,1.8,20)
    try:
        for x,y,w,h in human:
            cv2.rectangle(img,(x,y),(x+w,y+h),(20,20,250),2)
    except:
        pass
    return img

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while (cap.isOpened()):
    _, frame = cap.read()
    frame = detect(frame)
    cv2.imshow('lal',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()