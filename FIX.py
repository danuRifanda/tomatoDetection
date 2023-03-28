from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model1.h5", compile=False)

# Load the labels
class_names = open("labels1.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	
# CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(1)
webcam = 0
camera = cv2.VideoCapture(webcam, cv2.CAP_DSHOW)


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
  
# Line thickness of 2 px
thickness = 2

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image_resize = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

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

    if indeks == 0:
        cv2.putText(image, "No Data", (10, 50), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        cv2.rectangle(image, (120, 360), (480,120), (0,0,255),2)
    elif indeks == 1:
        cv2.putText(image, "Tomat Sudah Matang", (10, 50), font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
        cv2.rectangle(image, (120, 360), (480,120), (0,255,0),2)
    elif indeks == 2:
        cv2.putText(image, "Tomat Setengah Matang", (10, 50), font, fontScale, (0,255,255), thickness, cv2.LINE_AA)
        cv2.rectangle(image, (120, 360), (480,120), (0,255,255),2)
    elif indeks == 3:
        cv2.putText(image, "Tomat Masih Mentah", (10, 50), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        cv2.rectangle(image, (120, 360), (480,120), (0,0,255),2)
    
    # cv2.putText(image, konfidens, (100, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    
    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    print("Confidence Score:", konfidens)
    
    cv2.imshow("Pendeteksi Kematangan Buah", image)
    
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
