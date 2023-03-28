# import tensorflow.keras
from keras.models import load_model
import numpy as np
import cv2
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# model = tensorflow.keras.models.load_model('4classes.h5')
model = load_model("keras_Model.h5", compile=False)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
cam = cv2.VideoCapture(0)

text = ""

while True:
    # _,img = cv2.imread('right.jpg')
    _,img = cam.read()
    img = cv2.resize(img,(224, 224))

    #turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)
    for i in prediction:
        if i[0] > 0.5:
            text ="No data"
        if i[1] > 0.7:
            text ="matang"
        if i[2] > 0.7:
            text ="setengah matang"
        if i[3] > 0.7:
            text ="mentah"
        # print(text)
        img = cv2.resize(img,(500, 500))
        cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("EXIT")
cam.release()
cv2.destroyAllWindows()