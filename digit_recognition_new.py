import numpy as np
import cv2
import pickle
from tensorflow import keras
from keras.models import load_model

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#### LOAD THE TRAINED MODEL
# with open('model/trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)
model = load_model('model_trained.h5', compile=True)

#### PREPROCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    #### PREDICT


      # Get the index of the highest probability
    classIndex = model.predict(img)
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)


    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   " + str(probVal),(50, 50), cv2.FONT_HERSHEY_COMPLEX,1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
