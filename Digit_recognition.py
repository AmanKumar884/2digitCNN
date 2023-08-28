import cv2
import numpy as np
from keras.models import load_model
# from tensorflow.keras.optimizers import Adam
import winsound
import csv
from datetime import datetime

data_collected = [['TIME' , 'SPEED']]
curr_sec = 100
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# model = load_model('model/digits.h5')
model = load_model('model/digits.h5', compile=True)

def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img)
    prob = np.amax(predict)
    class_index = model.predict(img)
    # class_index = (model.predict(img) > 0.5).astype("int32")
    result = class_index[0]
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


while True:
    _,frame = cap.read()
    #frame = cv2.rotate(frame,cv2.ROTATE_0)
    frame_copy = frame.copy()

    bbox_size = (60, 60)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("cropped", img_gray)


    result,probability = prediction(img_gray,model)
    res = np.argmax(result)
    cv2.putText(frame_copy, f"Prediction : {res}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
                cv2.LINE_AA)
    res = np.argmax(result)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")


    if curr_sec - now.second != 0:
        print("Current Time = ", current_time, end="\n")
        print("Speed = ",res,end='\n')

        new_data = [current_time, res]
        data_collected.append(new_data)

    curr_sec = now.second



    ## SOUND ALARM FOR HIGH SPEED
    if res>=80 and res<=90:
        frequency = 2000  # Set Frequency To 2000 Hertz
        duration = 500  # Set Duration To 500 ms == 1.5 second
        winsound.Beep(frequency, duration)
    if res>90:
        frequency = 2500
        duration = 1500
        winsound.Beep(frequency, duration)


    cv2.putText(frame_copy, "Probability : " + "{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame_copy,bbox[0],bbox[1],(0,255,0),3)
    cv2.imshow("input",frame_copy)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        csv_file = 'speed_data.csv'
        with open(csv_file, 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the data to the CSV file
            writer.writerows(data_collected)

        print(f"CSV file '{csv_file}' has been updated.")
        break

cv2.destroyAllWindows()