import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import math
import pickle


plt.figure(figsize=(3,2))
plt.rcParams.update({'font.size': 4})

path = 'myDataMain'

test_ratio = 0.2
val_ratio = 0.2
images = []
targetNo = []
imgDim = (32,32,3)


myList = os.listdir(path)
print('Total no of classes : ',len(myList))

noOfClasses = len(myList)
print('Importing classes ...')
for x in range(0,noOfClasses):
    myPicList = os.listdir(path + '/' + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + '/' + str(x) + '/' + y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        targetNo.append(x)
    print(x, end=' ')

print('\n')
# print(len(images))
# print(len(targetNo))

images = np.array(images)
targetNo = np.array(targetNo)

print(images.shape)
print(targetNo.shape)

# DATA SPLITTING
X_train, X_test, y_train, y_test = train_test_split(images, targetNo, test_size = test_ratio)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = val_ratio)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

noOfSamp = []
for x in range(0,noOfClasses):
    noOfSamp.append(len(np.where(y_train == x)[0]))

print(noOfSamp)

# Class ID vs no of Images
plt.bar(range(0,noOfClasses),noOfSamp)
plt.title("No of Images for each class")
plt.xlabel('Class ID')
plt.ylabel('No of Images')
plt.show()

def prePrecessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# img = prePrecessing(X_train[2])
# img = cv2.resize(img,(300,300))
# cv2.imshow('Preprocessed', img)
# cv2.waitKey(0)

X_train = np.array(list(map(prePrecessing, X_train)))
X_test = np.array(list(map(prePrecessing, X_test)))
X_val = np.array(list(map(prePrecessing, X_val)))
print(X_train.shape)

#### RESHAPE IMAGES
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
print(X_train.shape)

dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.2,
                             shear_range = 0.1,
                             rotation_range = 10)

dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)

def TestModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imgDim[0],
                                                               imgDim[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = TestModel()
print(model.summary())


### STARTING THE TRAINING PROCESS

# Calculate the number of steps per epoch based on the dataset size and batch size
steps_per_epoch = math.ceil(len(X_train) / 50)

# Fit the model with the updated steps_per_epoch value
history = model.fit(dataGen.flow(X_train, y_train, batch_size=50),
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    validation_data=(X_val, y_val),
                    shuffle=True)

#### PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

### SAVE THE TRAINED MODEL
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model,file)

# Save the model in .h5 format
model.save('model_trained.h5')
