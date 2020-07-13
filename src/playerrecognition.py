import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

'''from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K '''

import cv2
import os
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers
from tensorflow.keras import Model


from keras.preprocessing import image
##################################################################################################

def getClassLabel(classes):
    
    label = "unknown"

    if classes[0][0]>0.8:
        label = "Cristiano Ronaldo"
    elif classes[0][1]>0.8:
        label= "Lionel Messi"
    elif classes[0][2]>0.8:
        label = "Paulo Dybala"
    elif classes[0][3]>0.8:
        label = "Sergio Romero"
    else:
        label = "Unknow"
        
    return label


##################################################################################################

model_path = "model/"

#model = tf.keras.models.load_model('model\\players_model\\')
model = tf.keras.models.load_model(model_path+'players_model')

# Check its architecture
#model.summary()

#----------------------------------------- Testing the model -----------------------------------#


# Load the cascade
face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
# Load the cascade
profileface_cascade = cv2.CascadeClassifier('src/haarcascade_profileface.xml')


# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input 
cap = cv2.VideoCapture('samples/ronaldovsmessi.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 
                                          scaleFactor=1.2, 
                                          minNeighbors=7,
                                          minSize=(50,50),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    
    profilefaces = profileface_cascade.detectMultiScale(gray, 
                                          scaleFactor=1.2, 
                                          minNeighbors=7,
                                          minSize=(50,50),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-50, y-10), (x+w+50, y+h+50), (255, 0, 0), 2)
        
        try:
            img_ = cv2.resize(img[y-10:y+h+50, x-50:x+w+50], (150,150))
            img_ = np.expand_dims(img_,axis=0)
            images = np.vstack([img_])
            
            classes = model.predict(images, batch_size=10)
            
            if np.amax(classes)>0.5:        
                cv2.putText(img,getClassLabel(classes),(x-50,y-15),cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,0,0), 2)
        except Exception as e:
            print(str(e))
        
        #faces_crop.append(img[y-10:y+h+50, x-50:x+w+50])
    for (x, y, w, h) in profilefaces:
        cv2.rectangle(img, (x-50, y-10), (x+w+50, y+h+50), (0, 255, 0), 2)
        
        try:
            img_ = cv2.resize(img[y-10:y+h+50, x-50:x+w+50], (150,150))
            img_ = np.expand_dims(img_,axis=0)
            images = np.vstack([img_])
            
            classes = model.predict(images, batch_size=10)
            
            if np.amax(classes)>0.5:    
                cv2.putText(img,getClassLabel(classes),(x-50,y-15),cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)
        except Exception as e:
            print(str(e))
        #faces_crop.append(img[y-10:y+h+50, x-50:x+w+50])
    
    
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()