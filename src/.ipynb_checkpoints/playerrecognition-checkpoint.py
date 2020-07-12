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
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers
from tensorflow.keras import Model


from keras.preprocessing import image
##################################################################################################

model_path = "model/"

#model = tf.keras.models.load_model('model\\players_model\\')
model = tf.keras.models.load_model(model_path+'players_model')

# Check its architecture
model.summary()

#----------------------------------------- Testing the model -----------------------------------#

test_path = 'players/test/'
player_path = 'Cristiano Ronaldo/CR7_prueba4.png' 

img=image.load_img(test_path+player_path, target_size=(150, 150))
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
  
classes = model.predict(images, batch_size=10)

  
print(classes[0])

if classes[0][0]==1:
    print("Cristiano Ronaldo")
    
elif classes[0][1]==1:
    print( "Lionel Messi")
elif classes[0][2]==1:
    print("Paulo Dybala")
elif classes[0][3]==1:
    print("Sergio Aguero")
elif classes[0][4]==1:
    print("Sergio Romero")


