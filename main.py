#importing libraries
import glob
import numpy as np
from data_preprocessing import read_data
from model import build_model
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from keras.optimizers import Adam

from keras import backend as K
K.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



#path of dataset
train_path = 'resized_preprocessed_dataset/training_data/'
test_path  = 'resized_preprocessed_dataset/test_data/'
scalar = MinMaxScaler()

#reading image volume
def read_img_volume(list_of_image_path, label = False):
    list_of_data = []
    for i in range(len(list_of_image_path)):
        img = sitk.ReadImage(list_of_image_path[i])
        img_array = sitk.GetArrayFromImage(img)
        if label == False:
            img_array = scalar.fit_transform(img_array.reshape(-1, img_array.shape[0])).reshape(img_array.shape)
            
        list_of_data.append(img_array)
    
    return list_of_data
    

#list of path of image and mask for training and testing image
path_of_train_mask, path_of_train_data = read_data(train_path)
path_of_test_mask, path_of_test_data = read_data(test_path)

#list of image and mask for training
train_data = read_img_volume(path_of_train_data)
segmentation_data = read_img_volume(path_of_train_mask, label= True)

#list of image and mask for testing
test_data = read_img_volume(path_of_test_data)
test_seg_data = read_img_volume(path_of_test_mask, label = True)


X = []
Y = []

for i in range(25):
  X.append(train_data[i])
  Y.append(segmentation_data[i])

for i in range(25):
  X.append(test_data[i])
  Y.append(test_seg_data[i])

#expanding dimensions of dataset 
for i in range(len(X)):
  X[i] = np.expand_dims(X[i],axis = -1)
  Y[i] = np.expand_dims(Y[i],axis = -1)

#stacking the dataset for batch processing
X = np.stack(X, axis=0)
Y = np.stack(Y, axis=0)


#splitting dataset into 80-20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#defining the shape for input tensor
IMG_DEPTH = 28
IMG_HEIGHT = 320
IMG_WIDTH = 320
input_shape = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1)


#building and compiling model
model = build_model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ["accuracy"] )
print(model.summary())


#model training
history = model.fit(X_train, Y_train, epochs=1000, verbose = 1, batch_size = 5)

#plotting training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#evaluating the model
model.evaluate(X_test, Y_test)

#model prediction
y_pred = model.predict(X_test)


#calculating the dice score for evaluation
dice_scores = []
for i in range(len(y_pred)):
  dice = 2 * (y_pred[i] * Y_test[i]).sum() / (y_pred[i].sum() + Y_test[i].sum())
  dice_scores.append(dice)

print(np.mean(dice_scores))


#to save model
# model.save('model.keras')