import glob
import numpy as np
from data_preprocessing import read_data
from model import build_model
import SimpleITK as sitk

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

#reading image volume
def read_img_volume(list_of_image_path):
    list_of_data = []
    for i in range(len(list_of_image_path)):
        img = sitk.ReadImage(list_of_image_path[i])
        img_array = sitk.GetArrayFromImage(img)
        list_of_data.append(img_array)
    
    return list_of_data
    

path_of_train_mask, path_of_train_data = read_data(train_path)

train_data = read_img_volume(path_of_train_data)
segmentation_data = read_img_volume(path_of_train_mask)


IMG_DEPTH = 28
IMG_HEIGHT = 320
IMG_WIDTH = 320


input_shape = (IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, 1)
model = build_model(input_shape)




model.compile(optimizer='Adam', loss = "binary_crossentropy", metrics = ["accuracy"] )

print(model.summary())


history = model.fit(train_data, segmentation_data, epochs=10, verbose = 1, batch_size=2)