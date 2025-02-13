import glob
import numpy as np
from data_preprocessing import read_data
from model import build_model, conv_block, encoder_block, decoder_block
import SimpleITK as sitk
from keras.optimizers import Adam

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
t_data = train_data[0]

BATCH = 64
IMG_DEPTH = t_data.shape[0]
IMG_HEIGHT = t_data.shape[1]
IMG_WIDTH = t_data.shape[2]

input_shape = (BATCH, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, )
model = build_model(input_shape)
model.compile(optimizer=Adam(lr = 1e-3), loss = "binary_crossentropy", metrics = ["accuracy"] )
model.summary()