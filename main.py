import glob
import numpy as np
from data_preprocessing import read_data
import SimpleITK as sitk
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

