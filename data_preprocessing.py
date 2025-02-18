import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import glob

#input path
path_of_data = 'preprocessed_dataset/test_data/'

#output path
path_to_save_data = 'resized_preprocessed_dataset/test_data/'

#reading the inputs and returning training data as well as ground truth
def read_data(path_of_data):
    
    #dividing training data and ground truth
    list_of_segmentation_data = glob.glob(path_of_data+'*_segmentation.mhd', recursive=True)
    list_data = glob.glob(path_of_data+'*.mhd', recursive=True)
    list_of_train_data = []

    for i in list_data:
        if(i not in list_of_segmentation_data):
            list_of_train_data.append(i)
    
    return list_of_segmentation_data, list_of_train_data

#making consistent input
def consistent_slices(list_of_data):
    
    #number of slices needed for each image
    no_of_target_slice = 28

    for i in range(30):
        #name of the input file
        filename = list_of_data[i][18:]
        
        #reading an image
        image = sitk.ReadImage(list_of_data[i])
        image_array = sitk.GetArrayViewFromImage(image)
        no_of_slice_in_image = image_array.shape[0]
        
        #if number of slice is > 28 then crop image   
        if(no_of_slice_in_image > no_of_target_slice):
            start = (no_of_slice_in_image - no_of_target_slice) // 2
            end = start + no_of_target_slice
            preprossed_array = image_array[start:end, :, :]
        
        #if number of slice is < 28 then add new slices
        else:
            pad = no_of_target_slice - no_of_slice_in_image
            
            if(pad%2 != 0):
                preprossed_array = np.pad(image_array, ((pad//2, (pad//2 + 1)), (0,0), (0,0)))
            else:
                preprossed_array = np.pad(image_array, ((pad//2, pad//2), (0,0), (0,0)))
        
        #saving a preprocessed image with all the meta data
        preprossed_array = sitk.GetImageFromArray(preprossed_array)
        preprossed_array.SetSpacing(image.GetSpacing())
        preprossed_array.SetOrigin(image.GetOrigin())
        preprossed_array.SetDirection(image.GetDirection())
        sitk.WriteImage(preprossed_array, path_to_save_data+filename) 
    
def resize_img(list_of_data, label = False):
    
    
    for i in range(30):
        
        filename = list_of_data[i][31:]
        image = sitk.ReadImage(list_of_data[i])
        new_size = [320, 320, image.GetSize()[2]]
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_spacing = [original_spacing[0] * (original_size[0] / new_size[0]),
                    original_spacing[1] * (original_size[1] / new_size[1]),
                    original_spacing[2]]
        
        if(label == False):
                
            resampled_img = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear, 
                                        image.GetOrigin(), new_spacing, image.GetDirection(),
                                        0.0, image.GetPixelID())
        else:
            resampled_img = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                          image.GetOrigin(), new_spacing, image.GetDirection(),
                                          0.0, image.GetPixelID())


        sitk.WriteImage(resampled_img, path_to_save_data+filename)       




list_of_segmentation_data, list_of_test_data = read_data(path_of_data) 
# resize_img(list_of_test_data)    

# consistent_slices(list_of_segmentation_data)