import SimpleITK as sitk
from skimage import exposure
from data_preprocessing import read_data

path_of_train_data = 'resized_preprocessed_dataset/training_data/'
path_of_test_data =  'resized_preprocessed_dataset/test_data/'


list_of_train_seg, list_of_train = read_data(path_of_train_data)
list_of_test_seg, list_of_test = read_data(path_of_test_data)

#equalizing each test input using histogram
def test_hist(list_of_test_seg, list_of_test):

    for i in range(30):

        filename = list_of_test[i][39:]
        file_mask = list_of_test_seg[i][39:]
        
        image = sitk.ReadImage(list_of_test[i])
        img_array = sitk.GetArrayFromImage(image)
        flatten_img = img_array.flatten()
        eq_flat = exposure.equalize_hist(flatten_img)
        eq_vol = eq_flat.reshape(img_array.shape)
        
        global_eq_vol = sitk.GetImageFromArray(eq_vol)
        global_eq_vol.CopyInformation(image)
        
        
        #segmentation mask
        mask = sitk.ReadImage(list_of_test_seg[i])
        mask_array = sitk.GetArrayFromImage(mask)
        
        global_mask = sitk.GetImageFromArray(mask_array)
        global_mask.CopyInformation(mask)
        
        sitk.WriteImage(global_eq_vol, fileName='resized_preprocessed_dataset1/test_data/' + filename)    
        sitk.WriteImage(global_mask, fileName='resized_preprocessed_dataset1/test_data/' + file_mask)
  
#equalizing each training input using histogram        
def train_hist(list_of_train_seg, list_of_train):

    for i in range(50):

        filename = list_of_train[i][43:]
        file_mask = list_of_train_seg[i][43:]
        
        image = sitk.ReadImage(list_of_train[i])
        img_array = sitk.GetArrayFromImage(image)
        flatten_img = img_array.flatten()
        eq_flat = exposure.equalize_hist(flatten_img)
        eq_vol = eq_flat.reshape(img_array.shape)
        
        global_eq_vol = sitk.GetImageFromArray(eq_vol)
        global_eq_vol.CopyInformation(image)
        
        
        #segmentation mask
        mask = sitk.ReadImage(list_of_train_seg[i])
        mask_array = sitk.GetArrayFromImage(mask)
        
        global_mask = sitk.GetImageFromArray(mask_array)
        global_mask.CopyInformation(mask)
        
        sitk.WriteImage(global_eq_vol, fileName='resized_preprocessed_dataset1/training_data/' + filename)    
        sitk.WriteImage(global_mask, fileName='resized_preprocessed_dataset1/training_data/' + file_mask)


#calling the functions   
test_hist(list_of_test_seg, list_of_test) 
train_hist(list_of_train_seg, list_of_train)       

