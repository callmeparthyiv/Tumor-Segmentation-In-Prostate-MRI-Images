import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
  
#dividing training data and ground truth
path_of_data = 'resized_preprocessed_dataset/test_data/'
list_of_segmentation_data = glob.glob(path_of_data+'*_segmentation.mhd', recursive=True)
list_data = glob.glob(path_of_data+'*.mhd', recursive=True)
list_of_train_data = []

for i in list_data:
    if(i not in list_of_segmentation_data):
        list_of_train_data.append(i)


#function to show the mri image
def show(image_array):
    no_of_slides = image_array.shape[0]
    #counting number of rows and columns
    if(no_of_slides%5 == 0):
        row = (no_of_slides // 5)
    else:
        row = (no_of_slides // 5) + 1

    col = 5

    no_of_subplots = row * col

    #plotting the figures
    fig, axes = plt.subplots(row, col, figsize= (15,15))
    axes = axes.ravel()
    for i in range(no_of_slides):
        axes[i].imshow(image_array[i, :, :], cmap = 'gray')
        axes[i].axis('off')
        
    for j in range(no_of_slides, row * col):
        axes[j].axis('off')  
    
    plt.tight_layout()
    plt.show() 

#reading data for analysis
# for i in range(50):   
image = sitk.ReadImage(list_of_segmentation_data[0])
image_array = sitk.GetArrayViewFromImage(image)

# show(image_array)