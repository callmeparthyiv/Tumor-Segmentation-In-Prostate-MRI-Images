import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from ipywidgets import interact

image = sitk.ReadImage('dataset/training_data/Case01_segmentation.mhd')
image_array = sitk.GetArrayViewFromImage(image)

row = 3
col = 3


no_of_subplots = row * col
no_of_slides = image_array.shape[0]
step_size = no_of_slides // no_of_subplots
plot_range = no_of_subplots * step_size
start_stop = int((no_of_slides - plot_range) / 2)


fig, axes = plt.subplots(row, col, figsize= (10,10))

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axes.flat[idx].imshow(image_array[img,:, :], cmap = 'gray')
    axes.flat[idx].axis('off')
    
plt.tight_layout()
plt.show()
