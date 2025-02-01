import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
image = sitk.ReadImage('dataset/training_data/Case00_segmentation.mhd')
image_array = sitk.GetArrayViewFromImage(image)

depth, height, width = image_array.shape

slice_z = image_array[depth//2, :, :]
slice_y = image_array[:, height//2, :]
slice_x = image_array[:, :, width//2]

fig, axes = plt.subplots(1, 3, figsize = (15,5))

axes[0].imshow(slice_z, cmap = 'grey')
axes[0].axis('off')

axes[1].imshow(slice_y, cmap='gray')
# axes[1].set_title('Sagittal Slice (Y)')
axes[1].axis('off')

axes[2].imshow(slice_x, cmap='gray')
# axes[2].set_title('Coronal Slice (X)')
axes[2].axis('off')


# plt.tight_layout()
plt.show()
print(image)

