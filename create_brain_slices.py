import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# https://github.com/vb100/Visualize-3D-MRI-Scans-Brain-case/blob/master/playground.ipynb

image_path = "/Users/elizabethnewman/Desktop/Visualize-3D-MRI-Scans-Brain-case-master/data/images/BRATS_001.nii.gz"
image_obj = nib.load(image_path)
print(f'Type of the image {type(image_obj)}')


# Extract data as numpy ndarray
image_data = image_obj.get_fdata()
type(image_data)

height, width, depth, channels = image_data.shape
print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}, channels:{channels}")


# Select random layer number
maxval = 154
i = np.random.randint(0, maxval)
# Define a channel to look at
channel = 0
print(f"Plotting Layer {i} Channel {channel} of Image")
plt.imshow(image_data[-1, :, :, channel].squeeze(), cmap='gray')
plt.axis('off')
plt.show()