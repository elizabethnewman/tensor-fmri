import numpy as np
from numpy.linalg import norm
import scipy.io
import utils.starplus_utils as starp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from utils.plotting_utils import montage_array
import time
from utils.general_utils import rescale

star_plus_data = scipy.io.loadmat('../../fmri_project/data-starplus-04847-v7.mat')

roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)

plt.figure(1)
montage_array(roi_tensor, cmap=my_color_map, names=names)
# plt.savefig('/Users/elizabethnewman/Desktop/brain1.jpg')
plt.show()

tensor_PS, labels = starp.get_labels(star_plus_data)

tensor_PS  = tensor_PS / norm(tensor_PS)
# tensor_PS = rescale(tensor_PS, 1, 64)

plt.figure(2)
montage_array(tensor_PS[:, :, :, 0, 1], cmap='viridis')
plt.show()


