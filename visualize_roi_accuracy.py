# import
import numpy as np
import tensor.tensor_product_wrapper as tp
from utils.plotting_utils import montage_array, slice_subplots, classification_plots
import matplotlib.pyplot as plt
import similarity_metrics as sm
import scipy.io
import utils.starplus_utils as starp


import os
import pickle


subject_ids = ['05710']

star_plus_data = scipy.io.loadmat('data/data-starplus-' + subject_ids[0] + '-v7.mat')
roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)

roi_accuracies = np.random.rand(len(names))

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'image.interpolation' : None})
plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 200

plt.figure()
plt.bar(np.arange(len(names)), roi_accuracies, color=my_color_map.colors, tick_label=names)
# plt.plot([0, len(names)], [0.5, 0.5], '--')
plt.xlabel('test accuracy')
plt.ylabel('ROI')
plt.show()

