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
import joblib

subject_ids = ['04847','04799','05710']
#subject_ids = ['04799']
for i in range(0,3):
    star_plus_data = scipy.io.loadmat('data/data-starplus-' + subject_ids[i] + '-v7.mat')
    roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)
    df = joblib.load('results/kk_testing_roi_{}_results.py'.format(subject_ids[i]))
    roi_accuracies = df.loc[df['M'] == 'ROI-ROI-ROI']['accuracy']
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'image.interpolation' : None})
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 200
    plt.figure()
    plt.barh(np.arange(len(names)), roi_accuracies, color=my_color_map.colors, tick_label=names)
    # plt.plot([0, len(names)], [0.5, 0.5], '--')
    plt.ylabel('ROI')
    plt.xlabel('Test Accuracy')
    plt.xlim(xmin=0.5,xmax=1.0)
    #plt.show()
    plt.savefig('results/roi_roi_roi_{}_xmin0.5.png'.format(subject_ids[i]))

for i in range(0,3):
    star_plus_data = scipy.io.loadmat('data/data-starplus-' + subject_ids[i] + '-v7.mat')
    roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)
    df = joblib.load('results/kk_testing_roi_{}_results.py'.format(subject_ids[i]))
    roi_accuracies = df.loc[df['M'] == 'ROI-DDM-DDM']['accuracy']
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'image.interpolation' : None})
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 200
    plt.figure()
    plt.barh(np.arange(len(names)), roi_accuracies, color=my_color_map.colors, tick_label=names)
    # plt.plot([0, len(names)], [0.5, 0.5], '--')
    plt.ylabel('ROI')
    plt.xlabel('Test Accuracy')
    plt.xlim(xmin=0.5,xmax=1.0)
    #plt.show()
    plt.savefig('results/roi_ddm_ddm_{}_xmin0.5.png'.format(subject_ids[i]))