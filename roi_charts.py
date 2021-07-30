#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:49:28 2021

@author: katie
"""

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

#not the same indices used for the charts in the poster, but if you change the indices, then you can get the figures for the poster!

subject_ids = ['04847']
star_plus_data = scipy.io.loadmat('data/data-starplus-' + subject_ids[0] + '-v7.mat')
roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)
df = joblib.load('results/kk_testing_roi_04847_results.py')
roi_accuracies = df.loc[df['M'] == 'ROI-Haar-Haar']['accuracy'].take([20,18,15,8,0])
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'image.interpolation' : None})
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['figure.dpi'] = 200
plt.figure()
plt.bar(np.arange(5), roi_accuracies, color=np.array([my_color_map.colors[i] for i in [24,18,15,8,0]]), tick_label=np.array([names[i] for i in [19,18,12,9,0]]))
# plt.plot([0, len(names)], [0.5, 0.5], '--')
plt.ylabel('test accuracy')
plt.xlabel('ROI')
plt.ylim(ymin=0.6,ymax=1.0)
plt.show()

subject_ids = ['05710']
star_plus_data = scipy.io.loadmat('data/data-starplus-' + subject_ids[0] + '-v7.mat')
roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)
df = joblib.load('results/kk_testing_roi_05710_results.py')
roi_accuracies = df.loc[df['M'] == 'ROI-Haar-Haar']['accuracy'].take([20,18,12,8,0])
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'image.interpolation' : None})
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['figure.dpi'] = 200
plt.figure()
plt.bar(np.arange(5), roi_accuracies, color=np.array([my_color_map.colors[i] for i in [24,18,15,7,0]]), tick_label=np.array([names[i] for i in [19,18,12,9,0]]))
# plt.plot([0, len(names)], [0.5, 0.5], '--')
plt.ylabel('test accuracy')
plt.xlabel('ROI')
plt.ylim(ymin=0.6,ymax=1.0)
plt.show()