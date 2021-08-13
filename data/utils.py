import numpy as np
import math


def rescale(data):
    # second-dimension are different images

    # reshape
    data = np.swapaxes(data, 0, 1)
    permuted_shape = data.shape
    data = data.reshape(data.shape[0], -1)

    # rescale each image between 0 and 1
    min_data = data.min(axis=1).reshape(-1, 1)
    max_data = data.max(axis=1).reshape(-1, 1)
    data = (data - min_data) / (max_data - min_data)

    # reshape
    data = data.reshape(permuted_shape)
    data = np.swapaxes(data, 0, 1)
    return data


def normalize(data):
    # adpated from https://github.com/XtractOpen/Meganet.m/blob/master/utils/normalizeData.m

    # rescale
    numel_features = data.size // data.shape[1]

    # make the mean of voxels in each image 0 with standard deviation of 1
    data = data - data.mean(axis=(0, 1))
    data = data / np.maximum(data.std(axis=(0, 1)), 1 / math.sqrt(numel_features))

    return data

