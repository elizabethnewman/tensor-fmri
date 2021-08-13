import numpy as np
from numpy.linalg import norm
import math
import cv2
from utils.general_utils import prod


def frobenius_metric(A, B, axis=None):
    # frobenius distance between tensors A and B, keeping one dimension fixed
    A = np.swapaxes(A, 0, axis)
    B = np.swapaxes(B, 0, axis)
    all_axis = tuple([i for i in range(1, A.ndim)])
    return np.sqrt(np.sum((A - B) ** 2, axis=all_axis)) / prod(A.shape[1:])


def cosine_metric(A, B, axis=None):
    # frobenius distance between tensors A and B, keeping one dimension fixed
    A = np.swapaxes(A, 0, axis)
    B = np.swapaxes(B, 0, axis)
    all_axis = [i for i in range(1, A.ndim)]
    A_nrm = np.sqrt(np.sum(A ** 2, axis=tuple(all_axis)))
    B_nrm = np.sqrt(np.sum(B ** 2, axis=tuple(all_axis)))

    return 1 - np.sum(A * B, axis=tuple(all_axis)) / (A_nrm * B_nrm)


def confusion_matrix(predicted_labels, true_labels):
    # actual class x predicted class

    predicted_labels = predicted_labels.reshape(-1)
    true_labels = true_labels.reshape(-1)

    num_classes = len(np.unique(true_labels))
    num_examples = len(predicted_labels)
    conf_mat = np.zeros([num_classes, num_classes], dtype=np.int8)

    for i in range(num_examples):
        conf_mat[int(true_labels[i]), int(predicted_labels[i])] += 1

    return conf_mat

