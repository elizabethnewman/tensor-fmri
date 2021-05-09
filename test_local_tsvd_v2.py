import numpy as np
from data.synthetic_data import stripes
import tensor.t_product as tprod
from utils.plotting_utils import montage_array, slice_subplots
import matplotlib.pyplot as plt
import math

# for reproducibility
np.random.seed(20)

# load data
img_shape = (8, 8)
training_data, training_labels = stripes(num_images_per_class=100, img_shape=img_shape)
test_data, test_labels = stripes(num_images_per_class=10, img_shape=img_shape)

# permute such that lateral slices are the images
training_data = training_data.transpose(0, 2, 1)
test_data = test_data.transpose(0, 2, 1)

# form local t-svd
num_classes = len(np.unique(training_labels))
k = 4

U = []
for i in range(num_classes):
    u, s, vh = tprod.t_svd(training_data[:, training_labels == i, :], k)
    U.append(u)

# compute results on training and test data
training_error = np.zeros([num_classes, training_data.shape[1]])
test_error = np.zeros([num_classes, test_data.shape[1]])
for i in range(num_classes):
    training_coeff = tprod.t_product(tprod.t_transpose(U[i]), training_data)
    training_projection = tprod.t_product(U[i], training_coeff)
    training_error[i, :] = np.sqrt(np.sum((training_data - training_projection) ** 2, axis=(0, 2)).squeeze())

    test_coeff = tprod.t_product(tprod.t_transpose(U[i]), test_data)
    test_projection = tprod.t_product(U[i], test_coeff)
    test_error[i, :] = np.sqrt(np.sum((test_data - test_projection) ** 2, axis=(0, 2)).squeeze())


# classification
training_predicted_classes = np.argmin(training_error, axis=0).reshape(-1)
test_predicted_classes = np.argmin(test_error, axis=0).reshape(-1)

# results
training_num_correct = np.sum(training_predicted_classes == training_labels)
training_accuracy = training_num_correct / training_data.shape[1]

test_num_correct = np.sum(test_predicted_classes == test_labels)
test_accuracy = test_num_correct / test_data.shape[1]


# visualizations
for i in range(num_classes):
    slice_subplots(U[i], axis=1, title='basis: class' + str(i))
    plt.show()

# j = 0
# A = training_data[:, training_labels == j, :]
# for i in range(num_classes):
#     A_projected = tprod.t_product(U[i], tprod.t_product(tprod.t_transpose(U[i]), A))
#     visualize_lateral_slices(A_projected, title='projection: class' + str(j) + ' onto class' + str(i))
#     visualize_lateral_slices(np.abs(A - A_projected), title='abs. diff: class ' + str(j) + ' onto class' + str(i))
