# import
import numpy as np
import tensor.tensor_product_wrapper as tp
from tensor.utils import reshape
from utils.plotting_utils import montage_array, slice_subplots, classification_plots
import matplotlib.pyplot as plt
import similarity_metrics as sm
import scipy.io
import utils.starplus_utils as starp
from numpy.linalg import norm, svd
from sklearn.model_selection import train_test_split
import os
import pickle


# ==================================================================================================================== #
# saving options
save = True
filename = 'local_svd'

# plotting options
plot = True


# ==================================================================================================================== #
# define projection
def projection(A, U):
    training_coeff = U.T @ A
    return U @ training_coeff


# ==================================================================================================================== #
# for reproducibility
np.random.seed(20)

# load data
star_plus_data = scipy.io.loadmat('data/data-starplus-04847-v7.mat')
tensor_PS, labels = starp.get_labels(star_plus_data)
tensor_PS = tensor_PS / norm(tensor_PS)

# vectorize
tensor_PS = np.moveaxis(tensor_PS, -1, 0)
tensor_PS = reshape(tensor_PS, [tensor_PS.shape[0], -1])
tensor_PS = tensor_PS.T

# split
training_data, test_data, training_labels, test_labels = train_test_split(np.moveaxis(tensor_PS, -1, 0), labels,
                                                                          test_size=0.33, random_state=42)
training_data = np.moveaxis(training_data, 0, 1)
test_data = np.moveaxis(test_data, 0, 1)


# ==================================================================================================================== #
# form local t-svd
num_classes = len(np.unique(training_labels))
range_k = (2, 3)

U = []
for i in range(num_classes):
    u, _, _ = svd(training_data[:, training_labels == i], full_matrices=False)
    U.append(u[:, :max(range_k)])


# ==================================================================================================================== #
# compute results on training and test data
training_error = np.zeros([num_classes, training_data.shape[1], len(range_k)])
test_error = np.zeros([num_classes, test_data.shape[1], len(range_k)])

for j, k in enumerate(range_k):
    for i in range(num_classes):
        training_projection = projection(training_data, U[i][:, :k])
        training_error[i, :, j] = sm.frobenius_metric(training_data, training_projection, axis=1)

        test_projection = projection(test_data, U[i][:, :k])
        test_error[i, :, j] = sm.frobenius_metric(test_data, test_projection, axis=1)


# classification
training_predicted_classes = np.argmin(training_error, axis=0).reshape(-1, len(range_k))
test_predicted_classes = np.argmin(test_error, axis=0).reshape(-1, len(range_k))

# results
training_num_correct = np.sum(training_predicted_classes == training_labels.reshape(-1, 1), axis=0)
training_accuracy = training_num_correct / training_data.shape[1]

test_num_correct = np.sum(test_predicted_classes == test_labels.reshape(-1, 1), axis=0)
test_accuracy = test_num_correct / test_data.shape[1]

for j, k in enumerate(range_k):
    print('k = %d: train accuracy = %0.2f\ttest accuracy = %0.2f' %
          (k, 100 * training_accuracy[j], 100 * test_accuracy[j]))

if save:
    if not os.path.exists('results/'):
        os.makedirs('results/')
    stored_results = {'training_error' : training_error,
                      'test_error' : test_error,
                      'training_accuracy' : training_accuracy,
                      'test_accuracy' : test_accuracy}
    pickle.dump(stored_results, open('results/' + filename + '.py', 'wb'))

# plot results
if plot:
    plt.figure()
    classification_plots(training_error[:, :, 0], training_labels)
    plt.show()
