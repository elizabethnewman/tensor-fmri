# import
import numpy as np
import tensor.tensor_product_wrapper as tp
from utils.plotting_utils import montage_array, slice_subplots, classification_plots
import matplotlib.pyplot as plt
import similarity_metrics as sm
import scipy.io
import utils.starplus_utils as starp
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import transform_matrices as tm

import os
import pickle

# ==================================================================================================================== #
# saving options
save = False
filename = 'local_tsvdm'

# plotting options
plot = False

# ==================================================================================================================== #
# choose product type {'m'}
prod_type = 'm'
ortho = True


# ==================================================================================================================== #
# define projection
def projection(A, U, prod_type='m', M=None, dim_list=(), ortho=True):
    training_coeff = tp.ten_prod(tp.ten_tran(U, prod_type=prod_type), A, prod_type=prod_type, M=M, dim_list=dim_list,
                                 ortho=ortho)
    return tp.ten_prod(U, training_coeff, prod_type=prod_type, M=M, dim_list=dim_list, ortho=ortho)


# ==================================================================================================================== #
#%%
# for reproducibility
np.random.seed(20)

# load data
# subject_ids = ['04847', '04799', '04820', '05675', '05710']
subject_ids = ['05710']
tensor_PS = np.empty([64, 64, 8, 16, 80])
labels = np.empty(0)
for id in subject_ids:
    star_plus_data = scipy.io.loadmat('data/data-starplus-' + id + '-v7.mat')
    tensor_PS_id, labels_id = starp.get_labels(star_plus_data)
    tensor_PS = np.concatenate((tensor_PS, tensor_PS_id), axis=-1)
    labels = np.concatenate((labels, labels_id.reshape(-1)))

    star_plus_data = scipy.io.loadmat('data/data-starplus-' + id + '-v7.mat')
    roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)

tensor_PS = tensor_PS[:, :, :, :, 80:]  / norm(tensor_PS)


# split
training_data, test_data, training_labels, test_labels = train_test_split(np.moveaxis(tensor_PS, -1, 0), labels,
                                                                          test_size=0.33, random_state=42)
training_data = np.moveaxis(training_data, 0, 1)
test_data = np.moveaxis(test_data, 0, 1)

roi_tensor = np.repeat(np.repeat(roi_tensor.reshape(64, 64, 8, 1, 1), 16, axis=3), training_data.shape[1], axis=4)
roi_tensor = np.moveaxis(np.moveaxis(roi_tensor, -1, 0), 0, 1)

print('Loaded data...')

# ==================================================================================================================== #
#%%
# choose transformations for M
M = []
for i in range(2, training_data.ndim):
    if i == 2:
        M.append(tm.roi_left_singular_matrix(training_data, i, roi_tensor, 4))
    else:
        M.append(tm.haar_matrix(training_data.shape[i], normalized=True))

print('Formed M...')
# ==================================================================================================================== #
#%%
# form local t-svd
num_classes = len(np.unique(training_labels))
range_k = (2, 3)

U = []
for i in range(num_classes):
    u, _, _, _ = tp.ten_svd(training_data[:, training_labels == i], k=max(range_k), prod_type='m', M=M, ortho=ortho)
    U.append(u)

print('Formed local t-SVDM...')
# ==================================================================================================================== #
# compute results on training and test data
training_error = np.zeros([num_classes, training_data.shape[1], len(range_k)])
test_error = np.zeros([num_classes, test_data.shape[1], len(range_k)])
for j, k in enumerate(range_k):
    for i in range(num_classes):
        training_projection = projection(training_data, U[i][:, :k], prod_type='m', M=M)
        training_error[i, :, j] = sm.frobenius_metric(training_data, training_projection, axis=1)

        test_projection = projection(test_data, U[i][:, :k], prod_type='m', M=M)
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
    pickle.dump(stored_results, open('results/' + filename + '-' + prod_type + '.py', 'wb'))

# plot results
if plot:
    plt.figure()
    classification_plots(training_error[:, :, 0], training_labels)
    plt.show()

