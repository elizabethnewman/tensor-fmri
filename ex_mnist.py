# import
import numpy as np
from data.mnist import load_tensor
import tensor.tensor_product_wrapper as tp
from data.utils import rescale, normalize
from utils.plotting_utils import montage_array, slice_subplots
import matplotlib.pyplot as plt
import similarity_metrics as sm
import os
import seaborn as sns


def projection(A, U, prod_type='t'):
    training_coeff = tp.ten_prod(tp.ten_tran(U, prod_type=prod_type), A, prod_type=prod_type)
    return tp.ten_prod(U, training_coeff, prod_type=prod_type)


#%%
# for reproducibility
np.random.seed(20)

# load data
os.chdir('/Users/elizabethnewman/Documents/Emory_Work/REU2021/tensor-fmri/')
num_classes = 2
training_data, training_labels, test_data, test_labels = load_tensor(num_per_class_train=100, num_per_class_test=9,
                                                                     num_classes=num_classes, path="data/")

# permute such that lateral slices are the images
training_data = training_data.transpose(0, 2, 1)
test_data = test_data.transpose(0, 2, 1)

# rescale and normalize
training_data = normalize(rescale(training_data))
test_data = normalize(rescale(test_data))


#%% Form local t-SVD basis

U = []
for i in range(num_classes):
    u, _, _, _ = tp.ten_svd(training_data[:, training_labels == i], k=2)
    U.append(u)

#%% Compute results
training_error = np.zeros([num_classes, training_data.shape[1]])
test_error = np.zeros([num_classes, test_data.shape[1]])

for i in range(num_classes):
    training_projection = projection(training_data, U[i])
    training_error[i, :] = sm.frobenius_metric(training_data, training_projection, axis=1)

    test_projection = projection(test_data, U[i])
    test_error[i, :] = sm.frobenius_metric(test_data, test_projection, axis=1)

# classification
training_predicted_classes = np.argmin(training_error, axis=0).reshape(-1, 1)
test_predicted_classes = np.argmin(test_error, axis=0).reshape(-1, 1)

# results
training_num_correct = np.sum(training_predicted_classes == training_labels.reshape(-1, 1), axis=0)
training_accuracy = training_num_correct / training_data.shape[1]

test_num_correct = np.sum(test_predicted_classes == test_labels.reshape(-1, 1), axis=0)
test_accuracy = test_num_correct / test_data.shape[1]

print('k = %d: train accuracy = %0.2f\ttest accuracy = %0.2f' % (2, 100 * training_accuracy, 100 * test_accuracy))

#%% Confusion matrix

conf_mat = sm.confusion_matrix(training_predicted_classes, training_labels)

plt.figure()
sns.heatmap(conf_mat, annot=True, fmt="d")
plt.show()
