import numpy as np
from numpy.linalg import norm
import utils.starplus_utils as starp
import scipy.io
import matplotlib.pyplot as plt

subject_ids = ['04847', '04799', '04820', '05675', '05680', '05710']
# subject_ids = ['04847']
tensor_PS = np.empty([64, 64, 8, 16, 80])
labels = np.empty(0)
roi_count = dict()
for id in subject_ids:
    star_plus_data = scipy.io.loadmat('data/data-starplus-' + id + '-v7.mat')
    tensor_PS_id, labels_id = starp.get_labels(star_plus_data)
    tensor_PS = np.concatenate((tensor_PS, tensor_PS_id), axis=-1)
    labels = np.concatenate((labels, labels_id.reshape(-1)))

    tensor_PS = tensor_PS[:, :, :, :, 80:] / norm(tensor_PS)

    # star_plus_data = scipy.io.loadmat('data/data-starplus-' + id + '-v7.mat')
    # roi_tensor, my_color_map, names = starp.visualize_roi(star_plus_data)
    # tmp = []
    # for i in range(int(roi_tensor.max() + 1)):
    #     tmp.append((roi_tensor == i).sum())
    #
    # roi_count[id] = tmp
    # print(tmp)

    # tensor_PS -= tensor_PS.mean(axis=(0, 1, 2, 3))
    tmp = tensor_PS.reshape(-1, tensor_PS.shape[-1])
    for i in range(tmp.shape[1]):
        tmp[tmp[:, i] != 0, i] -= tmp[tmp[:, i] != 0, i].mean()
        tmp[tmp[:, i] != 0, i] /= tmp[tmp[:, i] != 0, i].std()
    tensor_PS = tmp.reshape(*tensor_PS.shape)


    # tensor_PS -= tensor_PS.mean(axis=(0, 1, 2, 3))
    # tensor_PS /= np.std(tensor_PS)

    plt.figure()
    plt.hist(tensor_PS[tensor_PS != 0].reshape(-1), 100)
    plt.title(id)
    plt.show()


#%%

