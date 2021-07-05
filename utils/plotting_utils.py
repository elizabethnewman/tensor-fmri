import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.axes_grid1 import make_axes_locatable


def montage_array(A, num_col=None, cmap='viridis', names=None):
    # assume A is a 3D tensor for now
    # images from frontal slices

    assert np.ndim(A) == 3, "Montage array only available for third-order tensors"

    m1, m2, m3 = A.shape

    if num_col is None:
        num_col = np.ceil(np.sqrt(m3)).astype(np.int64)

    num_row = np.ceil(m3 / num_col).astype(np.int64)

    C = np.zeros([m1 * num_row, m2 * num_col])

    k = 0
    for p in range(0, C.shape[0], m1):
        for q in range(0, C.shape[1], m2):
            if k >= m3:
                C[p:p + m1, q:q + m2] = np.float64('nan')
                break
            C[p:p + m1, q:q + m2] = A[:, :, k]
            k += 1

    img = plt.imshow(C, cmap=cmap)
    plt.axis('off')
    cb = plt.colorbar()

    if names is not None:
        cb.set_ticks(np.arange(0, len(names)))
        cb.set_ticklabels(names)

    return img


def slice_subplots(A, axis=-1, num_slices=25, title='', num_col=None):
    # assume A is a 3D tensor for now
    # images from frontal slices

    assert np.ndim(A) == 3 or np.ndim(A) == 4, "Montage array only available for third- or fourth-order tensors"

    # permute
    A = np.moveaxis(A, axis, -1)

    m1, m2, m3 = A.shape
    m3 = min(m3, num_slices)

    if num_col is None:
        num_col = np.ceil(np.sqrt(m3)).astype(np.int64)

    num_row = np.ceil(m3 / num_col).astype(np.int64)

    fig = plt.figure()
    count = 0
    for i in range(num_row):
        for j in range(num_col):
            plt.subplot(num_row, num_col, count + 1)
            plt.imshow(A[:, :, count])
            plt.axis('off')
            count += 1
            if count >= A.shape[-1]:
                break
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar()
    fig.suptitle(title)


def classification_plots(error, labels):
    num_classes = error.shape[0]

    plt.figure()
    for i in range(num_classes):
        plt.subplot(1, 2, i + 1)
        for j in range(num_classes):
            plt.plot(error[j, labels == i], 'o', label=j)

        plt.xlabel('image index')
        plt.ylabel('distance score (lower is better)')
        plt.legend()
        plt.title('class %d' % i)