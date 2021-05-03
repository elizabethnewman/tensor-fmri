import numpy as np
import matplotlib.pyplot as plt


def montage_array(A, num_col=None, cmap='viridis', names=None):
    # assume A is a 3D tensor for now

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

