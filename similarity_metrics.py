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


def ssim_slices(A, B, axis=1):
    A = np.swapaxes(A, 0, axis)
    B = np.swapaxes(B, 0, axis)

    results = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        results[i] = ssim(A[i], B[i])

    return results

# signal to noise ratio
# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# structural similarity metrics
# https://en.wikipedia.org/wiki/Structural_similarity#:~:text=The%20structural%20similarity%20index%20measure,the%20similarity%20between%20two%20images.
# https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py
# http://mubeta06.github.io/python/sp/_modules/sp/ssim.html

# def ssim(img1, img2, cs_map=False):
#     """Return the Structural Similarity Map corresponding to input images img1
#     and img2 (images are assumed to be uint8)
#
#     This function attempts to mimic precisely the functionality of ssim.m a
#     MATLAB provided by the author's of SSIM
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
#     """
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     size = 11
#     sigma = 1.5
#     window = gauss.fspecial_gauss(size, sigma)
#     K1 = 0.01
#     K2 = 0.03
#     L = 255  # bitdepth of image
#     C1 = (K1 * L) ** 2
#     C2 = (K2 * L) ** 2
#     mu1 = signal.fftconvolve(window, img1, mode='valid')
#     mu2 = signal.fftconvolve(window, img2, mode='valid')
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
#     sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
#     sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
#     if cs_map:
#         return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                              (sigma1_sq + sigma2_sq + C2)),
#                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
#     else:
#         return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))


