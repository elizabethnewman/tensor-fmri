#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import scipy.io
import utils.starplus_utils as starp
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from tensor.mode_k import modek_unfold
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from scipy.stats import ortho_group
from tensor.tucker import hosvd


# In[2]:


def banded_matrix(dimension,bandwidth):
    A = np.eye(dimension)
    for i in range(0, -bandwidth, -1):
        A += np.eye(dimension, k = i -1)
    for i in range(dimension):
        temp = np.sum(A[i])
        for k in range(dimension):
            if A[i][k] == 1:
                A[i][k] = 1/temp
    return A


# In[3]:


# HaarMatrix utilized for m-product
def haarMatrix(n):
    # n is the power of 2
    if n > 2:
        M = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])
    M_n = np.kron(M, [1, 1])
    M_i = np.sqrt(n/2)*np.kron(np.eye(len(M)), [1, -1])
    M = np.vstack((M_n, M_i))
    return M
def haar_normalized(n):
    M = haarMatrix(n)
    M = M/np.sqrt(np.sum(M[0]))
    return M


# In[4]:


def random_ortho(n):
    return ortho_group.rvs(n)


# In[6]:


def data_dependent_matrix(data):
    G, U, ranks = hosvd(data, 0.00001, ranks = [53,64,64,8,16])
    M_ddm = (U[2].T,U[3].T,U[4].T)
    return M_ddm


# In[ ]:




