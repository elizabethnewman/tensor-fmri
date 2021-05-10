import tensor.f_product as fprod
import tensor.t_product as tprod
import tensor.c_product as cprod
import tensor.m_product as mprod


def get_prod_type(prod_type):
    if any(prod_type == s for s in ('tprod', 't_prod', 't_product', 'fft', 't')):
        prod_type = 't'
    elif any(prod_type == s for s in ('fprod', 'f_prod', 'f_product', 'facewise', 'facewise_product', 'f')):
        prod_type = 'f'
    elif any(prod_type == s for s in ('cprod', 'c_prod', 'c_product', 'dct', 'c')):
        prod_type = 'c'
    elif any(prod_type == s for s in ('mprod', 'm_prod', 'm_product', 'm')):
        prod_type = 'm'
    else:
        raise ValueError(prod_type + ' not yet implemented')

    return prod_type


def ten_transform(A, prod_type='tprod', M=None):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        C = tprod.t_transform(A)

    elif prod_type == 'f':
        C = fprod.f_transform(A)

    elif prod_type == 'c':
        C = cprod.c_transform(A)

    elif prod_type == 'm':
        C = mprod.m_transform(A, M=M)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return C


def ten_itransform(A, prod_type='tprod', M=None):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        C = tprod.t_itransform(A)

    elif prod_type == 'f':
        C = fprod.f_itransform(A)

    elif prod_type == 'c':
        C = cprod.c_itransform(A)

    elif prod_type == 'm':
        C = mprod.m_itransform(A, M=M)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return C


def ten_prod(A, B, prod_type='tprod', M=None):

    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        C = tprod.t_prod(A, B)

    elif prod_type == 'f':
        C = fprod.f_prod(A, B)

    elif prod_type == 'c':
        C = cprod.c_prod(A, B)

    elif prod_type == 'm':
        C = mprod.m_prod(A, B, M)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return C


def ten_tran(A, prod_type='tprod'):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        AT = tprod.t_tran(A)

    elif prod_type == 'f':
        AT = fprod.f_tran(A)

    elif prod_type == 'c':
        AT = cprod.c_tran(A)

    elif prod_type == 'm':
        AT = mprod.m_tran(A)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return AT


def ten_eye(shape_I, prod_type='tprod', M=None):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        AT = tprod.t_eye(shape_I)

    elif prod_type == 'f':
        AT = fprod.f_eye(shape_I)

    elif prod_type == 'c':
        AT = cprod.c_eye(shape_I)

    elif prod_type == 'm':
        AT = mprod.m_eye(shape_I, M)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return AT


def ten_svd(A, k=None, prod_type='tprod', M=None):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        u, s, vh, stats = tprod.t_svd(A, k=k)

    elif prod_type == 'f':
        u, s, vh, stats = fprod.f_svd(A, k=k)

    elif prod_type == 'c':
        u, s, vh, stats = cprod.c_svd(A, k=k)

    elif prod_type == 'm':
        u, s, vh, stats = mprod.m_svd(A, M, k=k)

    else:
        raise ValueError(prod_type + ' not yet implemented')

    return u, s, vh, stats


def ten_svdII(A, gamma, prod_type='tprod', M=None, compress_UV=False, return_spatial=True, implicit_rank=None):
    prod_type = get_prod_type(prod_type)

    if prod_type == 't':
        u, s, vh, stats = tprod.t_svdII(A, gamma, compress_UV=compress_UV, return_spatial=return_spatial,
                                        implicit_rank=implicit_rank)

    elif prod_type == 'f':
        u, s, vh, stats = fprod.f_svdII(A, gamma, compress_UV=compress_UV, implicit_rank=implicit_rank)

    elif prod_type == 'c':
        u, s, vh, stats = cprod.c_svdII(A, gamma, compress_UV=compress_UV, return_spatial=True,
                                        implicit_rank=implicit_rank)

    elif prod_type == 'm':
        u, s, vh, stats = mprod.m_svdII(A, M, gamma, compress_UV=compress_UV, return_spatial=True,
                                        implicit_rank=implicit_rank)

    else:
        raise ValueError(prod_type + ' nprojection(A, U, prod_type)ot yet implemented')

    return u, s, vh, stats