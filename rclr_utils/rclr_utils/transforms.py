import numpy as np

def matrix_closure(mat):
    mat = np.atleast_2d(mat)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

def tensor_rclr(T, branch_lengths=None):
    if len(T.shape) < 2:
        raise ValueError('Tensor is less than 2-dimensions')
    if np.count_nonzero(np.isinf(T)) != 0:
        raise ValueError('Tensor contains either np.inf or -np.inf.')
    if np.count_nonzero(np.isnan(T)) != 0:
        raise ValueError('Tensor contains np.nan or missing.')
    if (T < 0).any():
        raise ValueError('Tensor contains negative values.')
    if len(T.shape) < 3:
        M_tensor_rclr = matrix_rclr(T.transpose().copy(),
                                    branch_lengths=branch_lengths).T
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        return M_tensor_rclr
    else:
        T = T.copy()
        conditions_index = list(range(2, len(T.shape)))
        forward_T = tuple([0] + conditions_index + [1])
        reverse_T = tuple([0] + [conditions_index[-1]]
                          + [1] + conditions_index[:-1])
        T = T.transpose(forward_T)
        M = T.reshape(np.product(T.shape[:len(T.shape) - 1]),
                      T.shape[-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            M_tensor_rclr = matrix_rclr(M, branch_lengths=branch_lengths)
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        return M_tensor_rclr.reshape(T.shape).transpose(reverse_T)

def matrix_rclr(mat, branch_lengths=None):
    mat = np.atleast_2d(np.array(mat))
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if (mat < 0).any():
        raise ValueError('Array Contains Negative Values')
    if np.count_nonzero(np.isinf(mat)) != 0:
        raise ValueError('Data-matrix contains either np.inf or -np.inf')
    if np.count_nonzero(np.isnan(mat)) != 0:
        raise ValueError('Data-matrix contains nans')
    if branch_lengths is not None:
        with np.errstate(divide='ignore'):
            mat = np.log(matrix_closure(matrix_closure(mat) * branch_lengths))
    else:
        with np.errstate(divide='ignore'):
            mat = np.log(matrix_closure(mat))
    mask = [True] * mat.shape[0] * mat.shape[1]
    mask = np.array(mat).reshape(mat.shape)
    mask[np.isfinite(mat)] = False
    lmat = np.ma.array(mat, mask=mask)
    gm = lmat.mean(axis=-1, keepdims=True)
    lmat = (lmat - gm).squeeze().data
    lmat[~np.isfinite(mat)] = np.nan
    return lmat

def mask_value_only(mat):
    mat = np.atleast_2d(np.array(mat))
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    mask = [True] * mat.shape[0] * mat.shape[1]
    mask = np.array(mat).reshape(mat.shape)
    mask[np.isfinite(mat)] = False
    lmat = np.ma.array(mat, mask=mask)
    lmat[~np.isfinite(mat)] = np.nan
    return lmat