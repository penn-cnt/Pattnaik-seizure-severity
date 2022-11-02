import numpy as np

def format_network(upper_tris, n):
    '''
    This function takes a 2d matrix (time by upper triangular arrays) 
    and reformats it into a 3d matrix (time by n by n)
    '''

    if not isinstance(upper_tris, np.ndarray):
        upper_tris = np.array(upper_tris)
    (n_samples, n_utri) = upper_tris.shape
    
    assert n_utri == n * (n - 1) / 2

    out_mat = np.zeros((n_samples, n, n))

    for i_samples in range(n_samples):
        out_mat[i_samples][np.triu_indices(n, 1)] = upper_tris[i_samples, :]
        out_mat[i_samples] = out_mat[i_samples] + out_mat[i_samples].T

    return out_mat
