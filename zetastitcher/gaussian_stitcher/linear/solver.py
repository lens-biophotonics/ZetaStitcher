from scipy.sparse.linalg import lsqr

def sparse_lsqr_solver(A, b):
    x, istop, itn, r1norm, r2norm, *args = lsqr(A, b)
    return x

