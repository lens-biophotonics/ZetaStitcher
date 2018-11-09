from scipy.sparse.linalg import lsqr

def sparse_lsqr_solver(A, b):
    x, istop, itn, r1norm, r2norm, *args = lsqr(A, b)
    print('SOLVER LOG istop:', istop, 'itn:', itn, 'r1norm:', r1norm, 'r2norm:', r2norm)
    return x

