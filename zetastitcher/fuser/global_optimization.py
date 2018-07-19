import numpy as np
import pandas as pd

from zetastitcher.gaussian_stitcher.qp.stitching import GaussianStitcherQP as \
    GaussianStitcher
from ..gaussian_stitcher.solver import sparse_lsqr_solver


def absolute_position_global_optimization(df, sdf):
    """Perform global optimization to adjust tile absolute positions.

    The passed dataframe must come with an initial guess of absolute positions.

    Parameters
    ----------
    df : :attr:`.FileMatrix.data_frame`
    sdf : :attr:`.FileMatrix.stitch_data_frame`
    """
    N_DIMS = 3

    data_in = []
    Eye3 = np.eye(N_DIMS)

    for idx, (src, row) in enumerate(sdf.iterrows()):
        dst = row['bname']
        score = row['score']
        score = np.float32(score)
        p = np.array([row['px'], row['py'], row['pz']], dtype=np.float32)
        data_in.append((src, dst, Eye3 * score, p))

    cond = (df['X'] == 0) & (df['Y'] == 0) & (df['Z'] == 0)
    v_origin = df[cond].index[0]  # top-left tile

    stitcher = GaussianStitcher(
        n_dims=N_DIMS,
        solver='cvxopt'
    )
    node2coordinates, digraph = stitcher.stitch(data_in, v_origin)

    laplacian_df = pd.DataFrame(node2coordinates).transpose()

    keys = ['Xs', 'Ys', 'Zs']  # do not change order
    df[keys] = laplacian_df
    df[keys] -= df[keys].min()
    df[keys] = df[keys].apply(np.round).astype(int)
    df['Xs_end'] = df['Xs'] + df['xsize']
    df['Ys_end'] = df['Ys'] + df['ysize']
    df['Zs_end'] = df['Zs'] + df['nfrms']
