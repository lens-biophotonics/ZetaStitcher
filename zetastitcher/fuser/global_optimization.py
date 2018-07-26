import numpy as np
import pandas as pd

from ..gaussian_stitcher.qp.stitching import GaussianStitcherQP as \
    GaussianStitcher

from ..gaussian_stitcher.common import ConstraintTuple


def absolute_position_global_optimization(df, xcfm):
    """Perform global optimization to adjust tile absolute positions.

    The passed dataframe must come with an initial guess of absolute positions.

    Parameters
    ----------
    df : :attr:`.FileMatrix.data_frame`
    sdf : :attr:`.FileMatrix.stitch_data_frame`
    """
    N_DIMS = 3
    CTND = ConstraintTuple(N_DIMS)
    Eye3 = np.eye(N_DIMS)

    data_in = []

    xcorr_options = xcfm.xcorr_options
    max_dx = xcorr_options['max_dx']
    max_dy = xcorr_options['max_dy']
    max_dz = xcorr_options['max_dz']
    overlap = xcorr_options['overlap_h']

    sdf = xcfm.stitch_data_frame

    for idx, (src, row) in enumerate(sdf.iterrows()):
        dst = row['bname']
        score = row['score']
        score = np.float32(score)
        p = np.array([row['px'], row['py'], row['pz']], dtype=np.float32)
        src_tile = df.loc[src]
        if row['axis'] == 1:
            lb = np.array(
                [-max_dx, src_tile['ysize'] - overlap - max_dy, -max_dz])
            ub = np.array(
                [max_dx, src_tile['ysize'] - overlap + max_dy, max_dz])
        elif row['axis'] == 2:
            lb = np.array(
                [src_tile['xsize'] - overlap - max_dy, -max_dx, -max_dz])
            ub = np.array(
                [src_tile['xsize'] - overlap + max_dy, max_dx, max_dz])
        else:
            raise ValueError('invalid axis {}'.format(row['axis']))
        data_in.append(CTND(src, dst, Eye3 * score, p, lb=lb, ub=ub))

    cond = (df['X'] == 0) & (df['Y'] == 0) & (df['Z'] == 0)
    v_origin = df[cond].index[0]  # top-left tile

    stitcher = GaussianStitcher(
        n_dims=N_DIMS,
        solver='cvxpy'
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
