import numpy as np
import pygmo as pg
import pandas as pd

from scipy.ndimage.interpolation import shift


class TileDisplacementLeastSquares:
    def __init__(self, xcorr_options, frame_shapes,
                 p_ab_1, score_1, p_ab_2, score_2):
        self.xcorr_options = xcorr_options
        self.frame_shapes = frame_shapes
        self.shape = p_ab_1.shape
        self.p_ab_1 = p_ab_1
        self.score_1 = score_1
        self.p_ab_2 = p_ab_2
        self.score_2 = score_2

    def get_bounds(self):
        n_of_tiles = np.prod(self.shape[0:2])
        xsize = self.shape[1]

        mins = [0, 0, 0]
        mins[0] = -self.xcorr_options['max_dz']
        mins[1] = None
        mins[2] = -self.xcorr_options['max_dx']
        mins = mins * n_of_tiles
        # overwrite None
        mins[1::3] = self.frame_shapes[:, 0] - (
            self.xcorr_options['overlap_v'] + self.xcorr_options['max_dy'])
        mins[:3] = [0, 0, 0]  # first tile is fixed
        # overwrite first row with horizontal shifts
        temp = [0, 0, 0]
        temp[0] = -self.xcorr_options['max_dz']
        temp[1] = -self.xcorr_options['max_dx']
        temp[2] = None
        mins[3:3 * xsize] = temp * (xsize - 1)
        # overwrite None
        mins[5:3 * xsize:3] = (self.frame_shapes[1:xsize, 1]
                               - self.xcorr_options['overlap_h']
                               - self.xcorr_options['max_dy'])

        maxs = [0, 0, 0]
        maxs[0] = self.xcorr_options['max_dz']
        maxs[1] = None
        maxs[2] = self.xcorr_options['max_dx']
        maxs = maxs * n_of_tiles
        # overwrite None
        maxs[1::3] = (self.frame_shapes[:, 0] - self.xcorr_options['overlap_v']
                      + self.xcorr_options['max_dy'])
        maxs[:3] = [0, 0, 0]  # first tile is fixed
        # overwrite first row with horizontal shifts
        temp[0] = self.xcorr_options['max_dz']
        temp[1] = self.xcorr_options['max_dx']
        temp[2] = None
        maxs[3:3 * xsize] = temp * (xsize - 1)
        # overwrite None
        maxs[5:3 * xsize:3] = (self.frame_shapes[1:xsize, 1]
                               - self.xcorr_options['overlap_h']
                               + self.xcorr_options['max_dy'])

        mins = [int(x) for x in mins]
        maxs = [int(x) for x in maxs]

        bounds = (mins, maxs)
        return bounds

    def fitness(self, x):
        t = decision_vector_to_tile_coords(x, self.shape)

        sums = 0

        # add shifts along x
        shifted = shift(t, (0, -1, 0), cval=np.NaN)
        norms = np.linalg.norm(
            np.nan_to_num((shifted - t - self.p_ab_2)), axis=-1)
        sums += np.sum(self.score_2 * norms**2)

        # add shifts along y
        shifted = shift(t, (-1, 0, 0), cval=np.NaN)
        norms = np.linalg.norm(
            np.nan_to_num((shifted - t - self.p_ab_1)), axis=-1)
        sums += np.sum(self.score_1 * norms**2)

        return [sums]

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


def decision_vector_to_tile_coords(x, shape):
    """Covert decision vector `x` to tile coordinates.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Decision vector. First row: displacement vector from adjacent left
        tile. Other rows: displacement vector from adjacent top tile.
    shape : list
        `[ysize, xsize, 3]` where `ysize` and `xsize` are the mosaic size.

    Returns
    -------

    """
    t = x.reshape(shape)
    t[0, ...] = np.cumsum(t[0, ...], axis=0)
    t = np.cumsum(t, axis=0)
    return t


def absolute_position_global_optimization(df, sdf, xcorr_options):
    """Perform global optimization to adjust tile absolute positions.

    The passed dataframe must come with an initial guess of absolute positions.

    Parameters
    ----------
    df : :attr:`.FileMatrix.data_frame`
    sdf : :attr:`.FileMatrix.stitch_data_frame`
    """
    # mosaic size
    xsize = df['X'].unique().size
    ysize = df['Y'].unique().size

    idx = df.sort_values(['Z', 'Y', 'X']).index
    frame_shapes = np.array(df.loc[idx, ['ysize', 'xsize']])

    # shifts along Y
    temp = sdf.loc[idx, ['pz', 'py', 'px', 'axis', 'score']]
    try:
        temp = temp[temp['axis'] == 1].loc[idx]
        p_ab_1 = np.array(temp[['pz', 'py', 'px']]).reshape(ysize, xsize, 3)
        score_1 = np.nan_to_num(np.array(temp['score']).reshape(ysize, xsize))
    except KeyError:
        p_ab_1 = np.zeros((ysize, xsize, 3))
        score_1 = np.zeros((ysize, xsize))

    shape = p_ab_1.shape

    # shifts along X
    temp = sdf.loc[idx, ['pz', 'py', 'px', 'axis', 'score']]
    try:
        temp = temp[temp['axis'] == 2].loc[idx]
        p_ab_2 = np.array(temp[['pz', 'py', 'px']]).reshape(ysize, xsize, 3)
        score_2 = np.nan_to_num(np.array(temp['score']).reshape(ysize, xsize))
    except KeyError:
        p_ab_2 = np.zeros((ysize, xsize, 3))
        score_2 = np.zeros((ysize, xsize))

    # initial decision vector
    # first row: displacement vector from adjacent left tile
    # other rows: displacement vector from adjacent top tile
    x0 = np.nan_to_num(np.roll(p_ab_1, 1, axis=0))
    x0[0, ...] = np.nan_to_num(np.roll(p_ab_2[0, ...], 1, axis=0))

    prob = TileDisplacementLeastSquares(xcorr_options, frame_shapes,
                                        p_ab_1, score_1, p_ab_2, score_2)

    archi = pg.archipelago(n=0)

    pop = pg.population(prob=prob)
    pop.push_back(x0.flatten())
    for i in range(0, 8):
        # each algo gets a different seed
        algo = pg.algorithm(
            pg.simulated_annealing(Ts=10., Tf=1e-5, n_T_adj=10))
        if not i:
            algo.set_verbosity(500)
        archi.push_back(algo=algo, pop=pop)

    archi.evolve(2)
    archi.wait_check()

    fs = archi.get_champions_f()
    xs = archi.get_champions_x()
    champion = xs[0]
    current_min = fs[0][0]
    for x, f in zip(xs, fs):
        if f < current_min:
            current_min = f[0]
            champion = x

    print(current_min)

    # rebuild tile coordinates starting from decision vector x
    t = decision_vector_to_tile_coords(champion, shape).reshape(-1, 3)

    keys = ['Zs', 'Ys', 'Xs']
    ret_df = pd.DataFrame(data=t, index=idx, columns=keys)

    df[keys] = ret_df
    df[keys] -= df[keys].min()
    df[keys] = df[keys].apply(np.round).astype(int)

    df['Xs_end'] = df['Xs'] + df['xsize']
    df['Ys_end'] = df['Ys'] + df['ysize']
    df['Zs_end'] = df['Zs'] + df['nfrms']
