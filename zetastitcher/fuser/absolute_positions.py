import numpy as np
import pandas as pd
import networkx as nx


def compute_shift_vectors(filematrix_dataframe, stitch_dataframe):
    sdf = stitch_dataframe
    fm_df = filematrix_dataframe

    sdf['px'] = 0
    sdf['py'] = 0
    sdf['pz'] = 0

    idx = sdf['axis'] == 1
    fm_df_idx = idx[idx].index
    sdf.loc[idx, 'px'] = sdf.loc[idx, 'dx']
    sdf.loc[idx, 'py'] = fm_df.loc[fm_df_idx, 'ysize'] - sdf.loc[idx, 'dy']
    sdf.loc[idx, 'pz'] = sdf.loc[idx, 'dz']

    idx = sdf['axis'] == 2
    fm_df_idx = idx[idx].index
    sdf.loc[idx, 'px'] = fm_df.loc[fm_df_idx, 'xsize'] - sdf.loc[idx, 'dy']
    sdf.loc[idx, 'py'] = -sdf.loc[idx, 'dx']
    sdf.loc[idx, 'pz'] = sdf.loc[idx, 'dz']


def compute_initial_guess(filematrix_dataframe, stitch_dataframe):
    fm_df = filematrix_dataframe
    sdf = stitch_dataframe.reset_index()

    G = nx.Graph()
    G.add_edges_from(((
        u, v, {'weight': w}) for u, v, w in
        np.c_[sdf['filename'], sdf['bname'], 1 - sdf['score']]))

    keys = ['Xs', 'Ys', 'Zs']
    for k in keys:
        fm_df[k] = 0

    sdf = sdf.reset_index().set_index('bname')

    cond = (fm_df['X'] == 0) & (fm_df['Y'] == 0) & (fm_df['Z'] == 0)
    top_left_corner = fm_df[cond].index[0]
    for edge in nx.bfs_edges(G, source=top_left_corner):
        btile = edge[1]
        parents = fm_df.loc[sdf.loc[[btile], 'filename']]
        temp_sdf = sdf.loc[[btile]].set_index('filename')
        temp = pd.DataFrame()
        temp['Xs'] = parents['Xs'] + temp_sdf['px']
        temp['Ys'] = parents['Ys'] + temp_sdf['py']
        temp['Zs'] = parents['Zs'] + temp_sdf['pz']
        temp['score'] = temp_sdf['score']
        fm_df.loc[btile, keys] = temp.apply(
            lambda x: np.average(x, weights=temp.loc[x.index, 'score']) if
            temp.loc[x.index, 'score'].sum() != 0 else np.average(x))

    fm_df[keys] -= fm_df[keys].min()
    fm_df[keys] = fm_df[keys].apply(np.round).astype(int)

    fm_df['Xs_end'] = fm_df['Xs'] + fm_df['xsize']
    fm_df['Ys_end'] = fm_df['Ys'] + fm_df['ysize']
    fm_df['Zs_end'] = fm_df['Zs'] + fm_df['nfrms']
