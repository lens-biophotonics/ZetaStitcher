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
