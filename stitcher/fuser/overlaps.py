import pandas as pd


class Overlaps(object):
    def __init__(self, filematrix_dataframe, stitch_dataframe):
        self.fm_df = filematrix_dataframe
        self.sdf = stitch_dataframe

        self.overlap_n = None
        self.overlap_s = None
        self.overlap_e = None
        self.overlap_w = None
        self.overlap_nw = None
        self.overlap_ne = None
        self.overlap_sw = None
        self.overlap_se = None

        self._compute_overlaps()

    def _compute_overlaps(self):
        def comp_diff(dest, row, other_name):
            other = fm_df.loc[other_name]
            temp = pd.Series()
            temp['Z_from'] = max(other['Zs'], row['Zs']) - row['Zs']
            temp['Z_to'] = min(other['Zs_end'], row['Zs_end']) - row['Zs']

            temp['Y_from'] = max(other['Ys'], row['Ys']) - row['Ys']
            temp['Y_to'] = min(other['Ys_end'], row['Ys_end']) - row['Ys']

            temp['X_from'] = max(other['Xs'], row['Xs']) - row['Xs']
            temp['X_to'] = min(other['Xs_end'], row['Xs_end']) - row['Xs']

            cols_to_zero(dest, row.name)

            if temp['Z_from'] > temp['Z_to'] or temp['Y_from'] > temp['Y_to'] \
                    or temp['X_from'] > temp['X_to']:
                pass
            else:
                dest.loc[row.name] = temp

        def cols_to_zero(dest, row_name):
            cols = ['Z_from', 'Z_to', 'Y_from', 'Y_to', 'X_from', 'X_to']
            for c in cols:
                dest.loc[row_name, c] = 0

        fm_df = self.fm_df
        sdf = self.sdf

        overlap_n = pd.DataFrame()
        overlap_s = pd.DataFrame()
        overlap_e = pd.DataFrame()
        overlap_w = pd.DataFrame()
        overlap_nw = pd.DataFrame()
        overlap_ne = pd.DataFrame()
        overlap_sw = pd.DataFrame()
        overlap_se = pd.DataFrame()

        tmp_df_b = sdf.reset_index().set_index(['bname', 'axis'])
        tmp_df_a = sdf.reset_index().set_index(['filename', 'axis'])

        for index, row in fm_df.iterrows():
            # north
            try:
                parent = tmp_df_b.loc[(row.name, 1), 'filename']
                comp_diff(overlap_n, row, parent)
            except KeyError:
                cols_to_zero(overlap_n, row.name)

            # south
            try:
                parent = tmp_df_a.loc[(row.name, 1), 'bname']
                comp_diff(overlap_s, row, parent)
            except KeyError:
                cols_to_zero(overlap_s, row.name)

            # east
            try:
                parent = tmp_df_a.loc[(row.name, 2), 'bname']
                comp_diff(overlap_e, row, parent)
            except KeyError:
                cols_to_zero(overlap_e, row.name)

            # west
            try:
                parent = tmp_df_b.loc[(row.name, 2), 'filename']
                comp_diff(overlap_w, row, parent)
            except KeyError:
                cols_to_zero(overlap_w, row.name)

            # north-west
            try:
                other_name = tmp_df_b.loc[
                    (row.name, 2), 'filename']  # one step W
                other_name = tmp_df_b.loc[
                    (other_name, 1), 'filename']  # one step N
                comp_diff(overlap_nw, row, other_name)
            except KeyError:
                cols_to_zero(overlap_nw, row.name)

            # north-east
            try:
                other_name = tmp_df_a.loc[(row.name, 2), 'bname']  # one step E
                other_name = tmp_df_b.loc[
                    (other_name, 1), 'filename']  # one step N
                comp_diff(overlap_ne, row, other_name)
            except KeyError:
                cols_to_zero(overlap_ne, row.name)

            # south-west
            try:
                other_name = tmp_df_b.loc[
                    (row.name, 2), 'filename']  # one step W
                other_name = tmp_df_a.loc[
                    (other_name, 1), 'bname']  # one step S
                comp_diff(overlap_sw, row, other_name)
            except KeyError:
                cols_to_zero(overlap_sw, row.name)

            # south-east
            try:
                other_name = tmp_df_a.loc[(row.name, 2), 'bname']  # one step E
                other_name = tmp_df_a.loc[
                    (other_name, 1), 'bname']  # one step S
                comp_diff(overlap_se, row, other_name)
            except KeyError:
                cols_to_zero(overlap_se, row.name)

        self.overlap_n = overlap_n.astype(int)
        self.overlap_s = overlap_s.astype(int)
        self.overlap_e = overlap_e.astype(int)
        self.overlap_w = overlap_w.astype(int)
        self.overlap_nw = overlap_nw.astype(int)
        self.overlap_ne = overlap_ne.astype(int)
        self.overlap_sw = overlap_sw.astype(int)
        self.overlap_se = overlap_se.astype(int)

    def overlaps(self, tile_name):
        df = pd.DataFrame()
        names = ['n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se']
        for n in names:
            name = 'overlap_' + n
            overlap = getattr(self, name)
            df[n] = overlap.loc[tile_name]
        return df.transpose()
