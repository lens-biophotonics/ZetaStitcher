import pandas as pd


class Overlaps(object):
    def __init__(self, filematrix):
        self.fm = filematrix

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

        fm_df = self.fm.data_frame

        overlap_n = pd.DataFrame()
        overlap_s = pd.DataFrame()
        overlap_e = pd.DataFrame()
        overlap_w = pd.DataFrame()
        overlap_nw = pd.DataFrame()
        overlap_ne = pd.DataFrame()
        overlap_sw = pd.DataFrame()
        overlap_se = pd.DataFrame()

        for j in range(0, self.fm.Ny):
            for i in range(0, self.fm.Nx):
                row = self.fm.data_frame.loc[self.fm.name_array[j, i]]

                # north
                if j > 0:
                    parent = self.fm.name_array[j - 1, i]
                    comp_diff(overlap_n, row, parent)
                else:
                    cols_to_zero(overlap_n, row.name)

                # south
                if j < self.fm.Ny - 1:
                    parent = self.fm.name_array[j + 1, i]
                    comp_diff(overlap_s, row, parent)
                else:
                    cols_to_zero(overlap_s, row.name)

                # east
                if i < self.fm.Nx - 1:
                    parent = self.fm.name_array[j, i + 1]
                    comp_diff(overlap_e, row, parent)
                else:
                    cols_to_zero(overlap_e, row.name)

                # west
                if i > 0:
                    parent = self.fm.name_array[j, i - 1]
                    comp_diff(overlap_w, row, parent)
                else:
                    cols_to_zero(overlap_w, row.name)

                # north-west
                if i > 0 and j > 0:
                    parent = self.fm.name_array[j - 1, i - 1]
                    comp_diff(overlap_nw, row, parent)
                else:
                    cols_to_zero(overlap_nw, row.name)

                # north-east
                if i < self.fm.Nx - 1 and j > 0:
                    parent = self.fm.name_array[j - 1, i + 1]
                    comp_diff(overlap_ne, row, parent)
                else:
                    cols_to_zero(overlap_ne, row.name)

                # south-west
                if i > 0 and j < self.fm.Ny - 1:
                    parent = self.fm.name_array[j + 1, i - 1]
                    comp_diff(overlap_sw, row, parent)
                else:
                    cols_to_zero(overlap_sw, row.name)

                # south-east
                if i < self.fm.Nx - 1 and j < self.fm.Ny - 1:
                    parent = self.fm.name_array[j + 1, i + 1]
                    comp_diff(overlap_se, row, parent)
                else:
                    cols_to_zero(overlap_se, row.name)

        self.overlap_n = overlap_n.astype(int)
        self.overlap_s = overlap_s.astype(int)
        self.overlap_e = overlap_e.astype(int)
        self.overlap_w = overlap_w.astype(int)
        self.overlap_nw = overlap_nw.astype(int)
        self.overlap_ne = overlap_ne.astype(int)
        self.overlap_sw = overlap_sw.astype(int)
        self.overlap_se = overlap_se.astype(int)

    def __getitem__(self, tile_name):
        df = pd.DataFrame()
        names = ['n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se']
        for n in names:
            name = 'overlap_' + n
            overlap = getattr(self, name)
            df[n] = overlap.loc[tile_name]
        return df.transpose()
