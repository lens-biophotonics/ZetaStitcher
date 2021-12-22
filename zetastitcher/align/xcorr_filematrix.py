import yaml
import pandas as pd


class XcorrFileMatrix:
    def __init__(self):
        self.xcorr_options = None

        self.stitch_data_frame = None

        self.ascending_tiles_x = None
        self.ascending_tiles_y = None

    @classmethod
    def from_data(cls, xcorr_options, xcorr_df):
        inst = cls()

        attrs = ['ascending_tiles_x', 'ascending_tiles_y']

        for attr in attrs:
            setattr(inst, attr, xcorr_options[attr])

        stitch_data_frame = xcorr_df
        stitch_data_frame = stitch_data_frame.rename(columns={'aname': 'filename'})
        stitch_data_frame = stitch_data_frame.set_index('filename')

        inst.stitch_data_frame = stitch_data_frame
        inst.xcorr_options = xcorr_options

        return inst

    @classmethod
    def from_yaml(cls, fname):
        with open(fname, 'r') as f:
            y = yaml.safe_load(f)

        return XcorrFileMatrix.from_data(y['xcorr-options'], pd.DataFrame(y['xcorr']))

    def aggregate_results(self):
        sdf = self.stitch_data_frame.reset_index()

        view = sdf.loc[sdf.groupby(['filename', 'bname', 'axis']).score.idxmax()]
        overlap_dict = {1: self.xcorr_options['overlap_v'],
                        2: self.xcorr_options['overlap_h']}

        view.dz -= self.xcorr_options['max_dz']
        for a in [1, 2]:
            indexes = (view['axis'] == a)
            view.loc[indexes, 'dy'] = (overlap_dict[a]
                                       + self.xcorr_options['max_dy']
                                       - view.loc[indexes, 'dy'])
        view.dx -= self.xcorr_options['max_dx']

        self.stitch_data_frame = view.set_index('filename')
