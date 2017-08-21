import yaml
import numpy as np
import pandas as pd


class XcorrFileMatrix(object):
    def __init__(self):
        self.compute_average = False
        self.xcorr_options = None

        self.stitch_data_frame = None

    def load_yaml(self, fname):
        with open(fname, 'r') as f:
            y = yaml.load(f)

        attrs = ['ascending_tiles_x', 'ascending_tiles_y']

        for attr in attrs:
            setattr(self, attr, y['xcorr-options'][attr])

        self.stitch_data_frame = pd.DataFrame(y['xcorr'])
        self.stitch_data_frame = self.stitch_data_frame.rename(
            columns={'aname': 'filename'})
        self.stitch_data_frame = self.stitch_data_frame.set_index('filename')

        self.xcorr_options = y['xcorr-options']

    def aggregate_results(self, compute_average):
        sdf = self.stitch_data_frame.reset_index()

        if compute_average:
            view = sdf.groupby(['filename', 'bname', 'axis']).agg(
                lambda x: np.average(x, weights=sdf.loc[x.index, 'score']) if
                sdf.loc[x.index, 'score'].sum() != 0 else np.average(x))
        else:
            view = sdf.groupby(['filename', 'bname', 'axis']).agg(
                lambda x: sdf.loc[np.argmax(sdf.loc[x.index, 'score']), x.name]
            )

        view = view.reset_index()
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
