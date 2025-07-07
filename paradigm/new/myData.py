from mne.io import read_raw_curry
from metabci.brainda.datasets.base import BaseDataset
from typing import Union, Optional, Dict
from pathlib import Path
from mne.io import Raw
from mne.channels import make_standard_montage
from metabci.brainda.utils.channels import upper_ch_names

file_name = 'E:/Desktop/acquisition/huang01.cdt'


class MyData(BaseDataset):
    _EVENTS = {
        'hybrid_bci': {
            "left_hand": (1, (0, 4)),
            "right_hand": (2, (0, 4))
        }
    }

    _CHANNELS = {
        'hybrid_bci': ['C3', 'CZ', 'C4', 'P3', 'P4', 'O1', 'O2']
    }

    def __init__(self, subjects, srate, paradigm, pattern='hybrid_bci'):
        self.pattern = pattern
        self.subjects = subjects
        self.srate = srate
        self.paradigm = paradigm

        super().__init__(
            dataset_code="Brainon",
            subjects=self.subjects,
            events=self._EVENTS[self.pattern],
            channels=self._CHANNELS[self.pattern],
            srate=self.srate,
            paradigm=self.paradigm
        )

    # 返回数据的路径
    def data_path(self,
                  subject: Union[str, int],
                  path: Optional[Union[str, Path]] = None,
                  force_update: bool = False,
                  update_path: Optional[bool] = None,
                  proxies: Optional[Dict[str, str]] = None,
                  verbose: Optional[Union[bool, str, int]] = None):
        if subject not in self.subjects:
            raise ValueError('Invalid subject {:d} given'.format(subject))

        if self.pattern == '':
            runs = list(range(1, 3))

        dests = []
        dests.append([file_name])
        return dests

    # 返回单个被试的数据
    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        print(dests)
        montage = make_standard_montage('standard_1005')
        montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                print(run_file)
                raw = read_raw_curry(run_file, preload=True)
                raw = upper_ch_names(raw)
                raw = raw.pick_types(eeg=True, stim=True,
                                     selection=self.channels)
                raw.set_montage(montage)

                runs['run_{:d}'.format(irun)] = raw
            sess['session_{:d}'.format(isess)] = runs
        return sess
