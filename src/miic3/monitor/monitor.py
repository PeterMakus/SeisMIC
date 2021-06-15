'''
:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 3rd June 2021 04:15:57 pm
Last Modified: Tuesday, 15th June 2021 04:45:34 pm
'''
import os
from typing import Tuple
import yaml

import numpy as np
from obspy import UTCDateTime


class Monitor(object):
    def __init__(self, options: dict or str):
        if isinstance(options, str):
            with open(options) as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        self.options = options
        self.starttimes, self.endtimes = self._starttimes_list()
        self.outdir = os.path.join(
            options['proj_dir'], options['dv']['subdir'])
        self.indir = os.path.join(
            options['proj_dir'], options['co']['subdir']
        )

    def _starttimes_list(self) -> Tuple[np.ndarray, np.ndarray]:
        start = UTCDateTime(self.options['dv']['start_date']).timestamp
        end = UTCDateTime(self.options['dv']['end_date']).timestamp
        inc = self.options['dv']['date_inc']
        starttimes = np.arange(start, end, inc)
        endtimes = starttimes + self.options['dv']['win_len']
        return starttimes, endtimes