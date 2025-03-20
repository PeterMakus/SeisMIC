'''
Helper functions for preprocessing modules.

:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Johanna Lehr (jlehr@gfz-potsdam.de)

Created: Wednesday, 2025-03-20 15:15:21
Last Modified: 2025-03-20 15:15:28
'''

import numpy as np


def get_joint_norm(B, args: dict) -> None:
    """
    Average over n rows of a 2D numpy array."
    """

    if 'joint_norm' in list(args.keys()):
        if args['joint_norm'] is True:
            args['joint_norm'] = 3

        if args['joint_norm'] in [2, 3]:
            k = args['joint_norm']
            assert B.shape[0] % k == 0, "For joint normalization with %d the "\
                "number of traces needs to the multiple of it, but is %d" % (
                    k, B.shape[0])
            B[:, :] = np.repeat(np.mean(B.reshape(-1, k, B.shape[1]), axis=1),
                                k, axis=0)
        elif args['joint_norm'] == 1 or args['joint_norm'] is False:
            pass
        else:
            raise ValueError(
                "joint_norm must be int of 1, 2, 3 or False/True(==3 )")
