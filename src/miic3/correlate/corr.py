'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 26th March 2021 08:28:33 am
Last Modified: Friday, 26th March 2021 10:27:08 am
'''
import numpy as np
from mpi4py import MPI

from miic3.db.asdf_handler import get_available_stations, NoiseDB


class Correlator(object):
    def __init__(self, dir: str, network: list or str, station: list or str):
        # Init MPI
        self.comm = MPI.COMM_WORLD
        self.csize = self.comm.Get_size()

        # find the available noise dbs
        if isinstance(network, str):
            network = [network]
        if isinstance(station, str):
            station = [station]
        if len(station) != len(network):
            if len(station) == 1:
                station = station*len(network)
            elif len(network == 1):
                network = network*len(station)
            else:
                raise ValueError("""The lists containing network and station
                codes must either have the same length or one of them can have
                the length 1""")

        # Resolve patterns
        self.netlist = []
        self.statlist = []
        for net, stat in zip(network, station):
            if '*' in network+station or '?' in net+stat:
                net, stat = get_available_stations(dir, net, stat)
                self.netlist.extend(net)
                self.statlist.extend(stat)
            else:
                self.netlist.append(net)
                self.statlist.append(stat)

        # and, finally, find the h5 files associated to each of them
        self.noisedbs = [NoiseDB(dir, net, stat) for net, stat in zip(
            self.netlist, self.statlist)]
        # note that the indexing in the noisedb list and in the statlist
        # are identical, which could be useful to find them easily
        