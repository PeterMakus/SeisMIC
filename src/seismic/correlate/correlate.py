"""
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 29th March 2021 07:58:18 am
Last Modified: Wednesday, 25th Febuary 2025 01:48:00 pm (J. Lehr)
"""

from typing import Iterator, List, Tuple, Optional
from warnings import warn
import os
import logging
import json
import warnings
import yaml
import glob
import fnmatch

from mpi4py import MPI
import numpy as np
from obspy import Stream, UTCDateTime, Inventory, Trace
from tqdm import tqdm

from seismic.correlate.stream import CorrTrace, CorrStream
from seismic.correlate import preprocessing_td as pptd
from seismic.correlate import preprocessing_fd as ppfd
from seismic.correlate import preprocessing_stream as ppst
from seismic.db.corr_hdf5 import CorrelationDataBase, h5_FMTSTR
from seismic.trace_data.waveform import Store_Client, Local_Store_Client
from seismic.utils.fetch_func_from_str import func_from_str
from seismic.utils import miic_utils as mu
from .. import logfactory

LOGDIR = "log"
DEFAULT_LOGPARAMS = dict(
    loglevel="WARNING", logdir=LOGDIR, filename_fmt=logfactory.FILENAME_FMT
)
TSTR_FMT = "%Y-%m-%d-%H:%M:%S"

parentlogger = logfactory.create_logger()
module_logger = logging.getLogger(parentlogger.name + ".waveform")

# This is probably the most ugly way to check if the function accepts the
# joint_norm argument. But it works.
functions_accepting_joint_norm = [
    ".".join([m.__name__, f.__name__])
    for m in [pptd, ppfd]
    for f in m.functions_accepting_jointnorm
]
fsdn_component_ids = set(list("ENZ123ABCUVW"))


class Correlator(logfactory.LoggingMPIBaseClass):
    """
    Object to manage the actual Correlation (i.e., Green's function retrieval)
    for the database.
    """

    def __init__(self, options: dict | str, store_client: Store_Client = None):
        """
        Initiates the Correlator object. When executing
        :func:`~seismic.correlate.correlate.Correlator.pxcorr()`, it will
        actually compute the correlations and save them in an hdf5 file that
        can be handled using
        :class:`~seismic.db.corr_hdf5.CorrelationDataBase`.

        :param options: Dictionary containing all options for the correlation.
            Can also be a path to a yaml file containing all the keys required
            in options.
        :type options: dict or str
        :param store_client: Object that handles the data retrieval. If None,
            a :class:`~seismic.trace_data.waveform.Local_Store_Client` will be
            initiated. In this case all data has to be available locally.
        """
        if isinstance(options, str):
            with open(file=options) as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        elif isinstance(options, Store_Client):
            raise DeprecationWarning(
                "Order of arguments in Correlator has changed. "
                + "The Store_Client has to be passed as the second argument. "
                + "Can be None to init Local_Store_Client from options."
            )

        # init MPI, logging
        super().__init__()

        # directories
        self.proj_dir = options["proj_dir"]
        self.corr_dir = os.path.join(self.proj_dir, options["co"]["subdir"])
        try:
            self.save_comps_separately = options["save_comps_separately"]
        except KeyError:
            self.save_comps_separately = False

        logdir = os.path.join(self.proj_dir, options["log_subdir"])
        self.set_logger(options["log_level"], logdir)
        warnlog = logging.getLogger("py.warnings")
        warnlog.addHandler(
            [
                h
                for h in self.logger.parent.handlers
                if isinstance(h, logging.FileHandler)
            ][0]
        )
        self.logger.debug(
            "Warn logger has handler: {}".format(warnlog.hasHandlers())
        )

        if self.rank == 0:
            os.makedirs(self.corr_dir, exist_ok=True)

        # Write the options dictionary to the log file
        if self.rank == 0:
            opt_dump = mu.utcdatetime2str(options)
            # opt_dump = deepcopy(options)
            # # json cannot write the UTCDateTime objects that might be in here
            # for step in opt_dump['co']['preProcessing']:
            #     if 'stream_mask_at_utc' in step['function']:
            #         startsstr = [
            #             t.format_fissures() for t in step['args']['starts']]
            #         step['args']['starts'] = startsstr
            #         if 'ends' in step['args']:
            #             endsstr = [t.format_fissures()
            #                 for t in step['args']['ends']]
            #             step['args']['ends'] = endsstr
            tstr = UTCDateTime.now().strftime("%Y-%m-%d-%H:%M")
            with open(os.path.join(logdir, "params%s.txt" % tstr), "w") as file:
                file.write(json.dumps(opt_dump, indent=1))

        self.options = options["co"]
        self._set_joint_norm_arg()

        # requested combis?
        if "xcombinations" in self.options:
            self.rcombis = self.options["xcombinations"]
            if self.rcombis == "None":
                # cumbersome, but someone used it wrong so let's hardcode
                self.rcombis = None
        else:
            self.rcombis = None

        # Store_Client
        if store_client is None:
            store_client = Local_Store_Client(
                options,
                logparams=dict(
                    loglevel=options["log_level"],
                    logdir=logdir,
                    filename_fmt=logfactory.FILENAME_FMT,
                ),
            )
        self.store_client = store_client

        self._find_available_data(options)

        self.sampling_rate = self.options["sampling_rate"]
        if "allow_different_params" in self.options:
            self._allow_different_params = self.options[
                "allow_different_params"
            ]
        else:
            self._allow_different_params = False

        self._check_joint_norm()

    def _find_available_data(self, options):
        """
        Find the available data for the requested stations and networks.

        Will also filter the stations by the requested cross-correlations.
        It is mostly a wrapper to expand the wildcards in the station list.

        :param options: The options dictionary
        :type options: dict
        """
        assert hasattr(self, "store_client"), (
            "Store_Client has to be set before calling _find_available_data."
        )
        network = options["net"]["network"]
        station = options["net"]["station"]
        component = options["net"]["component"]
        # location = options['net']['location']

        if isinstance(station, list) and len(station) == 1:
            station = station[0]
        if isinstance(network, list) and len(network) == 1:
            network = network[0]
        # if isinstance(location, list) and len(network) == 1:
        #     network = network[0]

        if network == "*" and isinstance(station, str) and "*" not in station:
            raise ValueError(
                "Stations has to be either: \n"
                + "1. A list of the same length as the list of networks.\n"
                + "2. '*' That is, a wildcard (string).\n"
                + "3. A list and network is a string describing one "
                + "station code."
            )
        elif isinstance(station, str) and isinstance(network, str):
            station = [[network, station]]
        elif station == "*" and isinstance(network, list):
            # This is most likely not thread-safe
            if self.rank == 0:
                station = []
                for net in network:
                    station.extend(
                        self.store_client.get_available_stations(net)
                    )
            else:
                station = None
            station = self.comm.bcast(station, root=0)
        elif isinstance(network, list) and isinstance(station, list):
            if len(network) != len(station):
                raise ValueError(
                    "Stations has to be either: \n"
                    + "1. A list of the same length as the list of networks.\n"
                    + "2. '*' That is, a wildcard (string).\n"
                    + "3. A list and network is a string describing one "
                    + "station code."
                )
            station = list([n, s] for n, s in zip(network, station))
        elif isinstance(station, list) and isinstance(network, str):
            for ii, stat in enumerate(station):
                station[ii] = [network, stat]
        else:
            raise ValueError(
                "Stations has to be either: \n"
                + "1. A list of the same length as the list of networks.\n"
                + "2. '*' That is, a wildcard (string).\n"
                + "3. A list and network is a string describing one "
                + "station code."
            )
        if self.rank == 0:
            self.avail_raw_data = []
            for net, stat in station:
                self.avail_raw_data.extend(
                    self.store_client._translate_wildcards(
                        net, stat, component, location="*"
                    )
                )
            # make sure this only contains unique combinations
            # with several cores it added entries several times, don't know
            # why?
            # In contrast to self.station, self.avail_raw_data also contains
            # information about the available channels, so they can be
            # read and processed on different cores
            self.avail_raw_data = np.unique(
                self.avail_raw_data, axis=0
            ).tolist()
        else:
            self.avail_raw_data = None
        self.avail_raw_data = self.comm.bcast(self.avail_raw_data, root=0)
        self.station = np.unique(
            np.array([[d[0], d[1]] for d in self.avail_raw_data]), axis=0
        ).tolist()
        # if only certain combis are requested, remove stations not within
        # these
        self._filter_by_rcombis()
        self.logger.debug(
            "Fetching data from the following stations:\n%a"
            % [f"{n}.{s}" for n, s in self.station]
        )

    def _set_joint_norm_arg(self):
        """
        Set the joint_norm argument in processing functions.
        """
        if self.rank == 0:
            assert hasattr(self, "options"), (
                "Options have to be set before calling"
                + " `_set_joint_norm_arg`."
            )
            self.logger.debug("Setting joint_norm argument.")

            try:
                joint_norm = self.options["joint_norm"]
            except KeyError:
                warn(
                    "Keywork joint_norm not found in options['co']. "
                    + "Assuming False.",
                    UserWarning,
                )
                joint_norm = False
                self.options["joint_norm"] = joint_norm

            # Check if user input is valid
            err_msg = "joint_norm has to be either True or False."
            if not isinstance(joint_norm, (bool)):
                raise ValueError(err_msg)

            for proc, funcs in self.options["corr_args"].items():
                if proc not in ["TDpreProcessing", "FDpreProcessing"]:
                    continue
                for func in funcs:
                    if func["function"] in functions_accepting_joint_norm:
                        func["args"]["joint_norm"] = joint_norm

        self.options = self.comm.bcast(self.options, root=0)
        self.logger.info(
            "joint_norm set to %s" % str(self.options["joint_norm"])
        )

    def _check_joint_norm(self):
        """
        Run sanity checks if `joint_norm` is True.

        The following checks are performed:
        - at least 3 unique components are available
        - only components from the set `fsdn_component_ids` are available
        - each station must in principle have 3 components

        Checks are performed on `self.avail_raw_data`. Thus information
        is based on structure of sds and not on the actual data. If single
        days or time gaps are missing, this will not be detected here, but
        is handled later in `pxcorr` and its submethods.
        """
        assert hasattr(self, "options"), (
            "Options have to be set before calling" + " `_check_joint_norm`."
        )
        assert hasattr(self, "avail_raw_data"), (
            "avail_raw_data has to be set before calling"
            + " `_check_joint_norm`."
        )

        # Sanity checks if joint_norm makes sense to use given codes.
        # We do not check if there are actually 3 traces always available
        # for each station.
        if not self.options["joint_norm"]:
            return
        self.logger.debug("Checking if data is fit for joint_norm.")
        if self.rank == 0:
            unique_components = set(
                [item[-1][-1] for item in self.avail_raw_data]
            )
            if len(unique_components) < 3:
                raise ValueError(
                    "Expecting at least 3 unique components if "
                    + "joint_norm is True. Only the following "
                    + "components are available: %s" % str(unique_components)
                )
            elif not unique_components.issubset(fsdn_component_ids):
                raise ValueError(
                    "Expecting only components from the set %s. "
                    % (str(fsdn_component_ids))
                    + "The following components are available: %s"
                    % (str(unique_components))
                )

            # Check if the first two subitems in the items of avail_raw_data
            # are the same for every three items
            i = 0
            popped = []
            while i < len(self.avail_raw_data) - 2:
                if (
                    self.avail_raw_data[i][:-1]
                    == self.avail_raw_data[i + 1][:-1]
                    == self.avail_raw_data[i + 2][:-1]
                ):
                    i = i + 3
                else:
                    popped.append(self.avail_raw_data.pop(i))

            # Pop remaining items if less than 3
            while len(self.avail_raw_data) > i:
                popped.append(self.avail_raw_data.pop(-1))

            if popped:
                raise ValueError(
                    "Expecting 3-component stations if `joint_norm` is True. "
                    + "The following stations have less than 3 components: %s"
                    % str(popped)
                )

            self.station = np.unique(
                np.array([[d[0], d[1]] for d in self.avail_raw_data]), axis=0
            ).tolist()

        self.avail_raw_data = self.comm.bcast(self.avail_raw_data, root=0)
        self.station = self.comm.bcast(self.station, root=0)

    def _filter_by_rcombis(self):
        """
        Removes stations from the list of available stations that are not
        requested in the cross-combinations.
        """
        if (
            self.rcombis is None
            or self.options["combination_method"] != "betweenStations"
        ):
            return
        self.station = [
            [n, s]
            for n, s in self.station
            if fnmatch.filter(self.rcombis, f"{n}-*.{s}-*")
            or fnmatch.filter(self.rcombis, f"*-{n}.*-{s}")
        ]
        # same check for avail_raw_data
        self.avail_raw_data = [
            [n, s, loc, c]
            for n, s, loc, c in self.avail_raw_data
            if fnmatch.filter(self.rcombis, f"{n}-*.{s}-*")
            or fnmatch.filter(self.rcombis, f"*-{n}.*-{s}")
        ]

    def find_interstat_dist(self, dis: float):
        """
        Find stations in database with interstation distance smaller than
        dis.

        If no station inventories are available, they will be downloaded.

        :param dis: Find all Stations with distance less than `dis` [in m]
        :type dis: float

        .. note:: only the subset of the in ``params.yaml`` defined
            networks and stations will be queried.
        """
        if not self.options["combination_method"] == "betweenStations":
            raise ValueError(
                "This function is only available if combination method "
                + 'is set to "betweenStations".'
            )
        # Update the store clients invetory
        self.store_client.read_inventory()
        # list of requested combinations
        if self.rcombis is None:
            self.rcombis = []
            for ii, (n0, s0) in enumerate(self.station):
                inv0 = self.store_client.select_inventory_or_load_remote(n0, s0)
                for n1, s1 in self.station[ii:]:
                    inv1 = self.store_client.select_inventory_or_load_remote(
                        n1, s1
                    )
                    if mu.filter_stat_dist(inv0, inv1, dis):
                        self.rcombis.append("%s-%s.%s-%s" % (n0, n1, s0, s1))
        else:
            raise ValueError(
                "Either filter for specific cross correlations or a maximum "
                + "distance."
            )

    def find_existing_times(self, tag: str, channel: str = "*") -> dict:
        """
        Returns the already existing starttimes in form of a dictionary (see
        below)

        :param tag: The tag that the waveforms are saved under
        :type tag: str
        :param channel: Channel Combination Code (e.g., CH0-CH1),
            wildcards accepted. Defaults to '*'
        :type channel: str, optional
        :return: Dictionary that is structured as in the example below
        :rtype: dict

        Examples
        --------
        >>> out_dict = my_correlator.find_existing_times('mytag', 'BHZ-BHH')
        >>> print(out_dict)
        {'NET0.STAT0': {
            'NET1.STAT1': {'BHZ-BHH': [%list of starttimes] ,
            'NET2.STAT2': {'BHZ-BHH':[%list of starttimes]}}}
        """
        netlist, statlist = list(zip(*self.station))
        netcombs, statcombs = compute_network_station_combinations(
            netlist,
            statlist,
            method=self.options["combination_method"],
            combis=self.rcombis,
        )
        ex_dict = {}
        for nc, sc in zip(netcombs, statcombs):
            outfs = h5_FMTSTR.format(
                dir=self.corr_dir,
                network=nc,
                station=sc,
                location="*",
                channel="*",
            )
            if not len(glob.glob(outfs)):
                continue
            d = {}
            for outf in glob.glob(outfs):
                # retrieve location codes
                l0, l1 = os.path.basename(outf).split(".")[2].split("-")
                with CorrelationDataBase(
                    outf,
                    corr_options=self.options,
                    mode="r",
                    _force=self._allow_different_params,
                ) as cdb:
                    d.setdefault("%s-%s" % (l0, l1), {})
                    d[f"{l0}-{l1}"].update(
                        cdb.get_available_starttimes(
                            nc, sc, tag, f"{l0}-{l1}", channel
                        )
                    )
            s0, s1 = sc.split("-")
            n0, n1 = nc.split("-")
            # obtain location codes

            ex_dict.setdefault("%s.%s" % (n0, s0), {})
            ex_dict["%s.%s" % (n0, s0)]["%s.%s" % (n1, s1)] = d
        return ex_dict

    def pxcorr(self):
        """
        Start the correlation with the parameters that were defined when
        initiating the object.
        """
        cst = CorrStream()
        if self.rank == 0:
            self.logger.debug("Reading Inventory files.")
        # Fetch station coordinates
        if self.rank == 0:
            try:
                inv = self.store_client.read_inventory()
            except Exception as e:
                if self.options["remove_response"]:
                    raise FileNotFoundError(
                        "No response information could be found."
                        + "If you set remove_response to True, you will need"
                        + "a station inventory."
                    )
                logging.warning(e)
                warnings.warn(
                    "No Station Inventory found. Proceeding without.",
                    UserWarning,
                )
                inv = None
        else:
            inv = None
        inv = self.comm.bcast(inv, root=0)

        for st, write_flag in self._generate_data():
            cst.extend(self._pxcorr_inner(st, inv))
            if write_flag:
                self.logger.info("Writing %d correlations." % cst.count())
                # Here, we can recombine the correlations for the read_len
                # size (i.e., stack)
                # Write correlations to HDF5
                if cst.count():
                    self._write(cst)
                    cst.clear()

        # write the remaining data
        if cst.count():
            self.logger.info("Writing %d remaining correlations." % cst.count())
            self._write(cst)
            cst.clear()

    def _pxcorr_inner(self, st: Stream, inv: Inventory) -> CorrStream:
        """
        Inner loop of pxcorr. Don't call this function!
        """

        # We start out by moving the stream into a matrix

        # put all the data into a single stream
        starttime = []
        npts = []
        for tr in st:
            starttime.append(tr.stats["starttime"])
            npts.append(tr.stats["npts"])
        npts = np.max(np.array(npts))

        self.logger.info(
            "Processing & correlating %d traces" % len(st)
            + " in time window %s-%s"
            % (
                min(starttime).strftime(TSTR_FMT),
                (
                    min(starttime) + self.options["subdivision"]["corr_len"]
                ).strftime(TSTR_FMT),
            )
        )
        self.logger.debug("Converting Stream to Matrix")
        A, st = st_to_np_array(st, npts)
        self.options.update(
            {"starttime": starttime, "sampling_rate": self.sampling_rate}
        )
        self.logger.debug("Processing matrix...")
        A, startlags = self._pxcorr_matrix(A)
        self.logger.debug("Converting Matrix to CorrStream.")
        # put trace into a stream
        cst = CorrStream()
        if A is None:
            # No new data
            return cst
        if self.rank == 0:
            for ii, (startlag, comb) in enumerate(
                zip(startlags, self.options["combinations"])
            ):
                endlag = (
                    startlag + len(A[ii, :]) / self.options["sampling_rate"]
                )
                cst.append(
                    CorrTrace(
                        A[ii],
                        header1=st[comb[0]].stats,
                        header2=st[comb[1]].stats,
                        inv=inv,
                        start_lag=startlag,
                        end_lag=endlag,
                    )
                )
        else:
            cst = None
        cst = self.comm.bcast(cst, root=0)
        self.logger.info(
            "Core %d finished processing %d correlations."
            % (self.rank, cst.count())
        )
        return cst

    def _write(self, cst):
        """
        Write correlation stream to files.

        :param cst: CorrStream containing the correlations
        :type cst: :class:`~seismic.correlate.stream.CorrStream`
        """
        if not cst.count():
            self.logger.debug("No new data written.")
            return

        # Make sure that each core writes to a different file
        filelist = list(
            set(
                h5_FMTSTR.format(
                    dir=self.corr_dir,
                    network=tr.stats.network,
                    station=tr.stats.station,
                    location=tr.stats.location,
                    channel=tr.stats.channel,
                )
                for tr in cst
            )
        )
        # Better if the same cores keep writing to the same files
        filelist.sort()
        # Decide which process writes to which station
        pmap = np.arange(len(filelist)) * self.psize / len(filelist)
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        self.logger.info("Core %d writing to %d files." % (self.rank, len(ind)))

        for outf in np.array(filelist)[ind]:
            net, stat, loc, cha = os.path.basename(outf).split(".")[0:4]
            cstselect = cst.select(
                network=net, station=stat, location=loc, channel=cha
            )
            if self.options["subdivision"]["recombine_subdivision"]:
                stack = cstselect.stack()
                stacktag = "stack_%s" % str(self.options["read_len"])
            else:
                stack = None
            with CorrelationDataBase(
                outf,
                corr_options=self.options,
                _force=self._allow_different_params,
            ) as cdb:
                if cstselect.count():
                    self.logger.debug(
                        "Writing %d correlations to %s",
                        cstselect.count(),
                        outf,
                    )
                    cdb.add_correlation(cstselect, "subdivision")
                if stack is not None:
                    self.logger.debug(
                        "Writing %d stacked correlations to %s",
                        stack.count(),
                        outf,
                    )
                    cdb.add_correlation(stack, stacktag)

    def _generate_data(self) -> Iterator[Tuple[Stream, bool]]:
        """
        Returns an Iterator that loops over each start and end time with the
        requested window length.


        :yield: An obspy stream containing the time window x for all stations
        that were active during this time.
        :rtype: Iterator[Stream]
        """
        if self.rank == 0:
            # find already available times
            self.ex_dict = self.find_existing_times("subdivision")
            self.logger.info("Already existing data: %s" % str(self.ex_dict))
        else:
            self.ex_dict = None

        self.ex_dict = self.comm.bcast(self.ex_dict, root=0)

        if not self.ex_dict and self.options["preprocess_subdiv"]:
            self.options["preprocess_subdiv"] = False
            if self.rank == 0:
                self.logger.warning(
                    "No existing data found.\nAutomatically setting "
                    "preprocess_subdiv to False to optimise performance."
                )

        # the time window that the loop will go over
        t0 = UTCDateTime(self.options["read_start"]).timestamp
        t1 = UTCDateTime(self.options["read_end"]).timestamp
        loop_window = np.arange(t0, t1, self.options["read_inc"])

        # Taper ends for the deconvolution and filtering
        tl = 20

        # Decide which process reads data from which station
        # Better than just letting one core read as this avoids having to
        # send very big chunks of data using MPI (MPI communication does
        # not support more than 2GB/comm operation)
        pmap = (
            np.arange(len(self.avail_raw_data))
            * self.psize
            / len(self.avail_raw_data)
        )
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(len(self.avail_raw_data))[ind]
        self.logger.debug(
            "Core %d reading %s"
            % (self.rank, str(np.array(self.avail_raw_data)[ind]))
        )
        # Loop over read increments
        for t in tqdm(loop_window):
            write_flag = True  # Write length is same as read length
            startt = UTCDateTime(t)
            endt = startt + self.options["read_len"]
            self.logger.info(
                "Core %d reading data for time %s - %s",
                self.rank,
                startt.strftime(TSTR_FMT),
                endt.strftime(TSTR_FMT),
            )

            st = Stream()

            # loop over queried stations
            for net, stat, loc, cha in np.array(self.avail_raw_data)[ind]:
                self.logger.debug(
                    "Core %d reading %s, %s, %s" % (self.rank, net, stat, cha)
                )
                # Load data
                stext = self.store_client._load_local(
                    net, stat, loc, cha, startt, endt, True, False
                )
                try:
                    mu.get_valid_traces(stext)
                except TypeError:
                    # stext is None
                    continue
                if stext is None or not len(stext):
                    # No data for this station to read
                    continue
                st = st.extend(stext)

            self.logger.info("Core %d loaded %d traces." % (self.rank, len(st)))
            self.logger.debug("IDs: %s" % str([tr.id for tr in st]))

            # The stream has to be tapered ebfore decimating!
            # (Filter operation), added a taper here on 2025/02/21
            st = ppst.detrend_st(st, "linear")
            st = mu.cos_taper_st(st, tl, False, False)
            # Stream based preprocessing
            # Downsampling
            # 04/04/2023 Downsample before preprocessing for performance
            # Check sampling frequency
            sampling_rate = self.options["sampling_rate"]
            # AA-Filter is done in this function as well
            self.logger.debug("Start downsampling ...")
            try:
                st = mu.resample_or_decimate(st, sampling_rate)
                self.logger.debug("Finished downsampling.")
            except ValueError as e:
                self.logger.error(
                    "Downsampling failed for "
                    f"{st[0].stats.network}.{st[0].stats.station} and time"
                    f" {t}.\nThe Original Error Message was {e}."
                )
                continue
            # The actual data in the mseeds was changed from int to float64
            # now,
            # Save some space by changing it back to 32 bit (most of the
            # digitizers work at 24 bit anyways)
            mu.stream_require_dtype(st, np.float32)

            if not self.options["preprocess_subdiv"]:
                try:
                    self.logger.debug("Preprocessing read_len stream...")
                    st = preprocess_stream(
                        st, self.store_client, startt, endt, tl, **self.options
                    )
                    self.logger.debug("Finished preprocessing read_len stream.")
                except Exception as e:
                    self.logger.error(
                        "Stream preprocessing failed for "
                        f"{st[0].stats.network}.{st[0].stats.station} and time"
                        f" {t}.\nThe Original Error Message was {e}."
                    )
                    continue

            # Slice the stream in correlation length
            # -> Loop over correlation increments
            for ii, win in enumerate(generate_corr_inc(st, **self.options)):
                winstart = startt + ii * self.options["subdivision"]["corr_inc"]
                winend = winstart + self.options["subdivision"]["corr_len"]

                self.logger.info(
                    "Core %d working on time window %s - %s",
                    self.rank,
                    winstart.strftime(TSTR_FMT),
                    winend.strftime(TSTR_FMT),
                )

                # Gather time windows from all stations to all cores
                winl = self.comm.allgather(win)
                win = Stream()
                for winp in winl:
                    win.extend(winp)
                win = win.sort()

                # Get correlation combinations
                if self.rank == 0:
                    self.logger.info("Calculating combinations...")
                    self.options["combinations"] = calc_cross_combis(
                        win,
                        self.ex_dict,
                        self.options["combination_method"],
                        rcombis=self.rcombis,
                    )
                else:
                    self.logger.info(
                        "Core %d waiting for combinations..." % self.rank
                    )
                    self.options["combinations"] = None
                self.options["combinations"] = self.comm.bcast(
                    self.options["combinations"], root=0
                )

                if not len(self.options["combinations"]):
                    # no new combinations for this time period
                    self.logger.info(
                        f"No new data for times {winstart}-{winend}"
                    )
                    continue
                # Remove traces that won't be accessed at all
                win_indices = np.arange(len(win))
                combindices = np.unique(np.hstack(self.options["combinations"]))
                popindices = np.flip(np.setdiff1d(win_indices, combindices))
                self.logger.info(
                    "Core %d found %d traces not in combinations. Removing..."
                    % (self.rank, len(popindices))
                )
                for popi in popindices:
                    self.logger.debug(
                        "Core %d removing trace %s" % (self.rank, win[popi].id)
                    )
                    del win[popi]
                if len(popindices):
                    # now we have to recompute the combinations
                    if self.rank == 0:
                        self.logger.info("Recalculating combinations...")
                        self.options["combinations"] = calc_cross_combis(
                            win,
                            self.ex_dict,
                            self.options["combination_method"],
                            rcombis=self.rcombis,
                        )
                    else:
                        self.logger.info(
                            "Core %d waiting for combinations..." % self.rank
                        )
                        self.options["combinations"] = None
                    self.options["combinations"] = self.comm.bcast(
                        self.options["combinations"], root=0
                    )

                    if not len(self.options["combinations"]):
                        # no new combinations for this time period
                        self.logger.info(
                            f"No new data for times {winstart}-{winend}"
                        )
                        continue
                # Stream based preprocessing
                if self.options["preprocess_subdiv"]:
                    self.logger.debug("Core %d preprocessing corr_len stream")
                    try:
                        win = preprocess_stream(
                            win,
                            self.store_client,
                            winstart,
                            winend,
                            tl,
                            **self.options,
                        )
                        self.logger.debug(
                            "Core %d finished preprocessing corr_len stream"
                            % self.rank
                        )
                    except Exception as e:
                        if st.count():
                            self.logger.error(
                                "Stream preprocessing failed for "
                                f"{st[0].stats.network}.{st[0].stats.station}"
                                " and time "
                                f"{t}.\nThe Original Error Message was {e}."
                            )
                        else:
                            self.logger.error(
                                "Stream preprocessing failed for "
                                "time "
                                f"{t}.\nThe Original Error Message was {e}."
                            )
                        continue
                    if self.rank == 0:
                        self.logger.info("Recalculating combinations...")
                        self.options["combinations"] = calc_cross_combis(
                            win,
                            self.ex_dict,
                            self.options["combination_method"],
                            rcombis=self.rcombis,
                        )
                    else:
                        self.logger.info(
                            "Core %d waiting for combinations..." % self.rank
                        )
                        self.options["combinations"] = None
                    self.options["combinations"] = self.comm.bcast(
                        self.options["combinations"], root=0
                    )

                if not len(win):
                    # no new combinations for this time period
                    self.logger.info(
                        f"No new data for times {winstart}-{winend}"
                    )
                    continue

                win = win.merge()

                self.logger.debug(
                    "Core %d found %d traces." % (self.rank, len(win))
                )

                if self.options["joint_norm"]:
                    self.logger.info("Checking if 3 channels are available.")
                    check_for_missing_channels(win, self.avail_raw_data)

                win = win.trim(winstart, winend, pad=True)
                self.logger.info(
                    "Core %d provides %d traces for CC in time %s - %s."
                    % (
                        self.rank,
                        len(win),
                        winstart.strftime(TSTR_FMT),
                        winend.strftime(TSTR_FMT),
                    )
                )
                yield win, write_flag
                write_flag = False

    def _pxcorr_matrix(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ntrc = A.shape[0]
        # time domain processing
        # map of traces on processes
        ind = self._get_row_index_per_core(A)
        self.logger.info(
            "Core %d processing %d matrix rows" % (self.rank, np.sum(ind))
        )
        try:
            self.logger.debug(
                "Core {:d} working on indices {:d}-{:d}".format(
                    self.rank, *list(np.where(ind)[0][[0, -1]])
                )
            )
        except IndexError:
            self.logger.debug(
                "Core {:d} working on no indices".format(self.rank)
            )
        ######################################
        corr_args = self.options["corr_args"]
        # time domain pre-processing
        params = {}
        for key in list(corr_args.keys()):
            if "Processing" not in key:
                params.update({key: corr_args[key]})
        params["sampling_rate"] = self.sampling_rate
        # The steps that aren't done before

        # nans from the masked parts are set to 0
        np.nan_to_num(A, copy=False)

        for proc in corr_args["TDpreProcessing"]:
            self.logger.info("Running %s" % proc["function"])
            func = func_from_str(proc["function"])
            A[ind, :] = func(A[ind, :], proc["args"], params)
            self.logger.info("Finished %s" % proc["function"])

        # zero-padding
        A = pptd.zeroPadding(A, {"type": "avoidWrapFastLen"}, params)

        ######################################
        # FFT
        self.logger.info("Core %d performing FFT." % self.rank)

        # Allocate space for rfft of data
        # The rfft will only return the
        zmsize = A.shape

        # use next fast len instead?
        fftsize = zmsize[1] // 2 + 1
        B = np.zeros((ntrc, fftsize), dtype=np.csingle)

        B[ind, :] = np.fft.rfft(A[ind, :], axis=1)

        freqs = np.fft.rfftfreq(zmsize[1], 1.0 / self.sampling_rate)

        ######################################
        # frequency domain pre-processing
        params.update({"freqs": freqs})
        # Here, I will have to make sure to add all the functions to the module
        for proc in corr_args["FDpreProcessing"]:
            # The big advantage of this rather lengthy code is that we can also
            # import any function that has been defined anywhere else (i.e,
            # not only within the miic framework)
            self.logger.info("Running %s" % proc["function"])
            func = func_from_str(proc["function"])
            B[ind, :] = func(B[ind, :], proc["args"], params)
            self.logger.info("Finished %s" % proc["function"])

        ######################################
        # collect results
        self.comm.Allreduce(MPI.IN_PLACE, [B, MPI.FLOAT], op=MPI.SUM)

        ######################################
        # correlation
        self.logger.info("Core %d performing correlation." % self.rank)
        csize = len(self.options["combinations"])
        irfftsize = (fftsize - 1) * 2
        sampleToSave = int(
            np.ceil(corr_args["lengthToSave"] * self.sampling_rate)
        )
        C = np.zeros((csize, sampleToSave * 2 + 1), dtype=np.float32)

        pmap = np.arange(csize) * self.psize / csize
        pmap = pmap.astype(np.int32)
        ind = pmap == self.rank
        ind = np.arange(csize)[ind]
        startlags = np.zeros(csize, dtype=np.float32)
        self.logger.info(
            "Core %d correlating %d matrix rows" % (self.rank, ind.size)
        )
        for ii in ind:
            # offset of starttimes in samples(just remove fractions of samples)
            offset = (
                self.options["starttime"][self.options["combinations"][ii][0]]
                - self.options["starttime"][self.options["combinations"][ii][1]]
            )
            if corr_args["center_correlation"]:
                roffset = 0.0
            else:
                # offset exceeding a fraction of integer
                roffset = (
                    np.fix(offset * self.sampling_rate) / self.sampling_rate
                )
            # faction of samples to be compenasated by shifting
            offset -= roffset
            # normalization factor of fft correlation
            if corr_args["normalize_correlation"]:
                norm = (
                    np.sqrt(
                        2.0
                        * np.sum(
                            B[self.options["combinations"][ii][0], :]
                            * B[self.options["combinations"][ii][0], :].conj()
                        )
                        - B[self.options["combinations"][ii][0], 0] ** 2
                    )
                    * np.sqrt(
                        2.0
                        * np.sum(
                            B[self.options["combinations"][ii][1], :]
                            * B[self.options["combinations"][ii][1], :].conj()
                        )
                        - B[self.options["combinations"][ii][1], 0] ** 2
                    )
                    / irfftsize
                ).real
            else:
                norm = 1.0

            M = (
                B[self.options["combinations"][ii][0], :].conj()
                * B[self.options["combinations"][ii][1], :]
                * np.exp(1j * freqs * offset * 2 * np.pi)
            )

            ######################################
            # frequency domain postProcessing
            #
            tmp = np.fft.irfft(M).real

            # cut the center and do fftshift
            self.logger.debug("Normalizing ccf index %d with %f" % (ii, norm))
            C[ii, :] = (
                np.concatenate((tmp[-sampleToSave:], tmp[: sampleToSave + 1]))
                / norm
            )
            startlags[ii] = -sampleToSave / self.sampling_rate - roffset

        ######################################
        # time domain postProcessing

        ######################################
        # collect results
        # self.logger.debug("%s %s" % (C.shape, C.dtype))
        # self.logger.debug("combis: %s" % (self.options["combinations"]))

        self.comm.Allreduce(MPI.IN_PLACE, [C, MPI.FLOAT], op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, [startlags, MPI.FLOAT], op=MPI.SUM)
        self.logger.info("Core %d finished correlation." % self.rank)
        return (C, startlags)

    def _get_row_index_per_core(self, A: np.ndarray) -> np.ndarray:
        """
        Get indices for matrix rows to be processed by each core.

        If joint_norm is set to True, the same station has to be processed by
        the same core. We assume that three consecutive rows in the matrix
        belong to the same station. Thus, we make sure that number of rows
        in submatrix is divisible by 3.

        :param A: Input matrix
        :type A: np.ndarray
        :return: Indices for each core
        :rtype: np.ndarray
        :note: The function is called in the _pxcorr_matrix function.
        """
        if self.options["joint_norm"]:
            # For joint normalization, we need to make sure that the same
            # station is processed by the same core
            if A.shape[0] % 3 != 0:
                raise ValueError(
                    "Matrix size is not divisible by 3. Joint normalization "
                    + "is not possible."
                )

            nsta = A.shape[0] / 3
            pmap = np.arange(nsta) * self.psize // nsta
            pmap = np.repeat(pmap, 3)
            ind = pmap == self.rank
        else:
            ntrc = A.shape[0]
            pmap = np.arange(ntrc) * self.psize // ntrc
            ind = pmap == self.rank

        return ind


def st_to_np_array(st: Stream, npts: int) -> Tuple[np.ndarray, Stream]:
    """
    Converts an obspy stream to a matrix with the shape (npts, st.count()).
    Also returns the same stream but without the data arrays in tr.data.

    :param st: Input Stream
    :type st: Stream
    :param npts: Maximum number of samples per Trace
    :type npts: int
    :return: A stream and a matrix
    :rtype: np.ndarray
    """
    A = np.zeros((st.count(), npts), dtype=np.float32)
    for ii, tr in enumerate(st):
        A[ii, : tr.stats.npts] = tr.data
        del tr.data  # Not needed any more, just uses up RAM
    return A, st


def _compare_existing_data(ex_corr: dict, tr0: Trace, tr1: Trace) -> bool:
    # The actual starttime for the header is the later one of the two
    net0 = tr0.stats.network
    stat0 = tr0.stats.station
    cha0 = tr0.stats.channel
    net1 = tr1.stats.network
    stat1 = tr1.stats.station
    cha1 = tr1.stats.channel
    loc0 = tr0.stats.location
    loc1 = tr1.stats.location
    # Probably faster than checking a huge dict twice
    flip = (
        [net0, net1],
        [stat0, stat1],
        [loc0, loc1],
        [cha0, cha1],
    ) != sort_comb_name_alphabetically(
        net0, stat0, net1, stat1, loc0, loc1, cha0, cha1
    )
    corr_start = max(tr0.stats.starttime, tr1.stats.starttime)
    try:
        if flip:
            return (
                corr_start.format_fissures()
                in ex_corr[f"{net1}.{stat1}"][f"{net0}.{stat0}"][
                    f"{loc1}-{loc0}"
                ][f"{cha1}-{cha0}"]
            )
        else:
            return (
                corr_start.format_fissures()
                in ex_corr[f"{net0}.{stat0}"][f"{net1}.{stat1}"][
                    f"{loc0}-{loc1}"
                ][f"{cha0}-{cha1}"]
            )
    except KeyError:
        return False


def is_in_xcombis(id1: str, id2: str, rcombis: List[str] = None) -> bool:
    """
    Check if the specific combination is to be calculated according to
    xcombinations including the channel. xcombination are expected as
    Net1-Net2.Sta1-Sta2.Cha1-Cha2. (Channel information can be omitted)
    """
    n1, s1, _, c1 = id1.split(".")
    n2, s2, _, c2 = id2.split(".")
    tcombi = f"{n1}-{n2}.{s1}-{s2}.{c1}-{c2}"
    tcombi2 = f"{n2}-{n1}.{s2}-{s1}.{c2}-{c1}"
    for combi in rcombis:
        if fnmatch.fnmatch(tcombi, combi + "*") or fnmatch.fnmatch(
            tcombi2, combi + "*"
        ):
            return True
    return False


def calc_cross_combis(
    st: Stream,
    ex_corr: dict,
    method: str = "betweenStations",
    rcombis: List[str] = None,
) -> list:
    """
    Calculate a list of all cross correlation combination
    of traces in the stream: i.e. all combination with two different
    stations involved.

    :param st: Stream holding the tracecs to be correlated
    :type st: :class:`~obspy.Stream`
    :param ex_corr: dict holding the correlations that already exist in db
    :type ex_corr: dict
    :type method: stringf
    :param method: Determines which traces of the strem are combined.
    :param rcombis: requested combinations, only works if
        `method==betweenStations`.
    :type rcombis: List[str] strings are in form net0-net1.stat0-stat1

        ``'betweenStations'``:
            Traces are combined if either their station or
            their network names are different.
        ``'betweenComponents'``:
            Traces are combined if their components (last
            letter of channel name) names are different and their station and
            network names are identical (single station cross-correlation).
        ``'autoComponents'``:
            Traces are combined only with themselves.
        ``'allSimpleCombinations'``:
            All Traces are combined once (onle one of
            (0,1) and (1,0))
        ``'allCombinations'``:
            All traces are combined in both orders ((0,1) and (1,0))
    """
    # deprecate allCombinations
    if method == "allCombinations":
        warn(
            'Method "allCombinations" is deprecated. Using '
            '"allSimpleCombinations" instead.',
            DeprecationWarning,
        )
        method = "allSimpleCombinations"
    combis = []
    # sort alphabetically
    st = st.sort()
    if method == "betweenStations":
        for ii, tr in enumerate(st):
            for jj in range(ii + 1, len(st)):
                tr1 = st[jj]
                n = tr.stats.network
                n2 = tr1.stats.network
                s = tr.stats.station
                s2 = tr1.stats.station
                if n != n2 or s != s2:
                    # check first whether this combi is in dict
                    if _compare_existing_data(ex_corr, tr, tr1):
                        continue
                    if rcombis is not None and not is_in_xcombis(
                        tr.id, tr1.id, rcombis
                    ):
                        continue
                    combis.append((ii, jj))
    elif method == "betweenComponents":
        for ii, tr in enumerate(st):
            for jj in range(ii + 1, len(st)):
                tr1 = st[jj]
                if (
                    (tr.stats["network"] == tr1.stats["network"])
                    and (tr.stats["station"] == tr1.stats["station"])
                    and (tr.stats["channel"][-1] != tr1.stats["channel"][-1])
                ):
                    if _compare_existing_data(ex_corr, tr, tr1):
                        continue
                    combis.append((ii, jj))
    elif method == "autoComponents":
        for ii, tr in enumerate(st):
            if _compare_existing_data(ex_corr, tr, tr):
                continue
            combis.append((ii, ii))
    elif method == "allSimpleCombinations":
        for ii, tr in enumerate(st):
            for jj in range(ii, len(st)):
                tr1 = st[jj]
                if _compare_existing_data(ex_corr, tr, tr1):
                    continue
                combis.append((ii, jj))
    elif method == "allCombinations":
        for ii, tr in enumerate(st):
            for jj, tr1 in enumerate(st):
                if _compare_existing_data(ex_corr, tr, tr1):
                    continue
                combis.append((ii, jj))
    else:
        raise ValueError(
            "Method has to be one of ('betweenStations', "
            "'betweenComponents', 'autoComponents', "
            "'allSimpleCombinations' or 'allCombinations')."
        )
    if not len(combis):
        warn("Method %s found no combinations." % method, UserWarning)
    return combis


# All the rotations are still untested, should do that at some point

# def rotate_multi_corr_stream(st: Stream) -> Stream:
#     """Rotate a stream with full Greens tensor from ENZ to RTZ

#     Take a stream with numerous correlation traces and rotate the
#     combinations of ENZ components into combinations of RTZ components in
#     all nine components of the Green's tensor are present. If not all nine
#     components are present no trace for this station combination is returned.

#     :type st: obspy.stream
#     :param st: stream with data in ENZ system
#     :rtype: obspy.stream
#     :return: stream in the RTZ system
#     """

#     out_st = Stream()
#     while st:
#         tl = list(range(9))
#         tst = st.select(network=st[0].stats['network'],
#                         station=st[0].stats['station'])
#         cnt = 0
#         for ttr in tst:
#             if ttr.stats['channel'][2] == 'E':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[0] = ttr
#                     cnt += 1
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[1] = ttr
#                     cnt += 2
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[2] = ttr
#                     cnt += 4
#             elif ttr.stats['channel'][2] == 'N':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[3] = ttr
#                     cnt += 8
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[4] = ttr
#                     cnt += 16
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[5] = ttr
#                     cnt += 32
#             elif ttr.stats['channel'][2] == 'Z':
#                 if ttr.stats['channel'][6] == 'E':
#                     tl[6] = ttr
#                     cnt += 64
#                 elif ttr.stats['channel'][6] == 'N':
#                     tl[7] = ttr
#                     cnt += 128
#                 elif ttr.stats['channel'][6] == 'Z':
#                     tl[8] = ttr
#                     cnt += 256
#         if cnt == 2**9-1:
#             st0 = Stream()
#             for t in tl:
#                 st0.append(t)
#             st1 = _rotate_corr_stream(st0)
#             out_st += st1
#         elif cnt == 27:  # only horizontal component combinations present
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#         elif cnt == 283:  # horizontal combinations + ZZ
#             st0 = Stream()
#             for t in [0, 1, 3, 4]:
#                 st0.append(tl[t])
#             st1 = _rotate_corr_stream_horizontal(st0)
#             out_st += st1
#             out_st.append(tl[8])
#         for ttr in tst:
#             for ind, tr in enumerate(st):
#                 if ttr.id == tr.id:
#                     st.pop(ind)

#     return out_st


# def _rotate_corr_stream_horizontal(st: Stream) -> Stream:
#     """ Rotate traces in stream from the EE-EN-NE-NN system to
#     the RR-RT-TR-TT system. The letters give the component order
#     in the input and output streams. Input traces are assumed to be of same
#     length and simultaneously sampled.
#     """

#     # rotation angles
#     # phi1 : counter clockwise angle between E and R(towards second station)
#     # the leading -1 accounts fact that we rotate the coordinate system,
#     # not a vector
#     phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
#     # phi2 : counter clockwise angle between E and R(away from first station)
#     phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)

#     c1 = np.cos(phi1)
#     s1 = np.sin(phi1)
#     c2 = np.cos(phi2)
#     s2 = np.sin(phi2)

#     rt = Stream()
#     RR = st[0].copy()
#     RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[2].data +\
#         s1*s2*st[3].data
#     tcha = list(RR.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'R'
#     RR.stats['channel'] = ''.join(tcha)
#     rt.append(RR)

#     RT = st[0].copy()
#     RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[2].data -\
#         s1*c2*st[3].data
#     tcha = list(RT.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'T'
#     RT.stats['channel'] = ''.join(tcha)
#     rt.append(RT)

#     TR = st[0].copy()
#     TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[2].data -\
#         c1*s2*st[3].data
#     tcha = list(TR.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'R'
#     TR.stats['channel'] = ''.join(tcha)
#     rt.append(TR)

#     TT = st[0].copy()
#     TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[2].data +\
#         c1*c2*st[3].data
#     tcha = list(TT.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'T'
#     TT.stats['channel'] = ''.join(tcha)
#     rt.append(TT)

#     return rt


# def _rotate_corr_stream(st: Stream) -> Stream:
#     """ Rotate traces in stream from the EE-EN-EZ-NE-NN-NZ-ZE-ZN-ZZ system to
#     the RR-RT-RZ-TR-TT-TZ-ZR-ZT-ZZ system. The letters give the component
#     in the input and output streams. Input traces are assumed to be of same
#     length and simultaneously sampled.
#     """

#     # rotation angles
#     # phi1 : counter clockwise angle between E and R(towards second station)
#     # the leading -1 accounts fact that we rotate the coordinate system,
#     # not a vector
#     phi1 = - np.pi/180*(90-st[0].stats['sac']['az'])
#     # phi2 : counter clockwise angle between E and R(away from first station)
#     phi2 = - np.pi/180*(90-st[0].stats['sac']['baz']+180)

#     c1 = np.cos(phi1)
#     s1 = np.sin(phi1)
#     c2 = np.cos(phi2)
#     s2 = np.sin(phi2)

#     rtz = Stream()
#     RR = st[0].copy()
#     RR.data = c1*c2*st[0].data - c1*s2*st[1].data - s1*c2*st[3].data +\
#         s1*s2*st[4].data
#     tcha = list(RR.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'R'
#     RR.stats['channel'] = ''.join(tcha)
#     rtz.append(RR)

#     RT = st[0].copy()
#     RT.data = c1*s2*st[0].data + c1*c2*st[1].data - s1*s2*st[3].data -\
#         s1*c2*st[4].data
#     tcha = list(RT.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'T'
#     RT.stats['channel'] = ''.join(tcha)
#     rtz.append(RT)

#     RZ = st[0].copy()
#     RZ.data = c1*st[2].data - s1*st[5].data
#     tcha = list(RZ.stats['channel'])
#     tcha[2] = 'R'
#     tcha[6] = 'Z'
#     RZ.stats['channel'] = ''.join(tcha)
#     rtz.append(RZ)

#     TR = st[0].copy()
#     TR.data = s1*c2*st[0].data - s1*s2*st[1].data + c1*c2*st[3].data -\
#         c1*s2*st[4].data
#     tcha = list(TR.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'R'
#     TR.stats['channel'] = ''.join(tcha)
#     rtz.append(TR)

#     TT = st[0].copy()
#     TT.data = s1*s2*st[0].data + s1*c2*st[1].data + c1*s2*st[3].data +\
#         c1*c2*st[4].data
#     tcha = list(TT.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'T'
#     TT.stats['channel'] = ''.join(tcha)
#     rtz.append(TT)

#     TZ = st[0].copy()
#     TZ.data = s1*st[2].data + c1*st[5].data
#     tcha = list(TZ.stats['channel'])
#     tcha[2] = 'T'
#     tcha[6] = 'Z'
#     TZ.stats['channel'] = ''.join(tcha)
#     rtz.append(TZ)

#     ZR = st[0].copy()
#     ZR.data = c2*st[6].data - s2*st[7].data
#     tcha = list(ZR.stats['channel'])
#     tcha[2] = 'Z'
#     tcha[6] = 'R'
#     ZR.stats['channel'] = ''.join(tcha)
#     rtz.append(ZR)

#     ZT = st[0].copy()
#     ZT.data = s2*st[6].data + c2*st[7].tuple
#     ZT.stats['channel'] = ''.join(tcha)
#     rtz.append(ZT)

#     rtz.append(st[8].copy())

#     return rtz


def sort_comb_name_alphabetically(
    network1: str,
    station1: str,
    network2: str,
    station2: str,
    location1: Optional[str] = "",
    location2: Optional[str] = "",
    channel1: Optional[str] = "",
    channel2: Optional[str] = "",
) -> Tuple[list, list]:
    """
    Returns the alphabetically sorted network and station codes from the two
    station.

    :param network1: network code of first station
    :type network1: str
    :param station1: station code of first station
    :type station1: str
    :param network2: Network code of second station
    :type network2: str
    :param station2: Station Code of second Station
    :type station2: str
    :return: A tuple containing the list of the network codes sorted
        and the list of the station codes sorted.
    :rtype: Tuple[ list, list]

    Examples
    --------
    >>> net1 = 'IU'  # Network Code of first station
    >>> stat1 = 'HRV'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['IU', 'XN'], ['HRV', 'NEP06'])
    >>> print(sort_comb_name_aphabetically(
            net2, stat2, net1, stat1))
    (['IU', 'XN'], ['HRV', 'NEP06'])
    >>> # Different combination
    >>> net1 = 'YP'  # Network Code of first station
    >>> stat1 = 'AB3'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['XN', 'YP'], ['NEP06', 'AB3'])
    >>> # Different combination
    >>> net1 = 'XN'  # Network Code of first station
    >>> stat1 = 'NEP07'  # Station Code of first station
    >>> net2 = 'XN'
    >>> stat2 = 'NEP06'
    >>> print(sort_comb_name_aphabetically(
            net1, stat1, net2, stat2))
    (['XN', 'XN'], ['NEP06', 'NEP07'])
    """
    if not all(
        [
            isinstance(arg, str)
            for arg in [
                network1,
                network2,
                station1,
                station2,
                location2,
                location1,
                channel1,
                channel2,
            ]
        ]
    ):
        raise TypeError("All arguments have to be strings.")
    sort1 = network1 + station1 + location1 + channel1
    sort2 = network2 + station2 + location2 + channel2
    sort = [sort1, sort2]
    sorted = sort.copy()
    sorted.sort()
    if sort == sorted:
        netcomb = [network1, network2]
        statcomb = [station1, station2]
        loccomb = [location1, location2]
        chacomb = [channel1, channel2]
    else:
        netcomb = [network2, network1]
        statcomb = [station2, station1]
        loccomb = [location2, location1]
        chacomb = [channel2, channel1]
    return netcomb, statcomb, loccomb, chacomb


def compute_network_station_combinations(
    netlist: list,
    statlist: list,
    method: str = "betweenStations",
    combis: List[str] = None,
) -> Tuple[list, list]:
    """
    Return the network and station codes of the correlations for the provided
    lists of networks and stations and the queried combination method.

    :param netlist: List of network codes
    :type netlist: list
    :param statlist: List of Station Codes
    :type statlist: list
    :param method: The combination method to use. Has to be one of the
        following: `betweenStation`, `betweenComponents`, `autoComponents`,
        `allSimpleCombinations`, or `allCombinations`,
        defaults to 'betweenStations'.
    :type method: str, optional
    :param combis: List of desired station combinations.
        Given as [net0-net1.stat0-stat1]. Optional.
    :type combis: List[str]
    :raises ValueError: for unkown combination methods.
    :return: A tuple containing the list of the correlation network code
        and the list of the correlation station code.
    :rtype: Tuple[list, list]
    """
    if method == "allCombinations":
        warn(
            "allCombinations is deprecated, "
            "using allSimpleCombinations instead.",
            DeprecationWarning,
        )
        method = "allSimpleCombinations"
    netcombs = []
    statcombs = []
    if method == "betweenStations":
        for ii, (n, s) in enumerate(zip(netlist, statlist)):
            for jj in range(ii + 1, len(netlist)):
                n2 = netlist[jj]
                s2 = statlist[jj]
                if n != n2 or s != s2:
                    nc, sc, _, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                    # Check requested combinations
                    if (
                        combis is not None
                        and f"{nc[0]}-{nc[1]}.{sc[0]}-{sc[1]}" not in combis
                    ):
                        continue
                    netcombs.append("%s-%s" % (nc[0], nc[1]))
                    statcombs.append("%s-%s" % (sc[0], sc[1]))

    elif method == "betweenComponents" or method == "autoComponents":
        netcombs = [n + "-" + n for n in netlist]
        statcombs = [s + "-" + s for s in statlist]
    elif method == "allSimpleCombinations":
        for ii, (n, s) in enumerate(zip(netlist, statlist)):
            for jj in range(ii, len(netlist)):
                n2 = netlist[jj]
                s2 = statlist[jj]
                nc, sc, _, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                netcombs.append("%s-%s" % (nc[0], nc[1]))
                statcombs.append("%s-%s" % (sc[0], sc[1]))
    elif method == "allCombinations":
        for n, s in zip(netlist, statlist):
            for n2, s2 in zip(netlist, statlist):
                nc, sc, _, _ = sort_comb_name_alphabetically(n, s, n2, s2)
                netcombs.append("%s-%s" % (nc[0], nc[1]))
                statcombs.append("%s-%s" % (sc[0], sc[1]))
    else:
        raise ValueError(
            "Method has to be one of ('betweenStations', "
            "'betweenComponents', 'autoComponents', "
            "'allSimpleCombinations' or 'allCombinations')."
        )
    return netcombs, statcombs


def preprocess_stream(
    st: Stream,
    store_client: Store_Client,
    startt: UTCDateTime,
    endt: UTCDateTime,
    taper_len: float,
    remove_response: bool,
    subdivision: dict,
    preProcessing: List[dict] = None,
    **kwargs,
) -> Stream:
    """
    Does the preprocessing on a per stream basis. Most of the parameters can be
    fed in by using the "yaml" dict as kwargs.

    :param st: Input Stream to be processed
    :type st: :class:`obspy.core.stream.Stream`
    :param store_client: Store Client for the database
    :type store_client: :class:`~seismic.trace_data.waveform.Store_Client`
    :param inv: Station response, can be None if ``remove_response=False``.
    :type inv: Inventory or None
    :param startt: Starttime that the stream should be clipped / padded to.
    :type startt: :class:`obspy.UTCDateTime`
    :param endt: Endtime that the stream should be clipped / padded to.
    :type endt: :class:`obspy.UTCDateTime`
    :param taper_len: If the instrument response is removed, one might want to
        taper to mitigate the acausal effects of the deconvolution. This is
        the length of such a taper in seconds.
    :type taper_len: float
    :param remove_response: Should the instrument response be removed?
    :type remove_response: bool
    :param subdivision: Dictionary holding information about the correlation
        lenghts and increments.
    :type subdivision: dict
    :param preProcessing: List holding information about the different external
        preprocessing functions to be applied, defaults to None
    :type preProcessing: List[dict], optional
    :raises ValueError: For sampling rates higher than the stream's native
        sampling rate (upsampling is not permitted).
    :return: The preprocessed stream.
    :rtype: :class:`obspy.core.stream.Stream`
    """
    if not st.count():
        return st
    st = ppst.detrend_st(st, "linear")
    # deal with overlaps
    # This should be a setting in the parameter file
    st = mu.gap_handler(st, 1, taper_len * 4, taper_len)
    if not st.count():
        # could happen after handling gaps
        return st
    # To deal with any nans/masks
    st = st.split()
    st = st.sort(keys=["starttime"])

    inv = store_client.inventory
    if remove_response:
        try:
            st.remove_response(taper=False, inventory=inv)
            module_logger.debug("Removed instrument response from stream ...")
        except ValueError:
            module_logger.warning(
                "Station response not found ... loading" + " from remote."
            )
            # missing station response
            ninv = store_client.rclient.get_stations(
                network=st[0].stats.network,
                station=st[0].stats.station,
                channel="*",
                level="response",
            )
            st.remove_response(taper=False, inventory=ninv)
            store_client._write_inventory(ninv)
            inv += ninv

    # Sometimes Z has reversed polarity
    if inv:
        try:
            mu.correct_polarity(st, inv)
        except Exception:
            module_logger.error(
                "Exception while checking polarity of inventory ...",
                exc_info=True,
            )

    mu.discard_short_traces(st, subdivision["corr_len"] / 20)

    if preProcessing:
        for procStep in preProcessing:
            module_logger.debug(
                "Applying preprocessing function %s with args %s"
                % (procStep["function"], procStep["args"])
            )
            if (
                "detrend_st" in procStep["function"]
                or "cos_taper_st" in procStep["function"]
            ):
                warnings.warn(
                    "Tapering and Detrending are now always perfomed "
                    "as part of the preprocessing. Ignoring parameter...",
                    DeprecationWarning,
                )
                continue
            func = func_from_str(procStep["function"])
            st = func(st, **procStep["args"])
    st.merge()
    st.trim(startt, endt, pad=True)

    mu.discard_short_traces(st, subdivision["corr_len"] / 20)
    return st


def generate_corr_inc(
    st: Stream, subdivision: dict, read_len: int, **kwargs
) -> Iterator[Stream]:
    """
    Subdivides the preprocessed streams into parts of equal length using
    the parameters ``cor_inc`` and ``cor_len`` in ``subdivision``.
    This function can be acessed by several processes in parallel

    :param st: The preprocessed input stream
    :type st: :class:`obspy.core.stream.Stream`
    :param subdivision: Dictionary holding the information about the
        correlation length and increment.
    :type subdivision: dict
    :param read_len: Length to be read from disk in seconds
    :type read_len: int
    :yield: Equal length windows, padded with nans / masked if data is missing.
    :rtype: Generator[Stream]
    """

    try:
        # Second loop to return the time window in correlation length
        for ii, win0 in enumerate(
            st.slide(
                subdivision["corr_len"] - st[0].stats.delta,
                subdivision["corr_inc"],
                include_partial_windows=True,
            )
        ):
            # We use trim so the windows have the right time and
            # are filled with masked arrays for places without values
            starttrim = st[0].stats.starttime + ii * subdivision["corr_inc"]
            endtrim = (
                st[0].stats.starttime
                + ii * subdivision["corr_inc"]
                + subdivision["corr_len"]
                - st[0].stats.delta
            )
            win = win0.trim(starttrim, endtrim, pad=True)
            mu.get_valid_traces(win)
            yield win

    except IndexError:
        # processes with no data end up here
        win = Stream()
        # A little dirty, but it has to go through an equally long loop
        # else this will cause a deadlock
        for _ in range(int(np.ceil(read_len / subdivision["corr_inc"]))):
            yield win


def check_for_missing_channels(st: Stream, avail_channels: list):
    """
    Check if single channels are missing in the stream and add them as 0s.

    If entire stations are missing, they are not added.

    :param st: Stream to be checked
    :type st: Stream
    :param avail_channels: List of available channels, each element in
        the form [network, station, location, channel]
    :type avail_channels: list
    """
    unique_nslc = set([tr.id for tr in st])
    nsl_in_st = [tr.id.rpartition(".")[0] for tr in st]
    avail_nslc = set([".".join(item) for item in avail_channels])

    missing_id_in_st = avail_nslc.difference(unique_nslc)

    for nslc in missing_id_in_st:
        nsl = nslc.rpartition(".")[0]
        if nsl not in nsl_in_st:
            module_logger.debug(
                "Station %s is missing in stream." % nsl
                + " Not adding missing channel %s." % nslc
            )
            continue
        module_logger.debug("Adding missing channel %s." % nslc)
        ind = nsl_in_st.index(nsl)
        trins = st[ind].copy()
        trins.id = nslc
        trins.data = np.zeros_like(trins.data)
        st.insert(ind, trins)
    st.sort()
