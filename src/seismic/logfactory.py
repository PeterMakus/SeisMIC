'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Johanna Lehr (jlehr@gfz.de)

Created: Friday, 20th December 2024 14:00:00 pm
Last Modified: Friday, 11th April 2025 03:25:38 pm
'''
import logging
import os
from datetime import datetime
from mpi4py import MPI


cformatter = logging.Formatter(
    ('%(asctime)s - %(name)s.%(funcName)s - %(process)s - %(levelname)s: '
     + '%(message)s'),
    datefmt='%y-%m-%d %H:%M:%S')
LOGGER_LOGLVL = "WARNING"
HANDLER_LOGLVL = "DEBUG"
HANDLERNAME_CONSOLE = "default-console"
LOGDIR = None
PERF_LOGLEVEL = "INFO"
LOG_TSTRFMT = '%Y-%m-%dT%H%M%S'
RANK_STRFMT = "{rank:03d}"  # "%03d"
FILENAME_FMT = "{classname}-r"+RANK_STRFMT+"_{exectimestr}.log"


def create_logger() -> logging.Logger:
    """
    Set logger for the package.
    """
    # Try to get the package name, may not work for python <3.9 versions
    try:
        if __package__ is None and __name__ != "__main__":
            loggername = __name__.split('.')[0]
        elif __package__ == "":
            loggername = "seismic"
        else:
            loggername = __package__
    except UnboundLocalError:
        print("Error, using ", __name__.split('.')[0])
        loggername = __name__.split('.')[0]

    logger = logging.getLogger(loggername)
    return logger


def set_consoleHandler(logger, loglevel="DEBUG",
                       handlername="console"):
    """
    Add StreamHandler (stdout) using our our formatter.
    """
    ch = logging.StreamHandler()
    ch.set_name(handlername)
    ch.setLevel(loglevel)
    ch.setFormatter(cformatter)
    logger.addHandler(ch)
    logger.debug("Added console handler")


def set_fileHandler(logger, filename, loglevel="DEBUG",
                    handlername="file"):
    """
    Add FileHandler using our formatter.
    """
    fh = logging.FileHandler(filename,)
    fh.set_name(handlername)
    fh.setLevel(loglevel)
    fh.setFormatter(cformatter)
    logger.addHandler(fh)
    logger.debug("Added file handler %s" % str(fh))


def get_handlers_by_name(logger) -> dict:
    """
    Returns dictionary with handler names as keys and list of handlers as
    values.
    """
    handlers = {}
    for h in (logger.handlers):
        try:
            handlers[h.name].append(h)
        except KeyError:
            handlers[h.name] = [h]
    return handlers


def get_duplicate_handlers(logger) -> dict:
    """
    Returns dictionary with handler names as keys and list of handlers as
    values if there are more than one handler with the same name.
    """
    handlers = get_handlers_by_name(logger)
    handlers = {hn: h for hn, h in handlers.items() if len(h) > 1}
    return handlers


def remove_duplicate_handlers(logger):
    """
    Remove all but the first handler with the same name.
    """
    handlers = get_duplicate_handlers(logger)
    if len(handlers) == 0:
        logger.debug("Found no duplicate loggers")
        return

    for h in handlers.values():
        for hi in h[:-1]:
            logger.removeHandler(hi)


class LoggingMPIBaseClass():
    """
    Class for logging with MPI support.

    This class is intended to be used as a base class for classes that need
    a logger with MPI support. It provides a logger with default handlers which
    log to a file and to the console.
    The filename is created using the class name, the rank of the MPI process
    and the current time. Each process has its own log file.

    .. note:: For developers:
        This class serves as base class for classes that need a logger, such as
        :class:`seismic.correlate.correlate.Correlator` or
        class:`seismic.monitor.Monitor`. It must be initialized early in the
        __init__() function of the derived class. The logger is set up with the
        function :func:`set_logger`. In the child classes it is recommened to
        derive the arguments of :func:`set_logger` from the user parameters
        dict (or yaml file), e.g. the log level and directory. The filename
        format is currently fixed and not derived from the user parameters.
    """
    def __init__(self):
        # init MPI
        self.comm = MPI.COMM_WORLD
        self.psize = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.logfilename = None

        loggername = ".".join([self.__module__, "",
                               self.__class__.__name__
                               + RANK_STRFMT.format(rank=self.rank)])
        self.logger = logging.getLogger(loggername)
        logging.captureWarnings(True)

    def _set_filename(self, logdir=LOGDIR, filename_fmt=FILENAME_FMT):
        """Set filename for log file."""
        if logdir is None:
            return

        if self.rank == 0:
            tstr = datetime.now().strftime(LOG_TSTRFMT)
        else:
            tstr = None
        tstr = self.comm.bcast(tstr, root=0)
        filename = filename_fmt.format(classname=self.__class__.__name__,
                                       rank=self.rank, exectimestr=tstr)
        self.logfilename = os.path.join(logdir, filename)

    def _mk_logdir(self, logdir=LOGDIR):
        if self.rank == 0:
            if not logdir:
                pass
            else:
                os.makedirs(logdir, exist_ok=True)
        self.comm.Barrier()

    def _set_handlers(self):
        if self.logfilename is not None:
            set_fileHandler(self.logger.parent, self.logfilename,
                            HANDLER_LOGLVL, self.logfilename)
            self.logger.info("Logging to file %s" % self.logfilename)
        set_consoleHandler(self.logger.parent, HANDLER_LOGLVL,
                           HANDLERNAME_CONSOLE)
        self.logger.info("Logging to console")
        remove_duplicate_handlers(self.logger.parent)

        self.logger.debug("ID of core {:01d} is {:d}".format(
            self.rank, id(self.comm)))
        self.logger.debug("My parent logger is %s" % self.logger.parent.name)

    def _set_perf_logger(self):
        """
        Set dedicated logger for performance metrics.

        This logger does not propagate to the parent logger and can therefore
        record timings independently from the package log level.
        """
        self.perf_logger = logging.getLogger(f"{self.logger.name}.perf")
        self.perf_logger.setLevel(PERF_LOGLEVEL)
        self.perf_logger.propagate = False

        # Always add console handler for real-time monitoring
        set_consoleHandler(
            self.perf_logger,
            loglevel=PERF_LOGLEVEL,
            handlername="perf-console",
        )

        # Add file handler if logfilename is available
        if self.logfilename is not None:
            perf_logfile = f"{self.logfilename}.performance"
            set_fileHandler(
                self.perf_logger,
                perf_logfile,
                loglevel=PERF_LOGLEVEL,
                handlername=f"perf-{os.path.basename(perf_logfile)}",
            )
            self.logger.debug("Performance logging to %s", perf_logfile)

        remove_duplicate_handlers(self.perf_logger)

    def set_logger(self, loglevel=LOGGER_LOGLVL,
                   logdir=LOGDIR | str | os.PathLike,
                   filename_fmt=FILENAME_FMT):
        """
        Set logger including default handlers.

        **Only function to be executed after initialization!**

        Loglevel is set on parent logger ("seismic") to ensure that all log
        messages in the package are logged.

        :param loglevel: Level of verbosity of log messages. See :mod:`logging`
            for details.
        :type loglevel: str ["WARNING", "INFO", "DEBUG"]
        :param logdir: directory of the log files. If None, no log file is
            created. Default is None.
        :type logdir: str or os.PathLike
        :param filename_fmt: string using 'format()', containing variables
            `rank`, `classname` and `exectimestr`. `rank` must be a digit
            format.
            Default is "{classname}-r"+{rank:03d}+"_{exectimestr}.log"
        :type filename_fmt: str
        """
        self.logger.parent.setLevel(loglevel.upper())
        self._set_filename(logdir, filename_fmt)
        self._mk_logdir(logdir)
        self._set_handlers()

        warnlog = logging.getLogger("py.warnings")
        for h in self.logger.parent.handlers:
            if isinstance(h, logging.FileHandler):
                warnlog.addHandler(h)
                self.logger.debug(
                    "Adding handler: {} to warn logger".format(h)
                )
