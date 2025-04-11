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
HANDLERNAME_FILE = "default-file"
HANDLERNAME_CONSOLE = "default-console"
DEFAULT_HANDLERNAMES = [HANDLERNAME_CONSOLE, HANDLERNAME_FILE]
LOGDIR = "log"
LOG_TSTRFMT = '%Y-%m-%d-%H-%M-%S'
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
        logger.info("Found no duplicate loggers")
        return

    for h in handlers.values():
        for hi in h[1:]:
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

    def _set_filename(self, logdir=LOGDIR, filename_fmt=FILENAME_FMT):
        """Set filename for log file."""
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
            os.makedirs(logdir, exist_ok=True)

    @property
    def _default_handlers_set(self) -> None:
        try:
            _logger = self.logger.parent
        except AttributeError as E:
            raise E("Logger not set. Call _set_logger first.")
        if not _logger.hasHandlers():
            return False
        else:
            handlers = get_handlers_by_name(_logger)
            hns = list(handlers.keys())
            if len(hns) != len(DEFAULT_HANDLERNAMES):
                return False
            return all([dhn in hns for dhn in DEFAULT_HANDLERNAMES])

    def _set_logger(self, loglevel):
        loglvl = loglevel.upper()
        loggername = ".".join([self.__module__, "",
                               self.__class__.__name__
                               + RANK_STRFMT.format(rank=self.rank)])
        self.logger = logging.getLogger(loggername)
        self.logger.parent.setLevel(loglvl)
        logging.captureWarnings(True)

    def _set_check_default_handlers(self):
        if not self._default_handlers_set:
            loglvl = self.logger.parent.getEffectiveLevel()
            set_fileHandler(self.logger.parent, self.logfilename,
                            loglvl, HANDLERNAME_FILE)
            set_consoleHandler(self.logger.parent, loglvl,
                               HANDLERNAME_CONSOLE)
        remove_duplicate_handlers(self.logger.parent)

        self.logger.debug("ID of core {:01d} is {:d}".format(
            self.rank, id(self.comm)))
        self.logger.debug("My parent logger is %s" % self.logger.parent.name)

    def set_logger(self, loglevel="DEBUG", logdir=LOGDIR,
                   filename_fmt=FILENAME_FMT):
        """
        Set logger including default handlers.

        **Only function to be executed after initialization!**

        :param loglevel: Level of verbosity of log messages. See :mod:`logging`
            for details.
        :type loglevel: str ["WARNING", "INFO", "DEBUG"]
        :param logdir: directory of the log files.
        :type logdir: str
        :param filename_fmt: string using 'format()', containing variables
            `rank`, `classname` and `exectimestr`. `rank` must be a digit
            format.
            Default is "{classname}-r"+{rank:03d}+"_{exectimestr}.log"
        :type filename_fmt: str
        """
        self._set_logger(loglevel)
        self._set_filename(logdir, filename_fmt)
        self._mk_logdir(logdir)
        self.comm.Barrier()
        self._set_check_default_handlers()
        self.logger.info("Logging to file %s" % self.logfilename
                         + " and console")
