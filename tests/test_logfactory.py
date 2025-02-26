import logging.handlers
from datetime import datetime
from unittest import TestCase, main, mock
import logging
from seismic import logfactory
import seismic


class TestLogfactoryMock(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logger = mock.create_autospec(logging.Logger)
        logger.name = seismic.__name__
        handlers = []
        for hn in ["handler1", "handler2", "handler2"]:
            handler = mock.create_autospec(logging.StreamHandler)
            handler.name = hn
            handlers.append(handler)
        logger.handlers = handlers
        cls.logger = logger

    def setUp(self) -> None:
        return super().setUp()

    @mock.patch("seismic.logfactory.logging.getLogger")
    def test_create_logger(self, mock_getLogger):
        logfactory.create_logger()
        mock_getLogger.assert_called_with(seismic.__name__)

    @mock.patch("seismic.logfactory.logging.StreamHandler")
    def test_set_consoleHandler(self, mock_StreamHandler):
        logfactory.set_consoleHandler(self.logger, "INFO", "testconsole")
        mock_StreamHandler.assert_called_once()
        handler = mock_StreamHandler.return_value
        handler.set_name.assert_called_with("testconsole")
        handler.setLevel.assert_called_with("INFO")
        handler.setFormatter.assert_called_with(logfactory.cformatter)
        self.logger.addHandler.assert_called_with(handler)

    @mock.patch("seismic.logfactory.logging.FileHandler")
    def test_set_fileHandler(self, mock_FileHandler):
        filename = "filehandler_test.log"
        handlername = "testfile"
        logfactory.set_fileHandler(self.logger, filename, "INFO",
                                   handlername)
        mock_FileHandler.assert_called_with(filename)
        handler = mock_FileHandler.return_value
        handler.set_name.assert_called_with(handlername)
        handler.setLevel.assert_called_with("INFO")
        handler.setFormatter.assert_called_with(logfactory.cformatter)
        self.logger.addHandler.assert_called_with(handler)

    def test_get_handlers_by_name(self):
        handlers = logfactory.get_handlers_by_name(self.logger)
        self.assertIsInstance(handlers, dict, "handlers not a dict")
        self.assertEqual(len(handlers), 2,
                         "number of handlers not equal to 2")
        self.assertTrue(all([isinstance(h, list) for h in handlers.values()]),
                        "handler values not a list")
        self.assertTrue([len(h) for h in handlers.values()] == [1, 2],
                        "handler values not of length 1")

    def test_get_duplicate_handlers(self):
        handlers = logfactory.get_duplicate_handlers(self.logger)
        self.assertIsInstance(handlers, dict, "handlers not a dict")
        self.assertEqual(len(handlers), 1,
                         "number of duplicate handlers not equal to 1")
        self.assertIn("handler2", handlers.keys(),
                      "name handler2 not found in handlers")
        self.assertNotIn("handler1", handlers.keys(),
                         "name handler1 found in handlers")
        self.assertIsInstance(handlers["handler2"], list,
                              "handler not a list")
        self.assertEqual(len(handlers["handler2"]), 2,
                         "handler list not of length 2")

    def test_remove_duplicate_handlers(self):
        logfactory.remove_duplicate_handlers(self.logger)
        self.logger.removeHandler.assert_called_with(self.logger.handlers[2])
        self.logger.removeHandler.assert_called_once()


class TestLoggingMPIBaseClass(TestCase):

    classname = "LoggingMPIBaseClass"
    exec_time = datetime(2021, 1, 1, 0, 0, 0)
    exec_timestr = exec_time.strftime(logfactory.LOG_TSTRFMT)

    def setUp(self) -> None:
        return super().setUp()

    @mock.patch("seismic.logfactory.os")
    @mock.patch("seismic.logfactory.datetime")
    def test_set_logfilename(self, mock_time, mock_os):
        mock_time.now.return_value = self.exec_time
        mock_comm = mock.create_autospec(logfactory.MPI.COMM_WORLD)
        mock_comm.bcast.return_value = self.exec_timestr

        c = logfactory.LoggingMPIBaseClass()
        c.comm = mock_comm

        c.rank = 0
        c._set_filename()
        mock_time.now.assert_called_once()
        mock_os.path.join.assert_called_with(
            logfactory.LOGDIR, logfactory.FILENAME_FMT.format(
                classname=self.classname,
                rank=c.rank,
                exectimestr=self.exec_timestr))

        mock_time.reset_mock()
        c.rank = 1
        c._set_filename()
        mock_comm.bcast.assert_called_with(None, root=0)
        mock_time.now.assert_not_called()
        mock_os.path.join.assert_called_with(
            logfactory.LOGDIR, logfactory.FILENAME_FMT.format(
                classname=self.classname,
                rank=c.rank,
                exectimestr=self.exec_timestr))

    @mock.patch("seismic.logfactory.os")
    def test_mk_logdir(self, mock_os):
        logdir = "testlogdir"
        mock_comm = mock.create_autospec(logfactory.MPI.COMM_WORLD)

        c = logfactory.LoggingMPIBaseClass()
        c.comm = mock_comm

        c.rank = 0
        c._mk_logdir(logdir)
        mock_os.makedirs.assert_called_with(logdir, exist_ok=True)

        mock_os.reset_mock()
        c.rank = 1
        c._mk_logdir(logdir)
        mock_os.makedirs.assert_not_called()

    @mock.patch("seismic.logfactory.get_handlers_by_name")
    def test_default_handlers_set(self, mock_handlers):
        mock_logger = mock.create_autospec(logging.Logger)
        mock_logger.parent = mock.create_autospec(logging.Logger)
        c = logfactory.LoggingMPIBaseClass()
        c.logger = mock_logger
        handlernames = logfactory.DEFAULT_HANDLERNAMES

        # No handlers
        mock_logger.parent.hasHandlers.return_value = False
        self.assertFalse(c._default_handlers_set)
        mock_logger.parent.hasHandlers.assert_called_once()
        mock_handlers.assert_not_called()

        # Handlers present, with default names
        mock_logger.reset_mock()
        mock_handlers.reset_mock()
        mock_logger.parent.hasHandlers.return_value = True
        mock_handlers.return_value = {hn: [] for hn in handlernames}
        self.assertTrue(c._default_handlers_set)

        # Handlers present, with one default name missing
        mock_logger.reset_mock()
        mock_handlers.reset_mock()
        mock_logger.parent.hasHandlers.return_value = True
        mock_handlers.return_value = {hn: [] for hn in handlernames}
        mock_handlers.return_value.pop(handlernames[0])
        mock_handlers.return_value["extra"] = []
        self.assertFalse(c._default_handlers_set)

        # Handlers present, with only one default name
        mock_logger.reset_mock()
        mock_handlers.reset_mock()
        mock_logger.parent.hasHandlers.return_value = True
        mock_handlers.return_value = {hn: [] for hn in handlernames}
        mock_handlers.return_value.pop(handlernames[0])
        self.assertFalse(c._default_handlers_set)

    @mock.patch("seismic.logfactory.logging.getLogger")
    def test__set_logger(self, mock_getLogger):
        mock_logger = mock.create_autospec(logging.Logger)
        mock_logger.name = "mocking.logger"
        mock_logger.parent = mock.create_autospec(logging.Logger)
        mock_logger.parent.name = "mocking"
        mock_getLogger.return_value = mock_logger
        loglevel = "INFO"

        c = logfactory.LoggingMPIBaseClass()
        c._set_logger(loglevel)
        mock_getLogger.assert_called_with(".".join(
            [c.__module__, "", c.__class__.__name__
             + logfactory.RANK_STRFMT.format(rank=0)]))
        mock_logger.parent.setLevel.assert_called_once_with(loglevel.upper())

    @mock.patch("seismic.logfactory.set_consoleHandler")
    @mock.patch("seismic.logfactory.set_fileHandler")
    @mock.patch("seismic.logfactory.remove_duplicate_handlers")
    @mock.patch.object(logfactory.LoggingMPIBaseClass,
             "_default_handlers_set")
    def test_set_check_default_handlers(self, mock_handlers_set, mock_remove,
                                        mock_fh, mock_ch):
        mock_logger = mock.create_autospec(logging.Logger)
        mock_logger.parent = mock.create_autospec(logging.Logger)
        mock_logger.parent.name = "mocking"
        logfilename = "testlogfilename.log"
        loglevel = "INFO"
        mock_logger.parent.getEffectiveLevel.return_value = loglevel

        c = logfactory.LoggingMPIBaseClass()
        c.logger = mock_logger
        c.logfilename = logfilename

        # Default handlers already set
        c._default_handlers_set = True
        c._set_check_default_handlers()
        mock_fh.assert_not_called()
        mock_ch.assert_not_called()
        mock_remove.assert_called_once()

        # Default handlers not set
        for f in [mock_fh, mock_ch, mock_remove, mock_remove]:
            f.reset_mock()
        c._default_handlers_set = False
        c._set_check_default_handlers()
        mock_fh.assert_called_once_with(mock_logger.parent, logfilename,
                                        loglevel, logfactory.HANDLERNAME_FILE)
        mock_ch.assert_called_once_with(mock_logger.parent, loglevel,
                                        logfactory.HANDLERNAME_CONSOLE)
        mock_remove.assert_called_once()

    @mock.patch.object(logfactory.LoggingMPIBaseClass, "_set_logger")
    @mock.patch.object(logfactory.LoggingMPIBaseClass, "_set_filename")
    @mock.patch.object(logfactory.LoggingMPIBaseClass, "_mk_logdir")
    @mock.patch.object(logfactory.LoggingMPIBaseClass,
             "_set_check_default_handlers")
    def test_set_logger(self, mock_check, mock_mk, mock_fn, mock_sl):
        loglevel = "DEBUG"
        logdir = "testlogdir"
        filename_fmt = "testfilename"
        c = logfactory.LoggingMPIBaseClass()
        c.logger = mock.create_autospec(logging.Logger)
        c.set_logger(loglevel, logdir, filename_fmt)
        mock_sl.assert_called_once_with(loglevel)
        mock_fn.assert_called_once_with(logdir, filename_fmt)
        mock_mk.assert_called_once_with(logdir)
        mock_check.assert_called_once()


if __name__ == "__main__":
    main()
