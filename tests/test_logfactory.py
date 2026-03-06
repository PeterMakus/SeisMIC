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
        self.logger.removeHandler.assert_called_with(self.logger.handlers[1])
        self.logger.removeHandler.assert_called_once()


class TestLoggingMPIBaseClass(TestCase):

    classname = "LoggingMPIBaseClass"
    exec_time = datetime(2021, 1, 1, 0, 0, 0)
    exec_timestr = exec_time.strftime(logfactory.LOG_TSTRFMT)

    def setUp(self) -> None:
        return super().setUp()

    def test_init(self):
        """Test that logger name is set correctly."""
        c = logfactory.LoggingMPIBaseClass()
        self.assertEqual(
            c.logger.name,
            ".".join([
                      logfactory.__name__, "", self.classname + "000"]
                     ))

    @mock.patch("seismic.logfactory.datetime")
    def test_set_logfilename(self, mock_time):
        logdir = "testlogdir"
        filename = (f"{logdir}/{self.classname}" + "-r{:03d}_" +
                    f"{self.exec_timestr}.log")
        mock_time.now.return_value = self.exec_time
        mock_comm = mock.create_autospec(logfactory.MPI.COMM_WORLD)
        mock_comm.bcast.return_value = self.exec_timestr

        c = logfactory.LoggingMPIBaseClass()
        c.comm = mock_comm

        # No logdir given --> no filename set
        c.rank = 0
        c._set_filename()
        mock_time.now.assert_not_called()
        self.assertEqual(
            c.logfilename, None)

        # Logdir given on rank 0--> filename set
        c.rank = 0
        c._set_filename(logdir=logdir)
        mock_comm.bcast.assert_called_with(self.exec_timestr, root=0)
        mock_time.now.assert_called_once()
        self.assertEqual(
            c.logfilename,
            filename.format(c.rank))

        # Logdir given on rank 1 --> time received from rank 0, filename set
        mock_time.reset_mock()
        c.rank = 1
        c._set_filename(logdir=logdir)
        mock_comm.bcast.assert_called_with(None, root=0)
        mock_time.now.assert_not_called()
        self.assertEqual(
            c.logfilename,
            filename.format(c.rank))

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

    @mock.patch("seismic.logfactory.set_fileHandler")
    @mock.patch("seismic.logfactory.set_consoleHandler")
    @mock.patch("seismic.logfactory.remove_duplicate_handlers")
    def test_set_handlers_incl_filehandler(self, mock_remove, mock_ch, mock_fh):
        logdir = "testlogdir"
        filename = f"{logdir}/{self.classname}-r000_{self.exec_timestr}.log"

        c = logfactory.LoggingMPIBaseClass()
        c.logfilename = filename
        c.logger = mock.create_autospec(logging.Logger)
        c.logger.parent = mock.create_autospec(logging.Logger)
        c.logger.parent.name = "mocking"
        c._set_handlers()
        mock_fh.assert_called_once_with(
            c.logger.parent, filename, logfactory.HANDLER_LOGLVL, filename)
        mock_ch.assert_called_once_with(
            c.logger.parent, logfactory.HANDLER_LOGLVL,
            logfactory.HANDLERNAME_CONSOLE)
        mock_remove.assert_called_once_with(
            c.logger.parent)

    @mock.patch("seismic.logfactory.set_fileHandler")
    @mock.patch("seismic.logfactory.set_consoleHandler")
    @mock.patch("seismic.logfactory.remove_duplicate_handlers")
    def test_set_handlers_no_filehandler(self, mock_remove, mock_ch, mock_fh):
        c = logfactory.LoggingMPIBaseClass()
        c.logfilename = None
        c.logger = mock.create_autospec(logging.Logger)
        c.logger.parent = mock.create_autospec(logging.Logger)
        c.logger.parent.name = "mocking"
        c._set_handlers()
        mock_fh.assert_not_called()
        mock_ch.assert_called_once_with(
            c.logger.parent, logfactory.HANDLER_LOGLVL,
            logfactory.HANDLERNAME_CONSOLE)
        mock_remove.assert_called_once_with(
            c.logger.parent)

    def test_set_logger(self):
        """
        Integration test for set_logger. Test that the filename, loglevel
        and handlers are set correctly.
        """
        logdir = "testlogdir"
        filename_fmt = "testfilename"
        loglevel = "DEBUG"

        c = logfactory.LoggingMPIBaseClass()
        c._set_filename = mock.MagicMock()
        c._mk_logdir = mock.MagicMock()
        c._set_handlers = mock.MagicMock()

        c.set_logger(loglevel, logdir, filename_fmt)
        c._set_filename.assert_called_once_with(logdir, filename_fmt)
        c._mk_logdir.assert_called_once_with(logdir)
        c._set_handlers.assert_called_once()


if __name__ == "__main__":
    main()
