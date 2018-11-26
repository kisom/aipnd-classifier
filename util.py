"""
util.py contains various utilities.
"""


class Log:
    """
    Log defines a logging function. It is motivated by the mixture of
    Jupyter notebooks and commandline utils; logging in Jupyter goes
    to the console, while logging in the command line utilities might
    go to a file.
    """

    def __init__(self, pylogger=None):
        """
        Create a new logger. If pylogger is None, a print function is used.
        Otherwise, it should be a python logging instance.
        """

        if not pylogger:
            self._info = lambda x: print("INFO: " + x)
            self._warn = lambda x: print("WARN: " + x)
            self._error = lambda x: print("ERR: " + x)
        else:
            self._info = pylogger.info
            self._warn = pylogger.warning
            self._error = pylogger.error

    def info(self, message):
        """info writes an informational message."""
        return self._info(message)

    def warn(self, message):
        """warn writes a warning message."""
        return self._warn(message)

    def error(self, message):
        """error writes an error message."""


_LOGGER = Log()


def set_logger(pylogger):
    """set_logger replaces the global logger."""
    global _LOGGER
    _LOGGER = Log(pylogger)


def get_logger():
    """get_logger returns the global logger."""
    return _LOGGER
