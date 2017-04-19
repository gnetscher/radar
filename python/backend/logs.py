from .settings  import settings
import logging.handlers
import os
import sys
import traceback


# Indicates what level of log to display. Could be e.g. "DEBUG" or "WARNING"
DEFAULT_LOG_LEVEL = 'DEBUG' if settings.DEBUG else 'INFO'
settings.optional('LOG_LEVEL', DEFAULT_LOG_LEVEL)
settings.optional('LOG_GROUP', None)
LOG_LEVEL         = getattr(logging, settings.LOG_LEVEL, DEFAULT_LOG_LEVEL)


class NSLogFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(NSLogFormatter, self).__init__(*args, **kwargs)
        self.__prefix = ' ' * 25
        self.__limit  = '*' * 100

    def formatException(self, exc_info):
        trace    = ''.join(traceback.format_exception(*exc_info)).split('\n')
        res      = self.__prefix + self.__limit
        for line in trace:
            res += '\n%s%s' % (self.__prefix, line)
        res     += self.__limit
        return res


class NSRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super(NSRotatingFileHandler, self).__init__(*args, **kwargs)
        self.update_permissions()

    def doRollover(self):
        super(NSRotatingFileHandler, self).doRollover()
        self.update_permissions()

    def update_permissions(self):
        if settings.LOG_GROUP is not None:
            os.system('chown :%s %s' % (settings.LOG_GROUP, self.baseFilename))


__log_formatter__ = NSLogFormatter('[%(levelname).1s/%(asctime)s]: %(message)s', '%Y-%m-%d %H:%M:%S')


def rotating_log_handler(filename, backup_count=1):
    handler = NSRotatingFileHandler(filename, when="midnight", backupCount=backup_count)
    handler.setFormatter(__log_formatter__)
    return handler


def rebind_logger(module, log_level=logging.WARNING, backup_count=1):
    logger           = logging.getLogger(module)
    logger.propagate = False
    logger.setLevel(log_level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(rotating_log_handler(os.path.join(settings.LOG_DIR, '%s.txt' % module), backup_count))
    return logger


NSLog = rebind_logger('safely_you', log_level=LOG_LEVEL, backup_count=10)


# In debug mod, also output to the console
if settings.DEBUG or (not settings.RUN_SERVER and not settings.RUN_LOCAL_UNIT):
    NSLog._debug_handler = logging.StreamHandler(sys.stdout)
    NSLog._debug_handler.setLevel(logging.DEBUG)
    NSLog.addHandler(NSLog._debug_handler)
else:
    NSLog._debug_handler = None


# Don't override standard outputs if not running server
if settings.RUN_SERVER:
    class NSStreamWrapper(object):
        def __init__(self, logger, log_level=logging.INFO):
            self.logger    = logger
            self.log_level = log_level
            self.linebuf   = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

    sys.stdout = NSStreamWrapper(NSLog, logging.INFO)
    sys.stderr = NSStreamWrapper(NSLog, logging.ERROR)


# Make sure that we log all the exceptions, even if they are not caught
def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        NSLog.error('Uncaught exception!', exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_uncaught_exception







