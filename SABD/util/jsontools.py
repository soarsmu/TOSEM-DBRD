import json
from collections import OrderedDict
from logging import Formatter
import re

RESERVED_ATTRS = (
    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
    'funcName', 'levelname', 'levelno', 'lineno', 'module',
    'msecs', 'message', 'msg', 'name', 'pathname', 'process',
    'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName')


class JsonLogFormatter(Formatter):
    """
    Format messages to be included in JSON strings. It just pass the msg attribute of the LogRecord through
    json.dumps(msg).

    Code based on: https://github.com/madzak/python-json-logger/blob/master/src/pythonjsonlogger/jsonlogger.py
    """

    def __init__(self, fmt="%(asctime)s %(levelname)-4s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'):
        super(JsonLogFormatter, self).__init__(fmt, datefmt)

        self.requiredFields = self.parse()
        self.skipFields = dict(
            zip(self.requiredFields,
            self.requiredFields))
        self.skipFields.update(dict(zip(RESERVED_ATTRS, RESERVED_ATTRS)))

    def parse(self):
        """
        Parses format string looking for substitutions
        This method is responsible for returning a list of fields (as strings)
        to include in all log messages.
        """
        standard_formatters = re.compile(r'\((.+?)\)', re.IGNORECASE)

        return standard_formatters.findall(self._fmt)

    def format(self, record):
        """Formats a log record and serializes to json"""

        if isinstance(record.msg, dict):
            message_dict = record.msg
            record.message = None
        else:
            message_dict = {}
            record.message = record.getMessage()

        # only format time if needed
        if "asctime" in self.requiredFields:
            record.asctime = self.formatTime(record, self.datefmt)

        log_record = OrderedDict()

        # Record the required fields
        for field in self.requiredFields:
            log_record[field] = record.__dict__.get(field)

        # Update log_record using dictionary sent in the message
        log_record.update(message_dict)

        # Reading keys and values sending in the extra argument
        for key, value in record.__dict__.items():
            # this allows to have numeric keys
            if (key not in self.skipFields
                    and not (hasattr(key, "startswith") and key.startswith('_'))):
                log_record[key] = value

        return json.dumps(log_record)

