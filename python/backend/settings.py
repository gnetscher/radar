from .helpers import INDEX_TO_CHAR, CHAR_TO_INDEX, stable_hash, CHARS, NSVersion
import os
import json
import tempfile
import sys


class NSSettings(object):
    class KeyValuePair(object):
        def __init__(self, key, value, required, default_value):
            self.key           = key
            self.value         = value
            self.required      = required
            self.default_value = default_value

        def __str__(self):
            res = '%s: %s' % (self.key, self.value)
            if self.required:
                res += ' [REQUIRED]'
            else:
                res += ' [DEFAULT=%s]' % self.default_value
            return res

    def __init__(self):
        class SettingsHandler(object):
            def __init__(self, settings):
                self.settings          = settings
                self.global_values     = {}
                self.server_values     = {}
                self.local_unit_values = {}
                self.key_to_map        = {}

            def load(self, map, path):
                map['_path'] = path
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        for key, value in json.load(f).iteritems():
                            if isinstance(value, list):
                                self.key_to_map[key] = self
                                if NSSettings.is_path(key):
                                    value            = [os.path.abspath(v) for v in value]
                                map[key]             = value
                                value                = list(set(value + getattr(self.settings, key, [])))
                            else:
                                if key in self.key_to_map:
                                    print >> sys.stderr, 'Settings "%s" already defined!' % key
                                    exit(-1)
                                self.key_to_map[key] = map

                                if NSSettings.should_obfuscate(key):
                                    value    = self.settings.unobfuscate(key, value)
                                    map[key] = self.settings.obfuscate  (key, value)
                                else:
                                    map[key] = value

                                if NSSettings.is_path(key):
                                    value    = os.path.abspath(value)

                            super(NSSettings, self.settings).__setattr__(key, value)

            def save(self, map):
                path = map.pop('_path')
                if os.path.exists(path):
                    with open(path, 'w') as f:
                        json.dump(map, f, indent=4)
                map['_path'] = path

        self.__dict__['_NSSettings__handler'] = SettingsHandler(self)

        # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
        self.BASE_DIR                     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.CONF_DIR                     = os.path.join(self.BASE_DIR, 'conf')
        self.LOCAL_UNIT_CONF_DIR          = os.path.join(self.CONF_DIR, 'local_unit')
        self.SERVER_CONF_DIR              = os.path.join(self.CONF_DIR, 'server')
        self.LOG_DIR                      = os.path.join(self.BASE_DIR, 'logs')
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        self.TMP_DIR                      = os.path.join(tempfile.gettempdir(), 'NSBackEnd')
        self.LOCAL_DATA_PATH              = os.path.join(self.TMP_DIR , 'data')
        self.HOME_DIR                     = os.path.expanduser("~")
        self.UNIT_DB_PATH                 = os.path.join(self.CONF_DIR, 'db.sqlite3')
        self.DEBUG                        = bool(os.environ.get('NS_DEBUG', False))
        self.RUN_SERVER                   = False
        self.RUN_LOCAL_UNIT               = False
        self.MIGRATE                      = False

        # Load the settings
        self.__key = 'NestSenseLegacy!'
        self.__handler.load(self.__handler.global_values    , os.path.join(self.CONF_DIR           , 'settings.json'))
        self.__handler.load(self.__handler.server_values    , os.path.join(self.SERVER_CONF_DIR    , 'settings.json'))
        self.__handler.load(self.__handler.local_unit_values, os.path.join(self.LOCAL_UNIT_CONF_DIR, 'settings.json'))

        if len(self.__handler.server_values) > 0 and 'SECRET_KEY' not in self.__handler.server_values:
            import random
            key = ''.join([random.SystemRandom().choice(CHARS[:-1]) for i in range(64)])
            self.__handler.server_values['SECRET_KEY'] = self.obfuscate('SECRET_KEY', key)
            setattr(self, 'SECRET_KEY', key)

        self.VERSION    = NSVersion.parse(getattr(self, 'VERSION', '1.0.0'))

        self.__required = set()
        self.__optional = {}

        self.push()

    @property
    def RUN_SCHEDULED_TASKS(self):
        return self.RUN_SERVER or self.RUN_LOCAL_UNIT

    @classmethod
    def is_path(cls, key, all=False):
        if '_PATH' in key or '_DIR' in key or '_LOCATION' in key:
            return all or key != 'FTP_PATH'
        return False

    @classmethod
    def should_obfuscate(cls, key):
        return 'PASSWORD' in key or 'SECRET' in key

    def obfuscate(self, key, value):
        res    = ''
        value += '@ob'
        i      = stable_hash((self.__key, key))
        for c in value:
            i    = (CHAR_TO_INDEX[c] + i) % len(CHAR_TO_INDEX)
            res += INDEX_TO_CHAR[i]
        return res

    def unobfuscate(self, key, value):
        if len(value) < 3:
            return value

        res = ''
        i   = stable_hash((self.__key, key)) % len(CHAR_TO_INDEX)
        for c in value:
            j    = (len(CHAR_TO_INDEX) + CHAR_TO_INDEX[c] - i) % len(CHAR_TO_INDEX)
            i    = (j + i) % len(CHAR_TO_INDEX)
            res += INDEX_TO_CHAR[j]

        return res[:-3] if res[-3:] == '@ob' else value

    def required(self, key):
        if not hasattr(self, key):
            from .logs import NSLog
            NSLog.critical('Settings field "%s" is required but not set!', key)
            exit(-1)
        self.__required.add(key)

    def optional(self, key, default_value=None):
        if not hasattr(self, key):
            super(NSSettings, self).__setattr__(key,
                                                os.path.abspath(default_value) if self.is_path(key) else default_value)
        self.__optional[key] = default_value

    def save(self, key, value):
        if key not in self.__handler.key_to_map:
            self.__handler.key_to_map[key] = self.__handler.global_values
        setattr(self, key, value)

    def push(self):
        self.__handler.save(self.__handler.global_values)
        self.__handler.save(self.__handler.server_values)
        self.__handler.save(self.__handler.local_unit_values)

    def entry(self, key):
        return NSSettings.KeyValuePair(key, getattr(self, key), key in self.__required, self.__optional.get(key))

    def reset(self, key):
        if key in self.__required:
            raise RuntimeError('Cannot reset required value!')
        map = self.__handler.key_to_map.get(key)
        if map is not None:
            del map[key]
            del self.__handler.key_to_map[key]
            self.__handler.save(map)
        setattr(self, key, self.__optional[key])

    def __iter__(self):
        for key in self.__required:
            yield NSSettings.KeyValuePair(key, getattr(self, key), True, None)
        for key, default_value in self.__optional.iteritems():
            if key not in self.__required:
                yield NSSettings.KeyValuePair(key, getattr(self, key), False, default_value)

    def __setattr__(self, key, value):
        if key == 'DEBUG':
            if hasattr(self, key):
                from .logs import NSLog, logging

                if value:
                    if NSLog._debug_handler is None:
                        NSLog._debug_handler = logging.StreamHandler(sys.stdout)
                        NSLog._debug_handler.setLevel(logging.DEBUG)
                        NSLog.addHandler(NSLog._debug_handler)
                elif NSLog._debug_handler is not None:
                    NSLog.removeHandler(NSLog._debug_handler)
                    NSLog._debug_handler = None
        else:
            if key.endswith('VERSION') and isinstance(value, basestring):
                value = NSVersion.parse(value)
            elif self.is_path(key):
                value = os.path.abspath(value)

            if key == 'SENSOR_CONFIGS':
                map = self.__handler.local_unit_values
            else:
                map = self.__handler.key_to_map.get(key)

            if map is not None:
                map_value     = self.obfuscate(key, value) if self.should_obfuscate(key) else value
                if isinstance(map_value, NSVersion):
                    map_value = str(value)
                map[key]      = map_value

                self.__handler.save(map)
        super(NSSettings, self).__setattr__(key, value)

settings = NSSettings()

