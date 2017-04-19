from .settings import settings
from .logs     import NSLog
import json
import os


class NSEnumMeta(type):
    __enums__ = {}
    __ready__ = True

    class Meta:
        def __init__(self):
            self.enum = None

    def __new__(mcs, name, bases, args):
        enum_cls   = super(NSEnumMeta, mcs).__new__(mcs, name, bases, args)
        enum_files = [os.path.join(settings.CONF_DIR           , 'enums.json'),
                      os.path.join(settings.LOCAL_UNIT_CONF_DIR, 'enums.json'),
                      os.path.join(settings.SERVER_CONF_DIR    , 'enums.json')]

        for enum_file in enum_files:
            if os.path.exists(enum_file):
                with open(enum_file, 'r') as f:
                    for enum_name, values_list in json.load(f).iteritems():
                        assert 'NS' + enum_name not in globals(), \
                            'An object with the name "NS%s" already exists in global scope!' % enum_name

                        description = values_list.pop('__desc__', '')
                        globals()['NS' + enum_name] = mcs.create(enum_cls, enum_name, values_list, description)

        return enum_cls

    def __call__(cls, *args, **kwargs):
        if cls.__ready__:
            return cls.__enums__[args[0]]
        return super(NSEnumMeta, cls).__call__(*args, **kwargs)

    def create(cls, enum_name, values_list, description=''):
        cls.__ready__ = False
        try:
            assert enum_name not in cls.__enums__, 'An enumeration with the name "%s" already exists!' % enum_name

            meta        = cls.Meta()
            values      = {}
            bits        = set()

            class Value(object):
                _meta = meta

                def __init__(self, name, value, value_descr):
                    self.__dict__['_Value__name']        = name
                    self.__dict__['_Value__value']       = value
                    self.__dict__['_Value__description'] = value_descr

                @property
                def name(self):
                    return self.__name

                @property
                def value(self):
                    return self.__value

                @property
                def description(self):
                    return self.__description

                @property
                def enum(self):
                    return self._meta.enum

                def gt_mask(self):
                    return self._meta.enum.gt_mask(self)

                def gte_mask(self):
                    return self._meta.enum.gte_mask(self)

                def lt_mask(self):
                    return self._meta.enum.lt_mask(self)

                def lte_mask(self):
                    return self._meta.enum.lte_mask(self)

                def __setattr__(self, name, value):
                    raise Exception('Enumeration value does not allow modification!')

                def __delattr__(self, name):
                    raise Exception('Enumeration value does not allow modification!')

                def __or__(self, other):
                    return self.enum.Flags(self) | other

                def __and__(self, other):
                    return self.enum.Flags(self) & other

                def __add__(self, other):
                    return self.enum.Flags(self) + other

                def __sub__(self, other):
                    return self.enum.Flags(self) - other

                def __cmp__(self, other):
                    return cmp(self.enum.Flags(self), other)

                def __int__(self):
                    return self.value

                def __hash__(self):
                    return hash(self.value)

                def __str__(self):
                    return self.name

                def __repr__(self):
                    return '<NSEnumValue@%s.%s[%d]>' % (self.enum, self.name, self.value)

            for value_name, bit in values_list.iteritems():
                if isinstance(bit, dict):
                    value_descr = bit.get('__desc__', bit.get('description', ''))
                    bit         = bit['bit']
                else:
                    value_descr = ''

                assert value_name.upper() not in values, \
                    'A value for enumeration "%s" with name "%s" already exists!' % (enum_name, value_name)
                assert bit not in bits, \
                    'A value for enumeration "%s" with value %d already exists!' % (enum_name, 1 << bit)

                values[value_name.upper()] = Value(value_name, 1 << bit, value_descr)
                bits.add(bit)

            enum = meta.enum = cls(enum_name, Value, values, description)
            setattr(NSEnumMeta, enum_name, cls.__enum_property__(enum))

            NSLog.debug('Enumeration "%s" registered.', enum_name)

            return enum
        finally:
            cls.__ready__ = True

    @staticmethod
    def __enum_property__(value):
        return property(lambda cls: value)


class NSFlags(object):
    def __init__(self, value=0):
        self.__dict__['_NSFlags__value'] = self.__value__(value)

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        self.__value = self.__value__(v)

    @classmethod
    def __value__(cls, value):
        if not value:
            return 0
        if isinstance(value, int):
            assert (value & cls.Mask) == value, 'Invalid flag value!'
            return value
        if isinstance(value, cls):
            return value.value
        if isinstance(value, cls.Enum):
            return value.value
        if isinstance(value, list) or isinstance(value, set):
            res = 0
            for v in value:
                res |= cls.__value__(v)
            return res

        assert isinstance(value, basestring), 'Invalid flag value type!'

        res = 0
        for v in value.split('|'):
            res |= cls.Enum[v].value
        return res

    def __or__(self, other):
        return self.__class__(self.__value | self.__value__(other))

    def __and__(self, other):
        return self.__class__(self.__value & self.__value__(other))

    def __nonzero__(self):
        return self.__value != 0

    def __add__(self, other):
        return self.__class__(self.__value | self.__value__(other))

    def __sub__(self, other):
        return self.__class__(self.__value & ~self.__value__(other))

    def __getattr__(self, key):
        return getattr(self.__class__, key)

    def __setattr__(self, key, value):
        raise Exception('Flags do not allow modification!')

    def __delattr__(self, key):
        raise Exception('Enumeration do not allow modification!')

    def __iter__(self):
        for value in self.Enum:
            if (self.__value & value.value) == value.value:
                yield value

    def __contains__(self, item):
        value = self.__value__(item)
        return (self.__value & value) == value

    def __cmp__(self, other):
        return self.__value - self.__value__(other)

    def __int__(self):
        return self.__value

    def __str__(self):
        return '|'.join([str(value) for value in self])

    def __repr__(self):
        return '<NSFlags@%s<%s>[%d]>' % (self.Enum, self, self.__value)


class NSEnum(object):
    __metaclass__ = NSEnumMeta

    def __init__(self, name, value_type, values, description):
        enum = self
        mask = 0
        for value in values.itervalues():
            mask |= value.value

        class FlagsMeta(type):
            @property
            def Enum(cls):
                return enum

            @property
            def Mask(cls):
                return mask

        class Flags(NSFlags):
            __metaclass__ = FlagsMeta

        for value in values.itervalues():
            setattr(FlagsMeta, value.name, self.__class__.__enum_property__(value))

        self.__dict__['_NSEnum__name']        = name
        self.__dict__['_NSEnum__value_type']  = value_type
        self.__dict__['_NSEnum__values']      = values
        self.__dict__['_NSEnum__description'] = description
        self.__dict__['_NSEnum__pattern']     = '|'.join(values.iterkeys())
        self.__dict__['Flags']                = Flags

    @property
    def name(self):
        return self.__name

    @property
    def values(self):
        return self.__values.values()

    @property
    def description(self):
        return self.__description

    @property
    def pattern(self):
        return self.__pattern

    def itervalues(self):
        return self.__values.itervalues()

    def gt_mask(self, value):
        try:
            return self.__dict__['_NSEnum__gt']
        except KeyError:
            mask = self.Flags()
            for v in self.__values.itervalues():
                if v > value:
                    mask |= v
            self.__dict__['_NSEnum__gt'] = mask
            return mask

    def gte_mask(self, value):
        return self.gt_mask(value) | value

    def lt_mask(self, value):
        try:
            return self.__dict__['_NSEnum__lt']
        except KeyError:
            mask = self.Flags()
            for v in self.__values.itervalues():
                if v < value:
                    mask |= v
            self.__dict__['_NSEnum__lt'] = mask
            return mask

    def lte_mask(self, value):
        return self.lt_mask(value) | value

    def __instancecheck__(self, instance):
        return isinstance(instance, self.__value_type)

    def __getattr__(self, name):
        try:
            return self.__values[name.upper()]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name, value):
        raise Exception('Enumeration does not allow modification!')

    def __delattr__(self, name):
        raise Exception('Enumeration does not allow modification!')

    def __call__(self, value):
        if isinstance(value, self.__value_type):
            return value

        try:
            if isinstance(value, basestring):
                return self.__values[value.upper()]
            if isinstance(value, int):
                for v in self.__values.itervalues():
                    if v.value == value:
                        return v
        except KeyError:
            pass
        raise Exception('Invalid enumeration value "%s"!' % value)

    def __getitem__(self, key):
        try:
            return self.__call__(key)
        except:
            raise KeyError(key)

    def __setitem__(self, index, value):
        raise Exception('Enumeration does not allow modification!')

    def __delitem__(self, index):
        raise Exception('Enumeration does not allow modification!')

    def __len__(self):
        return len(self.__values)

    def __iter__(self):
        for value in self.__values.itervalues():
            yield value

    def __contains__(self, value):
        try:
            self.__call__(value)
            return True
        except:
            return False

    def __str__(self):
        return self.__name

    def __repr__(self):
        return '<NSEnum@%s>' % self.__name


