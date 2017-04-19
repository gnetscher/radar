from Crypto.Cipher    import AES
from Crypto.Cipher    import PKCS1_OAEP
from Crypto.PublicKey import RSA
from dateutil         import parser
import json    as    _json
import calendar
import commands
import datetime
import os
import sys
import copy
import re
import uuid
import pytz
import collections
import Crypto
import tempfile


REGEX_TYPE          = type(re.compile('foo'))
EPOCH_AWARE         = datetime.datetime.fromtimestamp(0, pytz.utc)
NAME_PATTERN        = r'[\w@][\w.@+-]*'
PRETTY_NAME_PATTERN = r"[\w@][\w\s.@+-]*"


def deepcopy(obj):
    if isinstance(obj, dict):
        return {key: deepcopy(value) for key, value in obj.iteritems()}
    if isinstance(obj, list):
        return [deepcopy(value) for value in obj]
    if isinstance(obj, REGEX_TYPE):
        return obj

    try:
        return copy.deepcopy(obj)
    except TypeError:
        from .logs import NSLog
        NSLog.warning('Unsupported copy of object of type "%s"!', type(obj))
        return obj


def make_pretty(tag):
    """ Returns the given string where all the _ have been replaced by spaces. """
    return ' '.join([word for word in tag.split('_')]).capitalize()


def make_title(tag):
    """ Returns the given string where all the _ have been replaced by spaces and all the words are capitalized. """
    return ' '.join([word.title() for word in tag.split('_')])


def make_pretty_tag(tag):
    """ Capitalize the tag and remove the underscores. """
    return ''.join([word.title() for word in tag.split('_')])


def make_url(hostname, suffix):
    from .settings import settings
    if getattr(settings, 'UNSECURED_CONNECTION', False):
        return 'http://%s:8000/%s' % (hostname, suffix)
    return 'https://%s/%s' % (hostname, suffix)


def parse_date(timestr):
    try:
        return parser.parse(timestr).astimezone(pytz.utc)
    except:
        raise ValueError("Failed to parse date '%s'!" % timestr)


def struct(typename, fields, base=object):
    all_fields       = list(base._fields) + fields if hasattr(base, '_fields') else fields
    class_definition = '''\
class {typename}({base}):
    _fields = set({fields!r})

    def __init__(self'''.format(typename=typename, base=base.__name__, fields=all_fields)

    for field in fields:
        class_definition += ', %s=None' % field
    if base is not None:
        class_definition += ', **kwargs'
    class_definition     += '):\n'
    if base is not None:
        class_definition += '        super(%s, self).__init__(**kwargs)\n' % typename
    for field in fields:
        class_definition += '        self.{field} = {field}\n'.format(field=field)
    class_definition     += '''\

    def replace(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def keys(self):
        return set(self._fields)

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def iterkeys(self):
        return iter(self._fields)

    def itervalues(self):
'''
    for field in all_fields:
        class_definition += '        yield self.%s\n' % field
    class_definition     += '    def iteritems(self):\n'
    for field in all_fields:
        class_definition += '        yield ("{field}", self.{field})\n'.format(field=field)
    class_definition     += '''\

    def __iter__(self):
        return self.iterkeys()

    def __getitem__(self, key):
        if key not in self._fields:
            raise KeyError(key)
        return getattr(self, key)

    def __repr__(self):
        return repr(dict(**self))
    '''

    namespace = dict(__name__=typename)
    if base is not None:
        namespace[base.__name__] = base
    try:
        exec class_definition in namespace
    except SyntaxError as e:
        raise SyntaxError(e.message + ':\n' + class_definition)
    result = namespace[typename]

    try:
        result.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result


CHARS = 'abcdefghijklmnopqrstuvwxyz' \
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
        '0123456789' \
        '!#$%&()*+,-./:;<=>?@[\]^_`{|}~ '
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARS)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.iteritems()}


def stable_hash(value, h=5381):
    if isinstance(value, (tuple, list, set)):
        for v in value:
            h = stable_hash(v, h)
    elif isinstance(value, basestring):
        for c in value:
            h = (((h << 5) + h) + CHAR_TO_INDEX[c]) % 19134702400093278081449423917L
    else:
        raise NotImplementedError()
    return h


NSVideoMetadata = struct('NSVideoMetadata', ['duration', 'width', 'height', 'fps', 'frame_count'])
NSVideoChunk    = struct('NSVideoChunk'   , ['path', 'start', 'end'], NSVideoMetadata)


def video_metadata(path):
    """
       Extracts metadata from the video file pointed by the given path. If the file is not valid, returns None.
       The returned object contains the following information:
       - duration   : A timedelta object representing the duration of the video
       - frame_count: Total number of frames of the video
       - fps
       - width
       - height
    """

    try:
        mediainfo = commands.getoutput('mediainfo --Inform="Video;%%Duration%%,%%Width%%,%%Height%%,'
                                       '%%FrameRate%%,%%FrameCount%%" "%s"' % path)
        info      = mediainfo.split(',')

        return NSVideoMetadata(duration    = datetime.timedelta(milliseconds=int(info[0])),
                               width       = int  (info[1]),
                               height      = int  (info[2]),
                               fps         = float(info[3]),
                               frame_count = int  (info[4]))
    except:
        from .logs import NSLog
        NSLog.exception('Failed to retrieve video "%s" metadata!', path)
    return None


def truncate_video(video_path, dst_dir, start, max_duration=60, width=None, height=None, codec='h264'):
    """ Truncates a video in shorter clips. """

    from .logs import NSLog

    # Get the video metadata
    filename      = os.path.basename(video_path)
    basename, ext = os.path.splitext(filename)
    metadata      = video_metadata(video_path)
    if metadata is None:
        raise RuntimeError('Cannot parse video metadata!')

    # Make sure that the destination folder exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Prepare video scaling
    if width is None:
        if height is None:
            scale_str = ''
        else:
            scale_str = '-vf "scale=-1:%d"' % height
    elif height is None:
        scale_str     = '-vf "scale=%d:-1"' % width
    else:
        scale_str     = '-vf "scale=%d:%d"' % (width, height)

    # Cut the video into small chunks and convert them to a standard mp4 format to be readable in a web admin
    fps          = metadata.fps
    end          = start + metadata.duration
    video_start  = 0
    duration     = datetime.timedelta(seconds=max_duration)
    min_duration = datetime.timedelta(seconds=max_duration / 4)
    index        = 0
    chunks       = []

    try:
        while start < end:
            # Check that the current chunk is not too long and that the next chunk will not be too small
            if end - (start + duration) < min_duration:
                duration = end - start
            video_delta  = duration.total_seconds()  # Includes microseconds
            chunk        = NSVideoChunk(path  = os.path.join(dst_dir, '%s.%d.mp4' % (basename, index)),
                                        start = start,
                                        end   = start + duration)
            chunks.append(chunk)

            # Remove existing file to avoid duplicates. Can sometime happen in case of a previous failure
            if os.path.exists(chunk.path):
                os.remove(chunk.path)

            NSLog.debug('Writing video chunk [%s -> %s: %3ds] from "%s" to "%s"...',
                        chunk.start, chunk.end, video_delta, filename, chunk.path)

            if video_start == 0:
                status, output = commands.getstatusoutput(
                    'ffmpeg -y '
                    '-loglevel error '
                    '-r {fps} '
                    '-i "{input}" {scale} '
                    '-vcodec {codec} '
                    '-t {duration} '
                    '-an "{output}"'
                        .format(fps      = fps,
                                input    = video_path,
                                scale    = scale_str,
                                codec    = codec,
                                duration = video_delta,
                                output   = chunk.path))
            else:
                status, output = commands.getstatusoutput(
                    'ffmpeg -y -loglevel error -r {fps} -ss {start} -i "{input}" '
                    '{scale} -vcodec {codec} -t {duration} -an "{output}"'
                        .format(fps      = fps,
                                start    = video_start,
                                input    = video_path,
                                scale    = scale_str,
                                codec    = codec,
                                duration = video_delta,
                                output   = chunk.path))

            if status != 0:
                error = 'ffmpeg failed:'
                for line in output.splitlines():
                    error += '   %s' % line
                raise RuntimeError(error)

            # Update chunk with real metadata info
            metadata     = video_metadata(chunk.path)
            chunk.replace(**metadata)
            chunk.end    = chunk.start + metadata.duration
            video_delta  = metadata.duration.total_seconds()

            if start + duration >= end:  # Prevent one frame videos
                break

            index       += 1
            start        = chunk.end
            video_start += video_delta
    except:
        for chunk in chunks:
            if os.path.exists(chunk.path):
                os.remove(chunk.path)
        raise

    return chunks


def concatenate_videos(files, dst, web=False, blur_amount=None):
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    list_file, list_path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(list_file, 'w') as f:
        for src_file in files:
            f.write("file '%s'\n" % src_file)

    tmp_dst        = dst + os.path.splitext(files[0])[1]
    cmd            = 'ffmpeg -y ' \
                     '-loglevel error ' \
                     '-f concat ' \
                     '-safe 0 ' \
                     '-i {file} ' \
                     '-c copy '
    if web:
        cmd       += '-vcodec libx264 ' \
                     '-vprofile main ' \
                     '-pix_fmt yuv420p ' \
                     '-preset slow ' \
                     '-b:v 400k ' \
                     '-maxrate 400k ' \
                     '-bufsize 800k '
    if blur_amount is not None:
        cmd       += '-vf "boxblur=%d" ' % blur_amount
    cmd           += '-an "{output}"'
    status, output = \
        commands.getstatusoutput(cmd.format(file=list_path, output=tmp_dst))

    os.remove(list_path)

    if status != 0:
        error = 'ffmpeg failed:'
        for line in output.splitlines():
            error += '   %s' % line
        raise RuntimeError(error)

    os.rename(tmp_dst, dst)


def concatenate_images(files, dst, fps=5):
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    tmp_dst        = dst + os.path.splitext(files[0])[1]
    status, output = \
        commands.getstatusoutput('ffmpeg -y '
                                 '-loglevel error '
                                 '-framerate {fps} '
                                 '-i "concat:{files}" '
                                 '-vcodec libx264 '
                                 '-pix_fmt yuv420p '
                                 '-an "{output}"'
                                 .format(files='|'.join(files), output=tmp_dst, fps=fps))

    if status != 0:
        error = 'ffmpeg failed:'
        for line in output.splitlines():
            error += '   %s' % line
        raise RuntimeError(error)

    os.rename(tmp_dst, dst)


def prepare_web_video(src, dst, blur_amount=None):
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    tmp_dst  = dst + '.mp4'
    cmd      = 'ffmpeg -y ' \
               '-loglevel error ' \
               '-i "{input}" ' \
               '-vcodec libx264 ' \
               '-vprofile main ' \
               '-pix_fmt yuv420p ' \
               '-preset slow ' \
               '-b:v 400k ' \
               '-maxrate 400k ' \
               '-bufsize 800k '
    if blur_amount is not None:
        cmd += '-vf "boxblur=%d" ' % blur_amount
    cmd     += '-an "{output}"'

    status, output = commands.getstatusoutput(cmd.format(input=src, output=tmp_dst))
    if status != 0:
        error = 'ffmpeg failed:'
        for line in output.splitlines():
            error += '   %s' % line
        raise RuntimeError(error)

    os.rename(tmp_dst, dst)


def encrypted_filename(filename):
    return filename + '.encrypt'


def decrypted_filename(filename):
    if filename.endswith('.encrypt'):
        return filename[:-8]
    return '%s.clear.%s' % os.path.splitext(filename)


def encrypt_file(src_path, dst_path, public_key):
    Crypto.Random.atfork()

    if isinstance(public_key, basestring):
        public_key = RSA.importKey(public_key)
    cipher = PKCS1_OAEP.new(public_key)

    # Generate the secret key and the symmetric cypher
    secret_key = os.urandom(48)
    sym_cipher = AES.new(secret_key[:-16], AES.MODE_CFB, secret_key[32:])

    # Encrypt the file using the symmetric cypher and store the encrypted secret key at the beginning
    encrypted_secret_key = cipher.encrypt(secret_key)
    with open(dst_path, 'wb') as dst:
        dst.write('%d\n' % len(encrypted_secret_key))
        dst.write(encrypted_secret_key)

        with open(src_path, 'rb') as src:
            dst.write(sym_cipher.encrypt(src.read()))

    return dst_path


def encrypt_chunks(chunks, public_key):
    if isinstance(public_key, basestring):
        public_key = RSA.importKey(public_key)

    for chunk in chunks:
        encrypted_path = encrypt_file(chunk.path, encrypted_filename(chunk.path), public_key)
        os.remove(chunk.path)
        chunk.path     = encrypted_path


def decrypt_file(src_path, dst_path, private_key):
    Crypto.Random.atfork()

    if isinstance(private_key, basestring):
        private_key = RSA.importKey(private_key)
    cipher = PKCS1_OAEP.new(private_key)

    # Read the secret key used to encrypt the file
    with open(src_path, 'rb') as src:
        key_len              = int(src.readline())
        encrypted_secret_key = src.read(key_len)
        secret_key           = cipher.decrypt(encrypted_secret_key)
        sym_cipher           = AES.new(secret_key[:-16], AES.MODE_CFB, secret_key[32:])

        # Decrypt the file
        with open(dst_path, 'wb') as dst:
            dst.write(sym_cipher.decrypt(src.read()))

    return dst_path


class NSSingleton(object):
    def __init__(self):
        assert not hasattr(self.__class__, '_%s__instance' % self.__class__.__name__), \
            '%s singleton already instantiated'
        self.__class__.__instance = self

    @classmethod
    def instance(cls):
        try:
            return cls.__instance
        except AttributeError:
            return cls()


class NSJsonParser(object):
    @classmethod
    def load(cls, fp):
        return _json.load(fp, object_hook=cls.object_hook)

    @classmethod
    def loads(cls, s):
        return _json.loads(s, object_hook=cls.object_hook)

    @classmethod
    def dump(cls, obj, fp, indent=None):
        return _json.dump(obj, fp, indent=indent, default=cls.default)

    @classmethod
    def dumps(cls, obj, indent=None):
        return _json.dumps(obj, indent=indent, default=cls.default)

    @classmethod
    def object_hook(cls, dct):
        for key, value in dct.iteritems():
            try:
                hook = cls.__object_hooks__.get(key)
            except AttributeError:
                from network import NSMacAddress, NSIpAddress, NSSensorId, NSLocalUnitId

                cls.__object_hooks__ = {
                    '$mac'      : lambda d, v: NSMacAddress(v),
                    '$ip'       : lambda d, v: NSIpAddress(v),
                    '$sid'      : lambda d, v: NSSensorId.parse(v),
                    '$luid'     : lambda d, v: NSLocalUnitId.parse(v),
                    '$uuid'     : lambda d, v: uuid.UUID(v),
                    '$undefined': lambda d, v: None,
                    '$date'     : cls.__datetime_hook__
                }
                hook = cls.__object_hooks__.get(key)

            if hook:
                return hook(dct, value)

            try:
                from bson import json_util
                return json_util.object_hook(dct)
            except:
                return dct
        return dct

    @classmethod
    def default(cls, obj):
        try:
            default = cls.__defaults__.get(type(obj))
        except AttributeError:
            from network import NSMacAddress, NSIpAddress, NSSensorId, NSLocalUnitId

            cls.__defaults__ = {
                NSMacAddress     : lambda o: {'$mac' : str(o)},
                NSIpAddress      : lambda o: {'$ip'  : str(o)},
                NSSensorId       : lambda o: {'$sid' : str(o)},
                NSLocalUnitId    : lambda o: {'$luid': str(o)},
                uuid.UUID        : lambda o: {"$uuid": o.hex},
                datetime.datetime: cls.__datetime_default__,
            }
            default = cls.__defaults__.get(type(obj))

        if default:
            return default(obj)

        try:
            from bson import json_util
            return json_util.default(obj)
        except:
            raise TypeError("%r is not JSON serializable" % obj)

    @classmethod
    def __datetime_default__(cls, obj):
        if obj.utcoffset() is not None:
            obj = obj - obj.utcoffset()
        millis = int(calendar.timegm(obj.timetuple()) * 1000 + obj.microsecond / 1000)
        return {"$date": millis}

    @classmethod
    def __datetime_hook__(cls, dct, value):
        if isinstance(value, basestring):
            aware  = datetime.datetime.strptime(value[:23], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=pytz.utc)
            offset = value[23:]
            if not offset or offset == 'Z': # UTC
                return aware
            else:
                if len(offset) == 5:  # Offset from mongoexport is in format (+|-)HHMM
                    secs           = (int(offset[1:3]) * 3600 + int(offset[3:]) * 60)
                elif ':' in offset and len(offset) == 6: # RFC-3339 format (+|-)HH:MM
                    hours, minutes = offset[1:].split(':')
                    secs           = (int(hours) * 3600 + int(minutes) * 60)
                else:  # Not RFC-3339 compliant or mongoexport output.
                    raise ValueError("invalid format for offset")
                if offset[0] == "-":
                    secs          *= -1
                return aware - datetime.timedelta(seconds=secs)
        elif isinstance(value, collections.Mapping):  # mongoexport 2.6 and newer, time before the epoch (SERVER-15275)
            secs = float(value["$numberLong"]) / 1000.0
        else:  # mongoexport before 2.6
            secs = float(value) / 1000.0

        return EPOCH_AWARE + datetime.timedelta(seconds=secs)

json = NSJsonParser()


class NSCaseInsensitiveDict(collections.MutableMapping):
    def __init__(self, data=None, **kwargs):
        self.__items = {}
        self.update(data or {}, **kwargs)

    def iteritems(self):
        return self.__items.itervalues()

    def items(self):
        return list(self.iteritems())

    def itervalues(self):
        for key, value in self.__items.itervalues():
            yield value

    def values(self):
        return list(self.itervalues())

    def __setitem__(self, key, value):
        self.__items[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self.__items[key.lower()][1]

    def __delitem__(self, key):
        del self.__items[key.lower()]

    def __iter__(self):
        for key, value in self.__items.itervalues():
            yield key

    def __len__(self):
        return len(self.__items)

    def __eq__(self, other):
        if isinstance(other, collections.Mapping):
            other = NSCaseInsensitiveDict(other)
        else:
            return NotImplemented

        if len(self.__items) != len(other.__items):
            return False

        try:
            for key, value in self.__items.itervalues():
                if value != other[key]:
                    return False
            return True
        except KeyError:
            return False

    def copy(self):
        return NSCaseInsensitiveDict(self.__items.itervalues())

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, dict(self.iteritems()))


class NSVersion(tuple):
    def __new__(cls, major, minor, patch, *args, **kwargs):
        return super(NSVersion, cls).__new__(cls, (int(major), int(minor), int(patch)))

    def __init__(self, major, minor, patch):
        pass

    @property
    def major(self):
        return self[0]

    @property
    def minor(self):
        return self[1]

    @property
    def patch(self):
        return self[2]

    @staticmethod
    def parse(text):
        try:
            tags = text.split('.')
            if len(tags) != 3:
                raise RuntimeError('Invalid number of tags!')
            return NSVersion(tags[0], tags[1], tags[2])
        except:
            from .logs import NSLog
            NSLog.exception('Failed to parse version string "%s"!', text)
            return None

    def __cmp__(self, other):
        if isinstance(other, NSVersion):
            if self.major < other.major:
                return -1
            if self.major > other.major:
                return 1
            if self.minor < other.minor:
                return -1
            if self.minor > other.minor:
                return 1
            if self.patch < other.patch:
                return -1
            if self.patch > other.patch:
                return 1
            return 0
        if other is None:
            return -1
        raise NotImplementedError()

    def __str__(self):
        return '%d.%d.%d' % self


def update(credentials, unit_type, install_dependencies=False):
    from .settings import settings
    from .network  import requests

    print 'Downloading update zip file...'
    zip_content = requests.post(make_url(settings.SERVER_HOSTNAME, 'source'), dict(credentials, code=unit_type))
    zip_content.raise_for_status()

    if not os.path.exists(settings.TMP_DIR):
        os.makedirs(settings.TMP_DIR)

    print 'Uncompressing zip file...'
    zip_path = os.path.join(settings.TMP_DIR, 'update.zip')
    with open(zip_path, 'w') as f:
        f.write(zip_content.content)

    # Extract the update
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(settings.BASE_DIR)

    # Make sure that we don't overwrite the settings except the new MODULES and the new VERSION
    with open(os.path.join(settings.CONF_DIR, unit_type, 'settings.json'), 'r') as f:
        new_settings = json.load(f)
        del settings.MODULES[:]
        settings.MODULES.extend(new_settings['MODULES'])
        settings.VERSION = NSVersion.parse(new_settings.get('VERSION', str(settings.VERSION)))
        settings.push()

    # Optionally install new dependencies
    if install_dependencies:
        import subprocess

        try:
            print subprocess.check_output([os.path.join(settings.LOCAL_UNIT_CONF_DIR, 'system_update.sh')])
        except subprocess.CalledProcessError as e:
            print >> sys.stderr, 'Failed to install dependencies: %s' % e

