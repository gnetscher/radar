from  backend.settings import settings
from  backend.enums    import NSEnum
from  backend.helpers  import json
from  backend.network  import NSSensorId
from .auth             import session
import os
import threading


BATCH_SIZE = 1000


class NSQuery(object):
    class Batch(object):
        def __init__(self, start):
            self.start    = start
            self.end      = start
            self.next     = None
            self.valid    = False
            self.entries  = []

    def __init__(self, model):
        self.__model      = model
        self.__filters    = [{}]
        self.__excludes   = [{}]
        self.__order_by   = []
        self.__last_batch = NSQuery.Batch(0)
        self.__batches    = {0: self.__last_batch}

    def get(self, **kwargs):
        clone = self.clone()
        clone.__update_filters__(clone.__filters , **kwargs)

        response = session.post('development/entry_metadata', data={
            'model'   : self.__model.__name__,
            'filters' : json.dumps(clone.__filters),
            'excludes': json.dumps(clone.__excludes)
        })
        response.raise_for_status()

        try:
            return self.__model(**response.json())
        except ValueError:
            raise RuntimeError(response.text)

    def filter(self, **kwargs):
        clone = self.clone()
        return clone.__update_filters__(clone.__filters , **kwargs)

    def exclude(self, **kwargs):
        clone = self.clone()
        return clone.__update_filters__(clone.__excludes , **kwargs)

    def order_by(self, *fields):
        clone            = self.clone()
        clone.__order_by = list(fields)
        return clone

    def count(self):
        try:
            return self.__count
        except AttributeError:
            if self.__last_batch.valid:
                batch        = self.__batches[0]
                self.__count = 0
                while batch:
                    if not batch.valid:
                        self.__fetch_batch__(batch)
                    self.__count += len(batch.entries)
                    batch         = batch.next
            else:
                response = session.post('development/count', data={
                    'model'   : self.__model.__name__,
                    'filters' : json.dumps(self.__filters),
                    'excludes': json.dumps(self.__excludes),
                })

                try:
                    response.raise_for_status()
                    response     = response.json()
                    self.__count = response['count']
                except ValueError:
                    print response.text
                    raise RuntimeError(response.text)
            return self.__count

    def exists(self):
        return self.count() > 0

    def clone(self):
        res            = NSQuery(self.__model)
        res.__filters  = [dict(filter) for filter in self.__filters]
        res.__excludes = [dict(filter) for filter in self.__excludes]
        res.__order_by = list(self.__order_by)
        return res

    def __update_filters__(self, filters, **kwargs):
        for key, value in kwargs.iteritems():
            inserted = False
            for filter in filters:
                if key not in filter:
                    inserted    = True
                    filter[key] = self.__fmt_value__(value)
                    break
            if not inserted:
                filters.append({key: self.__fmt_value__(value)})
        return self

    def __get_batch__(self, frame):
        b         = int(frame / BATCH_SIZE)
        try:
            batch = self.__batches[b]
        except KeyError:
            batch = self.__batches[b] = NSQuery.Batch(b * BATCH_SIZE)
            if batch.start > self.__last_batch.start:
                self.__last_batch = batch
        return batch

    def __fetch_batch__(self, batch):
        response = session.post('development/entry_ids', data={
            'model'   : self.__model.__name__,
            'filters' : json.dumps(self.__filters),
            'excludes': json.dumps(self.__excludes),
            'order_by': json.dumps(self.__order_by),
            'start'   : batch.start
        })

        try:
            response.raise_for_status()
            response       = response.json()
            batch.entries += response['entries']
            batch.end      = response['end']
            batch.valid    = True

            if response['has_more']:
                batch.next = self.__get_batch__(batch.end)

        except ValueError:
            print response.text
            raise RuntimeError(response.text)

    def __entries__(self):
        batch = self.__batches[0]
        while batch:
            if not batch.valid:
                self.__fetch_batch__(batch)

            for i, entry in enumerate(batch.entries):
                yield batch, i, entry

            batch = batch.next

    def __iter__(self):
        for i, (batch, j, entry) in enumerate(self.__entries__()):
            if not isinstance(entry, self.__model):
                batch.entries[j] = entry = NSQuery(self.__model).get(_id=entry)
            yield entry

    def __len__(self):
        return self.count()

    def __getitem__(self, index):
        if isinstance(index, slice):
            entries   = []
            batch     = self.__batches[0]
            for i in range(index.start, index.stop, index.step or 1):
                if not (batch.start <= i < batch.end):
                    batch = self.__get_batch__(i)

                if not batch.valid:
                    self.__fetch_batch__(batch)

                j     = i - batch.start * BATCH_SIZE
                entry = batch.entries[j]
                if not isinstance(entry, self.__model):
                    batch.entries[j] = entry = NSQuery(self.__model).get(_id=entry)
                entries.append(entry)
            return entries
        else:
            batch = self.__get_batch__(index)
            if not batch.valid:
                self.__fetch_batch__(batch)

            j     = index - batch.start * BATCH_SIZE
            entry = batch.entries[j]
            if not isinstance(entry, self.__model):
                batch.entries[j] = entry = NSQuery(self.__model).get(_id=entry)
            return entry

    @classmethod
    def __fmt_value__(cls, value):
        if isinstance(value, (str, int, float)):
            return value
        return str(value)


class NSDataEntryMetadata(type):
    @property
    def objects(cls):
        return NSQuery(cls)


class NSDataEntry(object):
    __metaclass__ = NSDataEntryMetadata

    def __init__(self, _id, sensor_id, start, end, stages=0, flags=0, **kwargs):
        self.__id            = _id
        if isinstance(sensor_id, basestring):
            self.__sensor_id = NSSensorId.parse(sensor_id)
        else:
            assert isinstance(sensor_id, NSSensorId)
            self.__sensor_id = sensor_id
        self.__start         = start
        self.__end           = end
        self.__stages        = NSEnum.Stage   .Flags(stages)
        self.__flags         = NSEnum.DataFlag.Flags(flags)

    @property
    def id(self):
        return self.__id

    @property
    def sensor_id(self):
        return self.__sensor_id

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def duration(self):
        return self.__end - self.__start

    @property
    def stages(self):
        return self.__stages

    @property
    def flags(self):
        return self.__flags

    @property
    def annotations(self):
        try:
            return self.__annotations
        except AttributeError:
            return self.__fetch_annotations__()[0]

    @property
    def global_annotations(self):
        try:
            return self.__global_annotations
        except AttributeError:
            return self.__fetch_annotations__()[1]

    def __fetch_annotations__(self):
        response                  = session.post('development/annotations', data={'id': self.id})
        response.raise_for_status()
        try:
            response              = response.json()
        except ValueError:
            raise RuntimeError(response.text)
        self.__annotations        = self, response['entries']
        self.__global_annotations = response['global_entries']
        return self.__annotations, self.__global_annotations

    def __repr__(self):
        return '<%s]%s: %s - %s>' % (self.__class__.__name__, self.sensor_id, self.__start, self.__end)


class NSVideo(NSDataEntry):
    def __init__(self, _id, sensor_id, start, end, width, height, fps, frame_count, stages=0, flags=0, **kwargs):
        super(NSVideo, self).__init__(_id, sensor_id, start, end, stages, flags, **kwargs)
        self.__width       = width
        self.__height      = height
        self.__fps         = fps
        self.__frame_count = frame_count
        self.__path        = None

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def fps(self):
        return self.__fps

    @property
    def frame_count(self):
        return self.__frame_count

    @property
    def path(self):
        return self.__path

    def download(self, dst_dir=None, callback=None, filename=None, overwrite=False):
        if dst_dir is None:
            dst_dir  = settings.TMP_DIR
        else:
            dst_dir  = os.path.abspath(dst_dir)
        if filename is None:
            filename = '%s.mp4' % self.id

        self.__path  = os.path.join(dst_dir, filename)
        if overwrite and os.path.exists(self.__path):
            os.remove(self.__path)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        result = {'success': False}
        if callback is None:
            return self.__fetch_video__(result)

        thread = threading.Thread(target=self.__fetch_video__, args=[result, callback])
        thread.start()
        return thread

    def open(self, mode='rb', dst_dir=None, filename=None, overwrite=False):
        if self.download(dst_dir=dst_dir, filename=filename, overwrite=overwrite):
            return open(self.path, mode)
        return open(os.devnull, mode)

    @classmethod
    def fetch_annotated_videos(cls, home=None, falls_only=False):
        flag   = NSEnum.Stage.BBoxAnnotationStep
        cursor = cls.objects.filter(stages__bexact=flag, stages__bcontains=flag.gt_mask() | NSEnum.Stage.Checked)
        if home is not None:
            cursor = cursor.filter(sensor_id__startswith=home)
        if falls_only:
            cursor = cursor.filter(flags__bexact=NSEnum.DataFlag.OnTheGround)
        return cursor

    def __fetch_video__(self, result, callback=None):
        if os.path.exists(self.__path):
            result['success'] = True
            return True

        response = session.post('development/video', data={'id': self.id})
        response.raise_for_status()

        try:
            response  = response.json()
            if response['status'] == 'PENDING':
                timer = threading.Timer(2.0, self.__fetch_video__, args=[result, callback])
                timer.start()
                timer.join()
                return result['success']

            if callback:
                callback(video=self, success=False)
            result['success'] = False
            return False
        except ValueError:  # File content
            with open(self.__path, 'wb') as f:
                f.write(response.content)
            if callback:
                callback(video=self, success=True)
            result['success'] = True
            return True

