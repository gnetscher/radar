# Setup
This is our backend used to access videos. To access the videos hassle-free, follow the steps below:
```
pip install conf/requirements.txt
```
Installs relevant dependencies for usage. Finally add this directory to your PYTHONPATH and follow the steps below

# Access Videos
To be able to use the backend in your own environment, first make sure to install the requirements in conf/requirements.txt. To retrieve a video, you can either directly fetch it or iterate on a sub set of videos.

```python
from standalone import session, NSVideo, NSEnum

# Open a session with the server
session.authenticate('Nokia', '**********')

# Direct access. Raise an exception if no video is matching the given filter or if more than one is...
try:
    video = NSVideo.objects.get(_id='...')
except:
    print 'No video found!'

# ... or through an iterator.
for video in NSVideo.objects.filter(stages__bexact=NSEnum.Stage.Checked):
    pass  # Do whatever...
```
            
The filtering methods follow the [Django model system](https://docs.djangoproject.com/en/1.10/ref/models/querysets/) with a couple of additions:
- The extension bcontains performs a bit or operation (a | b).
- The extension bexact performs a binary and operation ((a & b) == b).
Note though that a lot of the django features are not implemented.


Because the videos themselves are stored in a server, you will have to download them. You can specify the destination directory, the filename to use and if you should overwrite a video already existing with the same name (by default False). You can also provide a callback to be notified when the upload is finished.

```python
# Blocking call. Returns if the download was successful...
success = video.download(dst_dir=None, filename=None, overwrite=False)

# ... or asynchronous call. In this case the method returns the thread in which the download is happening in order to join.
def callback(video, success):
    pass # Do whatever...

thread = video.download(callback=callback)

# You can also open the video file itself if needed. If the video is not already present, it will be downloaded
with video.open(mode='rb', dst_dir=None, filename=None, overwrite=False) as f:
    pass # Do whatever...
```
            
            
# Access Annotations
Getting the video annotations is straightforward. Just call the appropriate field.

```python
entries        = video.annotations  
global_entries = video.global_annotations

# global_entries are labels on the whole video. For instance you could have:
#    global_entries = {'Importance': 'OnTheGround'}
# entries are labels for a region of the video. They are organized as follow:
#    entries        = {'Object1':                                     # Name of the labeled object
#                           {0: {                                     # Index of a frame
#                               'block' : [100, 43, 238, 219],        # Bounding box [left, top, right, bottom]
#                               'labels': ['Person', 'OnTheGround'],  # Labels for this bounding box
#                               'conf'  : 0.9,                        # Confidence, if any
#                               }, ...
#                           }, ...
#                     }
```
            
Finding the proper options to fetch annotated videos can be tricky, therefore you have a shortcut for that. You can also specify if you are only interested in fall videos (i.e. someone with something else than his feet touching the ground).

```python
fall_videos = NSVideo.fetch_annotated_videos(falls_only=True)
```
            
This method returns a cursor, so you can directly use it as a list or add more filters to it. All the videos returned by this methods have been validated and therefore are safe to use for training.
