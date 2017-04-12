'''
try:
    from .helpers       import make_pretty, format_dict, check_db_connection
    from .video_helpers import video_metadata, truncate_video, prepare_video
    from .structs       import *
    from .network       import *
except Exception as e:
    import sys
    import traceback

    print >> sys.stderr, "Failed to module classes and methods!"
    traceback.print_exc(file=sys.stderr)

    exit(-1)


try:
    # Module(s) only present in server units
    from .data import *
except Exception as e:
    pass
'''

from .settings import settings
