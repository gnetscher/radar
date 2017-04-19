from backend.helpers import NSSingleton
from backend.network import requests


class NSSession(NSSingleton):
    def __init__(self):
        super(NSSession, self).__init__()

        self.__session = requests.Session()
        self.__auth    = False
        self.hostname  = 'services.safely-you.com'
        self.port      = 443

    def authenticate(self, username, password):
        self.__auth = True

        # Retrieve some cookies used by djangos
        try:
            self.get('login').raise_for_status()
            token = self.__session.cookies['csrftoken']

            # Authenticate
            self.post('login', data={'username'           : username,
                                     'password'           : password,
                                     'csrfmiddlewaretoken': token}).raise_for_status()
        except:
            self.__auth = False
            raise

    def post(self, path, *args, **kwargs):
        if not self.__auth:
            raise RuntimeError('User not authenticated! Please execute session.authenticate(username, password).')

        if self.port == 443:
            url                = 'https://%s' % self.hostname
            headers            = kwargs.pop('headers', {})
            headers['Referer'] = url
            return self.__session.post('%s/%s' % (url, path), *args, headers=headers, **kwargs)
        return self.__session.post('http://%s:%d/%s' % (self.hostname, self.port, path), *args, **kwargs)

    def get(self, path, *args, **kwargs):
        if not self.__auth:
            raise RuntimeError('User not authenticated! Please execute session.authenticate(username, password).')

        if self.port == 443:
            return self.__session.get('https://%s/%s' % (self.hostname, path), *args, **kwargs)
        return self.__session.get('http://%s:%d/%s' % (self.hostname, self.port, path), *args, **kwargs)


session = NSSession()

