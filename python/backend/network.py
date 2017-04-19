from  abc              import abstractmethod, abstractproperty
from  netifaces        import interfaces, ifaddresses, AF_INET
from  Crypto.Cipher    import PKCS1_OAEP, AES
from  Crypto.PublicKey import RSA
from .helpers          import NSSingleton, struct as make_struct, PRETTY_NAME_PATTERN
from .enums            import NSEnum
from .logs             import NSLog, rebind_logger
from .settings         import settings
import requests        as    _requests
import ipaddress
import re
import socket
import threading
import struct
import os
import netifaces


settings.optional('SOCKET_BUFFER_SIZE'    , 2048)
settings.optional('MAX_SOCKET_CONNECTIONS', 20)
settings.optional('URLLIB_LOG_LEVEL'      , 'WARNING')

rebind_logger('urllib3', log_level=settings.URLLIB_LOG_LEVEL)


class NSIpAddress(object):
    def __init__(self, value):
        try:
            self.__value = ipaddress.ip_address(unicode(value) if isinstance(value, basestring) else value)
        except:
            NSLog.warning('Invalid ip address "%s"!', value)
            self.__value = value

    def __eq__(self, other):
        return isinstance(other, NSIpAddress) and self.__value == other.__value

    def __ne__(self, other):
        return not isinstance(other, NSIpAddress) or self.__value != other.__value

    def __hash__(self):
        return hash(self.__value)

    def __str__(self):
        return str(self.__value)

    @classmethod
    def localhost(cls):
        return NSIpAddress(socket.gethostbyname(socket.gethostname()))

    @classmethod
    def interfaces(cls):
        for interface_name in interfaces():
            for interface in ifaddresses(interface_name).setdefault(AF_INET, [{'addr': None}]):
                address = interface['addr']
                if address is not None:
                    yield interface_name, NSIpAddress(address)


class NSIpTarget(tuple):
    def __new__(cls, ip, port, *args, **kwargs):
        return super(NSIpTarget, cls).__new__(cls, tuple((str(ip), port)))

    def __init__(self, ip, port):
        if isinstance(ip, NSIpAddress):
            self.__ip = ip
        else:
            try:
                self.__ip = NSIpAddress(ipaddress.ip_address(unicode(ip) if isinstance(ip, basestring) else ip))
            except:
                self.__ip = str(ip)

    @property
    def ip_address(self):
        return self.__ip

    @property
    def port(self):
        return self[1]

    @property
    def target(self):
        return self

    def __str__(self):
        return '%s:%d' % self


class NSMacAddress(object):
    __limiters__ = '[:.-]'
    __pattern__  = r'(?:[0-9A-Fa-f]{2}%s){5}(?:[0-9A-Fa-f]{2})' % __limiters__
    __regex__    = re.compile(__pattern__)
    __breaks__   = re.compile(__limiters__)

    def __init__(self, mac=None):
        if isinstance(mac, NSMacAddress):
            self.__value = mac.__value
        elif mac is None:
            self.__value = NSMacAddress.local().__value
        elif isinstance(mac, basestring):
            mac          = self.__breaks__.sub('', mac)
            self.__value = int(mac, 16)
        else:
            assert isinstance(mac, (int, long)), 'Invalid mac address "%s"!' % mac
            self.__value = mac

    def format(self, sep=':'):
        assert self.__breaks__.match(sep)
        return sep.join(("%012X" % self.__value)[i:i + 2] for i in range(0, 12, 2))

    def __cmp__(self, other):
        if isinstance(other, NSMacAddress):
            return self.__value - other.__value
        if other is None:
            return -1
        raise NotImplementedError()

    def __hash__(self):
        return self.__value

    def __repr__(self):
        return self.format(':')

    @classmethod
    def pattern(cls):
        return cls.__pattern__

    @classmethod
    def regex(cls):
        return cls.__regex__

    @classmethod
    def local(cls):
        try:
            return cls.__local
        except AttributeError:
            try:
                interface   = netifaces.gateways()['default'][netifaces.AF_INET][1]
                cls.__local =  NSMacAddress(netifaces.ifaddresses(interface)[netifaces.AF_LINK][0]['addr'])
                return cls.__local
            except:
                import time
                time.sleep(2)
                return NSMacAddress.local()


class NSSensorId(object):
    __pattern__ = r'(' + PRETTY_NAME_PATTERN + ')@(' + NSMacAddress.pattern() + r')\[(\w+)\]'
    __regex__   = re.compile(__pattern__)

    def __init__(self, home, mac, type):
        self.__home = home
        self.__mac  = NSMacAddress(mac)
        self.__type = NSEnum.SensorType(type)

    @property
    def home(self):
        return self.__home

    @property
    def mac(self):
        return self.__mac

    @property
    def type(self):
        return self.__type

    @classmethod
    def parse(cls, text):
        m = cls.__regex__.match(text)
        return cls(m.group(1), m.group(2), m.group(3)) if m else None

    def __cmp__(self, other):
        if isinstance(other, NSSensorId):
            return self.__mac.__cmp__(other.__mac)
        if other is None:
            return -1
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.__mac)

    def __str__(self):
        return '%s@%s[%s]' % (self.__home, self.__mac, self.__type)

    def __repr__(self):
        return '<NSSensorId@%s>' % self


class NSLocalUnitId(object):
    __pattern__ = r'(' + PRETTY_NAME_PATTERN + ')@(' + NSMacAddress.pattern() + r')'
    __regex__   = re.compile(__pattern__)

    def __init__(self, home, mac):
        self.__home = home
        self.__mac  = NSMacAddress(mac)

    @property
    def home(self):
        return self.__home

    @property
    def mac(self):
        return self.__mac

    @classmethod
    def parse(cls, text):
        m = cls.__regex__.match(text)
        return cls(m.group(1), m.group(2)) if m else None

    def __cmp__(self, other):
        if isinstance(other, NSLocalUnitId):
            return self.__mac.__cmp__(other.__mac)
        if other is None:
            return -1
        raise NotImplemented()

    def __hash__(self):
        return hash(self.__mac)

    def __str__(self):
        return '%s@%s' % (self.__home, self.__mac)

    def __repr__(self):
        return '<NSLocalUnitId@%s>' % self


class NSAbstractServer(NSIpTarget):
    def __init__(self, ip, port):
        super(NSAbstractServer, self).__init__(ip, port)
        self.__thread = None

    @property
    def is_active(self):
        return self.__thread is not None and self.__thread.is_alive

    def start(self):
        if not self.is_active:
            self._start_()
            self.__thread        = threading.Thread(target=self.__run__)
            self.__thread.daemon = True
            self.__thread.start()

    def stop(self):
        if self.is_active:
            thread        = self.__thread
            self.__thread = None
            self._stop_()
            thread.join()

    @abstractmethod
    def _start_(self):
        pass

    @abstractmethod
    def _stop_(self):
        pass

    @abstractmethod
    def _accept_connection_(self):
        pass

    @abstractmethod
    def _on_client_connection_(self, connection):
        pass

    def __run__(self):
        while self.is_active:
            NSLog.debug('Waiting for client connection...')
            connection = self._accept_connection_()
            if connection:
                NSLog.debug('Client connection established with %s.', connection)

                client_thread        = threading.Thread(target=self.__on_client_connection__, args=[connection])
                client_thread.daemon = True
                client_thread.start()

    def __on_client_connection__(self, connection):
        try:
            self._on_client_connection_(connection)
        except:
            NSLog.exception('Unexpected exception while processing client connection %s!', connection)


class NSProtocol(object):
    class Udp(object):
        class Server(NSAbstractServer):
            def __init__(self, ip, port, callback):
                super(NSProtocol.Udp.Server, self).__init__(ip, port)
                self.__callback = callback
                self.__target   = ('', self.target[1])
                self.__socket   = None

            def send(self, msg):
                if len(msg) <= 256:
                    NSLog.debug('Broadcasting message: "%s".', msg)
                self.__socket.sendto(msg, self.target)

            def _start_(self):
                self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.__socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.__socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self.__socket.bind(self.__target)

            def _stop_(self):
                self.__socket.shutdown(socket.SHUT_RDWR)
                self.__socket.close()

            def _accept_connection_(self):
                try:
                    buf, target = self.__socket.recvfrom(settings.SOCKET_BUFFER_SIZE)
                    msg         = buf
                    while len(buf) == settings.SOCKET_BUFFER_SIZE:
                        buf     = self.__socket.recvfrom(settings.SOCKET_BUFFER_SIZE)[0]
                        msg    += buf
                except Exception as e:
                    return None

                if msg:
                    if len(msg) <= 256:
                        NSLog.debug('Received broadcast message: "%s:.', msg)

                    return NSProtocol.Udp.IncomingConnection(target[0], target[1], msg)
                return None

            def _on_client_connection_(self, connection):
                self.__callback(connection.msg)

        class IncomingConnection(NSIpTarget):
            def __init__(self, ip, port, msg):
                NSIpTarget.__init__(self, ip, port)
                self.msg = msg

    class Tcp(object):
        class Socket(NSIpTarget):
            def __init__(self, ip, port, cipher=None, s=None):
                super(NSProtocol.Tcp.Socket, self).__init__(ip, port)
                self._socket = s
                self._cipher = cipher

            def close(self):
                if self._socket:
                    self._socket.shutdown(socket.SHUT_RDWR)
                    self._socket.close()
                    self._socket = None

            def recv(self):
                msg = self.recvall(self._socket)
                msg = self._cipher.decrypt(msg) if msg else msg

                if len(msg) <= 256:
                    NSLog.debug('Received message: "%s".', msg)

                return msg

            def send(self, msg):
                if len(msg) <= 256:
                    NSLog.debug('Sending message: "%s".', msg)

                return self.sendall(self._socket, self._cipher.encrypt(msg))

            @classmethod
            def recvall(cls, connection):
                msg_len = cls.__recvall__(connection, 4)
                if not msg_len:
                    return ''
                msg_len = struct.unpack('>I', msg_len)[0]
                msg     = cls.__recvall__(connection, msg_len)
                return msg

            @classmethod
            def __recvall__(cls, connection, n):
                msg = ''
                while len(msg) < n:
                    try:
                        packet  = connection.recv(n - len(msg))
                        if not packet:
                            return msg
                        msg    += packet
                    except Exception as e:
                        return msg
                return msg

            @classmethod
            def sendall(cls, connection, msg):
                return connection.sendall(struct.pack('>I', len(msg)) + msg)

        class Server(NSAbstractServer):
            def __init__(self, ip, port, callback, key, timeout=0):
                super(NSProtocol.Tcp.Server, self).__init__(ip, port)

                if isinstance(key, basestring):
                    key = RSA.importKey(key)

                self.__callback = callback
                self.__timeout  = timeout
                self.__socket   = None
                self.__cipher   = PKCS1_OAEP.new(key)

            def _start_(self):
                self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.__socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.__socket.bind(self.target)
                self.__socket.listen(settings.MAX_SOCKET_CONNECTIONS)

            def _stop_(self):
                self.__socket.shutdown(socket.SHUT_RDWR)
                self.__socket.close()

            def _accept_connection_(self):
                try:
                    connection, target = self.__socket.accept()
                    if not connection:
                        return None
                except:
                    return None

                connection.settimeout(self.__timeout)

                # Handshake
                try:
                    key = NSProtocol.Tcp.Socket.recvall(connection)
                    if not key:
                        connection.shutdown(socket.SHUT_RDWR)
                        connection.close()
                        raise RuntimeError('Empty cipher key!')

                    try:
                        key = self.__cipher.decrypt(key)
                    except:
                        NSLog.error('Failed to decrypt cipher key "%s"!', key)
                        key = ''

                    if len(key) != 48:
                        connection.shutdown(socket.SHUT_RDWR)
                        connection.close()
                        raise RuntimeError('Invalid cipher key!')

                    cipher     = AES.new(key[:-16], AES.MODE_CFB, key[32:])
                    connection = NSProtocol.Tcp.Socket(target[0], target[1], cipher, connection)
                    connection.send('HAND')
                    if connection.recv() != 'SHAKE':
                        connection.close()
                        raise RuntimeError('Invalid handshake key word!')

                    NSLog.debug('Successful server handshake!')
                except:
                    NSLog.exception('Failed to perform server handshake!')
                    return None

                return connection

            def _on_client_connection_(self, connection):
                self.__callback(connection, connection.recv())

        class Client(Socket):
            def __init__(self, ip, port, key, timeout=0):
                if isinstance(key, basestring):
                    key = RSA.importKey(key)

                super(NSProtocol.Tcp.Client, self).__init__(ip, port, PKCS1_OAEP.new(key))

                self.__timeout = timeout
                self.__cipher  = self._cipher

            def open(self):
                self.close()

                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._socket.connect(self)
                self._socket.settimeout(self.__timeout)

                # Handshake
                key = os.urandom(48)
                self.send(key)
                self._cipher = AES.new(key[:-16], AES.MODE_CFB, key[32:])
                if self.recv() != 'HAND':
                    raise RuntimeError('Failed to perform client handshake!')
                self.send('SHAKE')

                NSLog.debug('Successful client handshake!')

            def close(self):
                super(NSProtocol.Tcp.Client, self).close()
                self._cipher = self.__cipher

            def __enter__(self):
                self.open()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()
                return False


class NSRequests(NSSingleton):
    class AbstractResponse(object):
        def raise_for_status(self):
            http_error_msg = ''
            if 400 <= self.status_code < 500:
                http_error_msg = '%s Client error for url "%s":\n%s' % (self.status_code, self.url, self.reason)
            elif 500 <= self.status_code < 600:
                http_error_msg = '%s Server error for url "%s":\n%s' % (self.status_code, self.url, self.reason)
            if http_error_msg:
                raise _requests.HTTPError(http_error_msg, response=self)

    class Response(AbstractResponse):
        def __init__(self, html_response):
            self.__response = html_response

        @abstractproperty
        def status_code(self):
            return self.__response.status_code

        @abstractproperty
        def url(self):
            pass

        @abstractproperty
        def reason(self):
            return self.__response.reason_phrase

        @abstractproperty
        def content(self):
            return self.__response.content

        @abstractproperty
        def text(self):
            try:
                self.raise_for_status()
                return self.__response.content
            except:
                return self.__response.reason_phrase

        @property
        def ok(self):
            try:
                self.raise_for_status()
                return True
            except:
                return False

    class ErrorResponse(make_struct('ResponseBase', ['url', 'status_code', 'reason', 'error']), AbstractResponse):
        @property
        def ok(self):
            return False

        @property
        def text(self):
            return self.reason

    class Session(_requests.Session):
        def request(self, method, url, **kwargs):
            try:
                return super(NSRequests.Session, self).request(method, url, **kwargs)
            except requests.exceptions.Timeout as e:
                return NSRequests.ErrorResponse(url=url, status_code=408, reason='Request Time-out', error=e)
            except requests.exceptions.TooManyRedirects as e:
                return NSRequests.ErrorResponse(url=url, status_code=310, reason='Too many Redirects', error=e)
            except requests.exceptions.RequestException as e:
                try:
                    return NSRequests.ErrorResponse(url=url, status_code=520, reason=str(e), error=e)
                except TypeError:  # Fucking urllib3 bug!!!
                    return NSRequests.ErrorResponse(url=url, status_code=520, reason=str(type(e)), error=e)
            except Exception as e:
                return NSRequests.ErrorResponse(url=url, status_code=500, reason=str(e), error=e)

    class IgnoringHostnameAdapter(_requests.adapters.HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False):
            self.poolmanager = _requests.packages.urllib3.poolmanager.PoolManager(num_pools       = connections,
                                                                                  maxsize         = maxsize,
                                                                                  block           = block,
                                                                                  assert_hostname = False)

    exceptions = _requests.exceptions

    @classmethod
    def request(cls, method, url, **kwargs):
        with cls.Session() as session:
            return session.request(method=method, url=url, **kwargs)

    @classmethod
    def get(cls, url, params=None, **kwargs):
        kwargs.setdefault('allow_redirects', True)
        return cls.request('get', url, params=params, **kwargs)

    @classmethod
    def post(cls, url, data=None, json=None, **kwargs):
        return cls.request('post', url, data=data, json=json, **kwargs)

requests = NSRequests.instance()

