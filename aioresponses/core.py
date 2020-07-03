import asyncio
import dataclasses
import json
import copy
from dataclasses import dataclass
from distutils.version import StrictVersion
from functools import wraps
from types import SimpleNamespace
from typing import Callable, Dict, Tuple, Union, Optional, List, Mapping, Any, Iterable, MutableMapping  # noqa
from unittest.mock import Mock, patch
import inspect
from aiohttp import (
    ClientConnectionError,
    ClientResponse,
    ClientSession,
    hdrs,
    http, Fingerprint, ClientTimeout
)
from aiohttp.connector import SSLContext
from aiohttp.helpers import TimerNoop, BasicAuth, sentinel
from aiohttp.typedefs import LooseCookies, LooseHeaders, StrOrURL
from multidict import CIMultiDict, CIMultiDictProxy

from .compat import (
    AIOHTTP_VERSION,
    URL,
    Pattern,
    stream_reader_factory,
    merge_params,
    normalize_url,
)


class CallbackResult:
    def __init__(self, method: str = hdrs.METH_GET,
                 status: int = 200,
                 body: str = '',
                 content_type: str = 'application/json',
                 payload: Dict = None,
                 headers: Dict = None,
                 response_class: 'ClientResponse' = None,
                 reason: Optional[str] = None):
        self.method = method
        self.status = status
        self.body = body
        self.content_type = content_type
        self.payload = payload
        self.headers = headers
        self.response_class = response_class
        self.reason = reason


class RequestMatcher:
    url_or_pattern = None  # type: Union[URL, Pattern]

    def __init__(self, url: Union[URL, str, Pattern],
                 method: str = hdrs.METH_GET,
                 status: int = 200,
                 body: str = '',
                 payload: Dict = None,
                 exception: 'Exception' = None,
                 headers: Dict = None,
                 content_type: str = 'application/json',
                 response_class: 'ClientResponse' = None,
                 timeout: bool = False,
                 repeat: bool = False,
                 reason: Optional[str] = None,
                 callback: Optional[Callable] = None):
        if isinstance(url, Pattern):
            self.url_or_pattern = url
            self.match_func = self.match_regexp
        else:
            self.url_or_pattern = normalize_url(url)
            self.match_func = self.match_str
        self.method = method.lower()
        self.status = status
        self.body = body
        self.payload = payload
        self.exception = exception
        if timeout:
            self.exception = asyncio.TimeoutError('Connection timeout test')
        self.headers = headers
        self.content_type = content_type
        self.response_class = response_class
        self.repeat = repeat
        self.reason = reason
        if self.reason is None:
            try:
                self.reason = http.RESPONSES[self.status][0]
            except (IndexError, KeyError):
                self.reason = ''
        self.callback = callback

        self.requests: List[RequestInfo] = []

    def match_str(self, url: URL) -> bool:
        return self.url_or_pattern == url

    def match_regexp(self, url: URL) -> bool:
        return bool(self.url_or_pattern.match(str(url)))

    def match(self, method: str, url: URL) -> bool:
        if self.method != method.lower():
            return False
        return self.match_func(url)

    def _build_raw_headers(self, headers: MutableMapping) -> Tuple:
        """
        Convert a dict of headers to a tuple of tuples

        Mimics the format of ClientResponse.
        """
        raw_headers = []
        for k, v in headers.items():
            raw_headers.append((k.encode('utf8'), v.encode('utf8')))
        return tuple(raw_headers)

    def _build_response(self, url: 'Union[URL, str]',
                        method: str = hdrs.METH_GET,
                        request_headers: Dict = None,
                        status: int = 200,
                        body: str = '',
                        content_type: str = 'application/json',
                        payload: Dict = None,
                        headers: Dict = None,
                        response_class: 'ClientResponse' = None,
                        reason: Optional[str] = None) -> ClientResponse:
        if response_class is None:
            response_class = ClientResponse
        if payload is not None:
            body = json.dumps(payload)
        if not isinstance(body, bytes):
            body = str.encode(body)
        if request_headers is None:
            request_headers = {}
        kwargs = {}
        if AIOHTTP_VERSION >= StrictVersion('3.1.0'):
            loop = Mock()
            loop.get_debug = Mock()
            loop.get_debug.return_value = True
            kwargs['request_info'] = Mock(
                url=url,
                method=method,
                headers=CIMultiDictProxy(CIMultiDict(**request_headers)),
            )
            kwargs['writer'] = Mock()
            kwargs['continue100'] = None
            kwargs['timer'] = TimerNoop()
            if AIOHTTP_VERSION < StrictVersion('3.3.0'):
                kwargs['auto_decompress'] = True
            kwargs['traces'] = []
            kwargs['loop'] = loop
            kwargs['session'] = None
        else:
            loop = None
        # We need to initialize headers manually
        _headers = CIMultiDict({hdrs.CONTENT_TYPE: content_type})
        if headers:
            _headers.update(headers)
        raw_headers = self._build_raw_headers(_headers)
        resp = response_class(method, url, **kwargs)

        for hdr in _headers.getall(hdrs.SET_COOKIE, ()):
            resp.cookies.load(hdr)

        if AIOHTTP_VERSION >= StrictVersion('3.3.0'):
            # Reified attributes
            resp._headers = _headers
            resp._raw_headers = raw_headers
        else:
            resp.headers = _headers
            resp.raw_headers = raw_headers
        resp.status = status
        resp.reason = reason
        resp.content = stream_reader_factory(loop)
        resp.content.feed_data(body)
        resp.content.feed_eof()
        return resp

    async def build_response(
            self, url: URL, **kwargs
    ) -> 'Union[ClientResponse, Exception]':
        if self.exception is not None:
            return self.exception

        if callable(self.callback):
            if asyncio.iscoroutinefunction(self.callback):
                result = await self.callback(url, **kwargs)
            else:
                result = self.callback(url, **kwargs)
        else:
            result = None
        result = self if result is None else result
        resp = self._build_response(
            url=url,
            method=result.method,
            request_headers=kwargs.get("headers"),
            status=result.status,
            body=result.body,
            content_type=result.content_type,
            payload=result.payload,
            headers=result.headers,
            response_class=result.response_class,
            reason=result.reason)
        return resp


@dataclass
class RequestInfo:
    method: str
    url: URL
    params: Optional[Mapping[str, str]] = None
    data: Any = None
    json: Any = None
    cookies: Optional[LooseCookies] = None
    headers: LooseHeaders = None
    skip_auto_headers: Optional[Iterable[str]] = None
    auth: Optional[BasicAuth] = None
    allow_redirects: bool = True
    max_redirects: int = 10
    compress: Optional[str] = None
    chunked: Optional[bool] = None
    expect100: bool = False
    raise_for_status: Optional[bool] = None
    read_until_eof: bool = True
    proxy: Optional[StrOrURL] = None
    proxy_auth: Optional[BasicAuth] = None
    timeout: Union[ClientTimeout, object] = sentinel
    verify_ssl: Optional[bool] = None
    fingerprint: Optional[bytes] = None
    ssl_context: Optional[SSLContext] = None
    ssl: Optional[Union[SSLContext, bool, Fingerprint]] = None
    proxy_headers: Optional[LooseHeaders] = None
    trace_request_ctx: Optional[SimpleNamespace] = None

    kwargs: Dict = None

    response: Optional[ClientResponse] = None

    def __init__(self, **kwargs):
        self.kwargs = {}
        self.response = None

        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
            else:
                self.kwargs[k] = v


RequestsKey = Tuple[str, URL]


class aioresponses(object):
    """Mock aiohttp requests made by ClientSession."""

    def __init__(self, **kwargs):
        self._param = kwargs.pop('param', None)
        self._passthrough = kwargs.pop('passthrough', [])
        self._passthrough_all = kwargs.pop('passthrough_all', True)
        self.patcher = patch('aiohttp.client.ClientSession._request',
                             side_effect=self._request_mock,
                             autospec=True)

        self._matches: List[RequestMatcher] = []
        self.requests: Dict[RequestsKey, List[RequestInfo]] = {}

    def __enter__(self) -> 'aioresponses':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __call__(self, f):
        def _pack_arguments(ctx, *args, **kwargs) -> Tuple[Tuple, Dict]:
            if self._param:
                kwargs[self._param] = ctx
            else:
                args += (ctx,)
            return args, kwargs

        if asyncio.iscoroutinefunction(f):
            @wraps(f)
            async def wrapped(*args, **kwargs):
                with self as ctx:
                    args, kwargs = _pack_arguments(ctx, *args, **kwargs)
                    return await f(*args, **kwargs)
        else:
            @wraps(f)
            def wrapped(*args, **kwargs):
                with self as ctx:
                    args, kwargs = _pack_arguments(ctx, *args, **kwargs)
                    return f(*args, **kwargs)
        return wrapped

    def clear(self):
        self.requests.clear()
        self._matches.clear()

    def start(self):
        self._matches = []
        self.patcher.start()
        self.patcher.return_value = self._request_mock

    def stop(self) -> None:
        self.patcher.stop()
        self.clear()

    def head(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_HEAD, **kwargs)

    def get(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_GET, **kwargs)

    def post(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_POST, **kwargs)

    def put(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_PUT, **kwargs)

    def patch(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_PATCH, **kwargs)

    def delete(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_DELETE, **kwargs)

    def options(self, url: 'Union[URL, str, Pattern]', **kwargs):
        return self.add(url, method=hdrs.METH_OPTIONS, **kwargs)

    def add(self, url: 'Union[URL, str, Pattern]', method: str = hdrs.METH_GET,
            status: int = 200,
            body: str = '',
            exception: 'Exception' = None,
            content_type: str = 'application/json',
            payload: Dict = None,
            headers: Dict = None,
            response_class: 'ClientResponse' = None,
            repeat: bool = False,
            timeout: bool = False,
            reason: Optional[str] = None,
            callback: Optional[Callable] = None) -> RequestMatcher:

        matcher = RequestMatcher(
            url,
            method=method,
            status=status,
            content_type=content_type,
            body=body,
            exception=exception,
            payload=payload,
            headers=headers,
            response_class=response_class,
            repeat=repeat,
            timeout=timeout,
            reason=reason,
            callback=callback,
        )
        self._matches.append(matcher)

        return matcher

    @staticmethod
    def is_exception(resp_or_exc: Union[ClientResponse, Exception]) -> bool:
        if inspect.isclass(resp_or_exc):
            parent_classes = set(inspect.getmro(resp_or_exc))
            if {Exception, BaseException} & parent_classes:
                return True
        else:
            if isinstance(resp_or_exc, (Exception, BaseException)):
                return True
        return False

    async def match(
            self, method: str, url: URL,
            allow_redirects: bool = True, **kwargs: Dict
    ) -> Optional[ClientResponse]:
        history = []
        while True:
            for i, matcher in enumerate(self._matches):
                if matcher.match(method, url):
                    # noinspection PyTypeChecker
                    response = await matcher.build_response(
                        url, allow_redirects=allow_redirects, **kwargs
                    )

                    if not history:
                        # record original call once

                        key = (method, url)
                        self.requests.setdefault(key, [])
                        try:
                            kwargs_copy = copy.deepcopy(kwargs)
                        except TypeError:
                            # Handle the fact that some values cannot be deep copied
                            kwargs_copy = kwargs

                        request_info = RequestInfo(
                            method=method,
                            url=url,
                            **kwargs_copy,
                            response=response
                        )

                        self.requests[key].append(request_info)
                        matcher.requests.append(request_info)

                    break
            else:
                return None

            if matcher.repeat is False:
                del self._matches[i]
            if isinstance(response, Exception):
                raise response

            if response.status in (
                    301, 302, 303, 307, 308) and allow_redirects:
                if hdrs.LOCATION not in response.headers:
                    break
                history.append(response)
                url = URL(response.headers[hdrs.LOCATION])
                continue
            else:
                break

        response._history = tuple(history)

        return response

    async def _request_mock(self, orig_self: ClientSession,
                            method: str, url: 'Union[URL, str]',
                            *args: Tuple,
                            **kwargs: Dict) -> 'ClientResponse':
        """Return mocked response object or raise connection error."""
        if orig_self.closed:
            raise RuntimeError('Session is closed')

        not_normalized_url = merge_params(url, kwargs.get('params'))
        url = normalize_url(not_normalized_url)
        url_str = str(url)
        for prefix in self._passthrough:
            if url_str.startswith(prefix):
                return (await self.patcher.temp_original(
                    orig_self, method, not_normalized_url, *args, **kwargs
                ))

        response = await self.match(method, url, *args, **kwargs)
        if response is None:
            if not self._passthrough_all:
                raise ClientConnectionError(
                    'Connection refused: {} {}'.format(method, url)
                )
            else:
                return (await self.patcher.temp_original(
                    orig_self, method, not_normalized_url, *args, **kwargs
                ))

        # Automatically call response.raise_for_status() on a request if the
        # request was initialized with raise_for_status=True. Also call
        # response.raise_for_status() if the client session was initialized
        # with raise_for_status=True, unless the request was called with
        # raise_for_status=False.
        raise_for_status = kwargs.get('raise_for_status')
        if raise_for_status is None:
            raise_for_status = getattr(
                orig_self, '_raise_for_status', False
            )
        if raise_for_status:
            response.raise_for_status()

        return response
