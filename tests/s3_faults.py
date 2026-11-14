"""S3 fault injection for the instance store tests.

`S3Proxy` is an aiohttp server in the test's own event loop, in front of
moto. Point the store at `proxy.endpoint_url`: each request is classified
(operation, key relative to the test prefix), logged with what the client got,
and rules can fail, drop, hold, edit or hook it.

The proxy gives each request to moto's WSGI app in this process (in a worker
thread), as moto's HTTP server would: that server opens a new connection and a
new thread for each request, because it closes every connection after its answer.

Conditional PUTs (If-Match / If-None-Match) on one key are forwarded one at a
time. moto checks the condition, then writes, with no lock between the two:
two concurrent conditional PUTs on one key could both win on bare moto, while
a real store lets only one win.

`Bucket` reads and writes moto directly, never through the proxy: what the
bucket expiry rule, an operator or another bucket writer does.

`moto_server` serves moto's S3 API over HTTP for the clients that need an endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import re
import socketserver
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote, urlencode

from aiohttp import web
from moto.moto_server.werkzeug_app import DomainDispatcherApplication, MotoRequestHandler, create_backend_app
from multidict import CIMultiDict, MultiDict
from werkzeug.serving import ThreadedWSGIServer
from werkzeug.test import EnvironBuilder, run_wsgi_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Iterator
    from wsgiref.types import WSGIApplication, WSGIEnvironment

    from werkzeug.datastructures import Headers

BUCKET = "test-snapshots"

OPS = frozenset(
    {
        "HEAD",
        "GET",  # whole object (the head, a body object of one part)
        "GET_RANGE",  # ranged GET (larger body objects, assets)
        "PUT",  # unconditional single PUT (body objects, assets)
        "PUT_IF",  # conditional PUT (If-Match / If-None-Match: the head)
        "MPU_CREATE",
        "MPU_PART",
        "MPU_COMPLETE",
        "MPU_ABORT",
        "LIST",
        "DELETE",
        "DELETE_MANY",
        "OTHER",
    }
)
_HOP_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "expect", "upgrade", "te"}
)
_DROPPED_RESPONSE_HEADERS = _HOP_HEADERS | {"server", "date"}
_MAX_BODY = 256 * 1024 * 1024


class MotoServer(ThreadedWSGIServer):
    """The HTTP server of moto.server.ThreadedMotoServer, bound with no name lookup:
    http.server looks up the name of the bind address (socket.getfqdn), which can block for
    more than 10 s on each xdist worker of the macOS runners. The server name is the address."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


@contextlib.contextmanager
def moto_server(host: str) -> Iterator[MotoServer]:
    """moto's S3 API on `host` and a free port, served by a thread until the block ends."""
    server = MotoServer(host, 0, DomainDispatcherApplication(create_backend_app), handler=MotoRequestHandler)
    thread = threading.Thread(target=server.serve_forever, name="moto-server", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class LogEntry(NamedTuple):
    """One request as the client saw it."""

    op: str
    key: str
    """Key relative to the test prefix (LIST: the asked prefix; DELETE_MANY: the keys, space-separated)."""
    status: int
    """HTTP status the client got; 0 when the proxy dropped the connection without a full answer."""


@dataclass(slots=True)
class Request:
    """A request as the rules see it (Edit.request may change method-independent parts)."""

    op: str
    key: str
    method: str
    path: str
    query: MultiDict[str]
    headers: CIMultiDict[str]
    body: bytes


@dataclass(slots=True)
class Response:
    """The upstream answer as the rules see it (Edit.response may change it)."""

    status: int
    headers: CIMultiDict[str]
    body: bytes


# ----------------------------------------------------------------------------- actions


@dataclass(frozen=True, slots=True)
class Fail:
    """Answer with this status and S3 error code; the request never reaches moto. HEAD gets no body."""

    status: int
    code: str


@dataclass(frozen=True, slots=True)
class DropBefore:
    """Close the connection without forwarding: the store never saw the request."""


@dataclass(frozen=True, slots=True)
class DropAfter:
    """Forward (moto applies it), then close the connection before the answer: an unknown outcome."""


@dataclass(frozen=True, slots=True)
class Hold:
    """Wait until the rule is released, then go on (the rule's `reached` is set when a request waits)."""


@dataclass(frozen=True, slots=True)
class Truncate:
    """Forward, send the headers (full Content-Length) and only `keep` body bytes, then close."""

    keep: int


@dataclass(frozen=True, slots=True)
class Edit:
    """Change the request before it is forwarded and/or the answer before it is sent."""

    request: Callable[[Request], None] | None = None
    response: Callable[[Response], None] | None = None


@dataclass(frozen=True, slots=True)
class Before:
    """Run `hook` before the request is forwarded (e.g. damage the object it reads)."""

    hook: Callable[[Request], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class After:
    """Run `hook` after moto answered, before the client gets the answer."""

    hook: Callable[[Request, Response], Awaitable[None]]


type Action = Fail | DropBefore | DropAfter | Hold | Truncate | Edit | Before | After


class Rule:
    """A fault rule and its handle: counts, the `reached` event, `release()` for Hold, `remove()`."""

    def __init__(
        self,
        ops: frozenset[str],
        key: str,
        action: Action,
        *,
        times: int | None,
        skip: int,
        when: Callable[[Request], bool] | None,
    ) -> None:
        unknown = ops - OPS
        if unknown:
            raise ValueError(f"unknown S3 operations: {sorted(unknown)}")
        self.ops = ops
        self.key = re.compile(key)
        self.action = action
        self.times = times
        self.skip = skip
        self.when = when
        self.matched = 0
        """Requests whose operation and key matched (skipped ones included)."""
        self.hits = 0
        """Requests the action applied to."""
        self.reached = asyncio.Event()
        """Set when the action applies the first time (for Hold: a request is waiting)."""
        self._released = asyncio.Event()
        self._active = True

    def release(self) -> None:
        """Let held requests go (now and later)."""
        self._released.set()

    def remove(self) -> None:
        """Stop applying this rule, and let held requests go."""
        self._active = False
        self._released.set()

    def _take(self, req: Request) -> bool:
        if not self._active or req.op not in self.ops or not self.key.search(req.key):
            return False
        if self.when is not None and not self.when(req):
            return False
        self.matched += 1
        if self.matched <= self.skip or (self.times is not None and self.hits >= self.times):
            return False
        self.hits += 1
        self.reached.set()
        return True


class S3Proxy:
    """Fault-injecting proxy in front of moto. Start with `S3Proxy.start` (the `s3_proxy` fixture does)."""

    def __init__(self, *, moto: WSGIApplication, moto_url: str, bucket: Bucket, pool: ThreadPoolExecutor) -> None:
        self.bucket = bucket
        self.prefix = bucket.prefix
        self.endpoint_url = ""
        self.log: list[LogEntry] = []
        self.rules: list[Rule] = []
        self._moto = moto
        self._moto_url = moto_url
        self._pool = pool
        self._key_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    @contextlib.asynccontextmanager
    async def start(cls, *, moto: WSGIApplication, moto_url: str, bucket: Bucket) -> AsyncGenerator[S3Proxy]:
        """Listen on 127.0.0.1 (random port) in the running loop until the block ends. Requests go
        to `moto` (moto's WSGI app) as if they reached moto's server at `moto_url`."""
        with ThreadPoolExecutor(thread_name_prefix="moto") as pool:
            proxy = cls(moto=moto, moto_url=moto_url, bucket=bucket, pool=pool)
            app = web.Application(client_max_size=_MAX_BODY)
            app.router.add_route("*", "/{tail:.*}", proxy._handle)
            runner = web.AppRunner(app, access_log=None, handle_signals=False)
            await runner.setup()
            try:
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                host, port = runner.addresses[0][:2]
                proxy.endpoint_url = f"http://{host}:{port}"
                yield proxy
            finally:
                for rule in proxy.rules:
                    rule.remove()  # never leave a request held at teardown
                await runner.cleanup()

    # ------------------------------------------------------------------ rules

    def add(
        self,
        op: str | Iterable[str],
        key: str,
        action: Action,
        *,
        times: int | None = 1,
        skip: int = 0,
        when: Callable[[Request], bool] | None = None,
    ) -> Rule:
        """Apply `action` to requests of `op` (one or several) whose key matches the regex `key`
        (re.search) and, when given, `when(request)`; after `skip` matches, at most `times` times
        (None: always). aiohttp sends an idempotent request (GET, HEAD, PUT, DELETE) once more
        after a dropped connection: a drop that must leave a call without an answer uses times=None."""
        ops = frozenset({op} if isinstance(op, str) else op)
        rule = Rule(ops, key, action, times=times, skip=skip, when=when)
        self.rules.append(rule)
        return rule

    def clear(self) -> None:
        """Remove every rule (held requests go on)."""
        for rule in self.rules:
            rule.remove()
        self.rules.clear()

    # ------------------------------------------------------------------ log

    def entries(self, op: str | Iterable[str] | None = None, key: str = "", *, since: int = 0) -> list[LogEntry]:
        """Log entries from index `since` with this operation (or any) and a key matching `key`."""
        ops = None if op is None else frozenset({op} if isinstance(op, str) else op)
        pattern = re.compile(key)
        return [e for e in self.log[since:] if (ops is None or e.op in ops) and pattern.search(e.key)]

    def count(self, op: str | Iterable[str] | None = None, key: str = "", *, since: int = 0) -> int:
        return len(self.entries(op, key, since=since))

    # ------------------------------------------------------------------ serving

    async def _handle(self, http: web.Request) -> web.StreamResponse:
        body = await http.read()
        headers = CIMultiDict(http.headers)
        query: MultiDict[str] = MultiDict(http.query)
        op, key = self._classify(http.method, http.path, query, headers, body)
        req = Request(op=op, key=key, method=http.method, path=http.path, query=query, headers=headers, body=body)
        original_query = MultiDict(query)
        fired = [rule for rule in list(self.rules) if rule._take(req)]

        await _before(fired, req)
        for rule in fired:
            action = rule.action
            if isinstance(action, Fail):
                self.log.append(LogEntry(op, key, action.status))
                return _error_response(http.method, action.status, action.code)
            if isinstance(action, DropBefore):
                self.log.append(LogEntry(op, key, 0))
                return _drop(http)

        if op == "PUT_IF":
            async with self._key_locks.setdefault(key, asyncio.Lock()):
                resp = await self._forward(http, req, query_changed=req.query != original_query)
        else:
            resp = await self._forward(http, req, query_changed=req.query != original_query)

        await _after(fired, req, resp)
        for rule in fired:
            action = rule.action
            if isinstance(action, DropAfter):
                self.log.append(LogEntry(op, key, 0))
                return _drop(http)
            if isinstance(action, Truncate):
                self.log.append(LogEntry(op, key, 0))
                return await _truncated(http, resp, action.keep)

        self.log.append(LogEntry(op, key, resp.status))
        return _response(http.method, resp)

    def _classify(
        self, method: str, path: str, query: MultiDict[str], headers: CIMultiDict[str], body: bytes
    ) -> tuple[str, str]:
        bucket_path = f"/{self.bucket.name}"
        if path in (bucket_path, f"{bucket_path}/"):
            if method == "GET" and "list-type" in query:
                return "LIST", self._relative(query.get("prefix", ""))
            if method == "POST" and "delete" in query:
                keys = [self._relative(html.unescape(k)) for k in re.findall(r"<Key>(.*?)</Key>", body.decode())]
                return "DELETE_MANY", " ".join(keys)
            return "OTHER", ""
        key = self._relative(path.removeprefix(f"{bucket_path}/"))
        match method:
            case "HEAD":
                return "HEAD", key
            case "GET":
                return ("GET_RANGE" if "Range" in headers else "GET"), key
            case "PUT" if "uploadId" in query:
                return "MPU_PART", key
            case "PUT":
                return ("PUT_IF" if "If-Match" in headers or "If-None-Match" in headers else "PUT"), key
            case "POST" if "uploads" in query:
                return "MPU_CREATE", key
            case "POST" if "uploadId" in query:
                return "MPU_COMPLETE", key
            case "DELETE":
                return ("MPU_ABORT" if "uploadId" in query else "DELETE"), key
            case _:
                return "OTHER", key

    def _relative(self, full_key: str) -> str:
        return full_key.removeprefix(self.prefix)

    async def _forward(self, http: web.Request, req: Request, *, query_changed: bool) -> Response:
        """moto's answer: the request as moto's server would hand it to its WSGI app (the path as
        sent, still encoded; the query as sent unless a rule changed it)."""
        path, _, query = http.raw_path.partition("?")
        if query_changed:
            query = urlencode(list(req.query.items()), quote_via=quote)
        environ = EnvironBuilder(
            path=path,
            base_url=self._moto_url,
            query_string=query,
            method=req.method,
            headers=[(k, v) for k, v in req.headers.items() if k.lower() not in _HOP_HEADERS],
            input_stream=io.BytesIO(req.body) if req.method in ("PUT", "POST") else None,
        ).get_environ()
        loop = asyncio.get_running_loop()
        status, headers, body = await loop.run_in_executor(self._pool, _wsgi_answer, self._moto, environ)
        return Response(status=status, headers=CIMultiDict(headers.items()), body=body)


def _wsgi_answer(app: WSGIApplication, environ: WSGIEnvironment) -> tuple[int, Headers, bytes]:
    chunks, status, headers = run_wsgi_app(app, environ, buffered=True)
    return int(status.split(" ", 1)[0]), headers, b"".join(chunks)


async def _before(fired: list[Rule], req: Request) -> None:
    """Hold, Before hooks and request edits, in rule order, before anything is forwarded."""
    for rule in fired:
        action = rule.action
        if isinstance(action, Hold):
            await rule._released.wait()
        elif isinstance(action, Before):
            await action.hook(req)
        elif isinstance(action, Edit) and action.request is not None:
            action.request(req)


async def _after(fired: list[Rule], req: Request, resp: Response) -> None:
    """Answer edits and After hooks, in rule order, once moto answered."""
    upstream_body = resp.body
    for rule in fired:
        action = rule.action
        if isinstance(action, Edit) and action.response is not None:
            action.response(resp)
        elif isinstance(action, After):
            await action.hook(req, resp)
    if resp.body != upstream_body:
        # A changed body no longer matches the checksums moto computed for the stored bytes.
        for name in [h for h in resp.headers if h.lower().startswith("x-amz-checksum-") or h.lower() == "content-md5"]:
            resp.headers.popall(name, None)


def _response(method: str, resp: Response) -> web.StreamResponse:
    headers = CIMultiDict((k, v) for k, v in resp.headers.items() if k.lower() not in _DROPPED_RESPONSE_HEADERS)
    if method == "HEAD":
        # No body, but the object's size: aiohttp keeps an explicit Content-Length on HEAD.
        length = resp.headers.get("Content-Length")
        if length is not None:
            headers["Content-Length"] = length
        return web.Response(status=resp.status, headers=headers)
    return web.Response(status=resp.status, headers=headers, body=resp.body)


def _error_response(method: str, status: int, code: str) -> web.StreamResponse:
    if method == "HEAD":
        return web.Response(status=status)
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<Error><Code>{html.escape(code)}</Code><Message>injected by the test proxy</Message>"
        "<RequestId>s3-proxy</RequestId></Error>"
    ).encode()
    return web.Response(status=status, body=body, content_type="application/xml")


def _drop(http: web.Request) -> web.StreamResponse:
    """Close the client's connection; aiohttp then fails to send the returned answer (and ignores that)."""
    if http.transport is not None:
        http.transport.abort()
    return web.Response(status=599)


async def _truncated(http: web.Request, resp: Response, keep: int) -> web.StreamResponse:
    headers = CIMultiDict((k, v) for k, v in resp.headers.items() if k.lower() not in _DROPPED_RESPONSE_HEADERS)
    stream = web.StreamResponse(status=resp.status, headers=headers)
    stream.content_length = len(resp.body)
    await stream.prepare(http)
    await stream.write(resp.body[:keep])
    if http.transport is not None:
        http.transport.close()  # flushes the bytes written so far, then closes
    return stream


# ----------------------------------------------------------------------------- direct bucket access


@dataclass(slots=True)
class Bucket:
    """The moto bucket under one prefix, read and written directly (sync boto3 client, keys relative to the prefix)."""

    s3: Any
    prefix: str = field(default_factory=lambda: f"inst-{uuid.uuid4().hex}/")
    name: str = BUCKET

    def listing(self, under: str = "") -> list[str]:
        """Every key under `under` (relative to the prefix), sorted."""
        found: list[str] = []
        token: str | None = None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            page = self.s3.list_objects_v2(Bucket=self.name, Prefix=f"{self.prefix}{under}", **extra)
            found.extend(o["Key"].removeprefix(self.prefix) for o in page.get("Contents", []))
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated") or not token:
                return sorted(found)

    def body_keys(self, instance_id: str) -> list[str]:
        """Body object keys of one instance (s/<id>/b/...), sorted."""
        return self.listing(f"s/{instance_id}/b/")

    def read(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.name, Key=f"{self.prefix}{key}")["Body"].read()

    def write(self, key: str, data: bytes) -> str:
        """Unconditional PUT; returns the new ETag (quotes removed)."""
        return self.s3.put_object(Bucket=self.name, Key=f"{self.prefix}{key}", Body=data)["ETag"].strip('"')

    def size(self, key: str) -> int:
        return int(self.s3.head_object(Bucket=self.name, Key=f"{self.prefix}{key}")["ContentLength"])

    def etag(self, key: str) -> str:
        return self.s3.head_object(Bucket=self.name, Key=f"{self.prefix}{key}")["ETag"].strip('"')

    def exists(self, key: str) -> bool:
        return key in self.listing(key)

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.name, Key=f"{self.prefix}{key}")

    def flip(self, key: str, offset: int) -> None:
        """Change one bit at `offset` (negative: from the end); the size stays."""
        data = bytearray(self.read(key))
        data[offset] ^= 0x01
        self.write(key, bytes(data))

    def truncate(self, key: str, size: int) -> None:
        """Keep the first `size` bytes."""
        self.write(key, self.read(key)[:size])

    def append(self, key: str, data: bytes) -> None:
        self.write(key, self.read(key) + data)

    def swap(self, src: str, dst: str) -> None:
        """Put the bytes of `src` at `dst` (dst gets src's ETag; src stays)."""
        self.write(dst, self.read(src))

    def index_offset(self, key: str) -> int:
        """Offset of a body object's sealed index, from its trailer (u64 at size - 20)."""
        trailer = self.read(key)[-20:]
        return int.from_bytes(trailer[:8], "big")
