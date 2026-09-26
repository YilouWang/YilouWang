"""Browser dashboard: static page + JSON snapshot + Server-Sent Events + commands.

Routes::

    GET  /                 static/index.html (the dashboard, Simplified Chinese)
    GET  /api/health       {"ok": true, ...}
    GET  /api/snapshot     JSON of ``bus.snapshot()`` (latest payload per topic + log)
    GET  /api/events       text/event-stream: first ``event: snapshot``, then every bus
                           event as ``event: <topic>``, keepalive comments every 15 s
    POST /api/command      {"cmd": "...", ...} -> ``bus.command(cmd, **args)``

Security model:
  * localhost mode (host 127.0.0.1 / localhost / ::1): no token, but the Host
    header must be a loopback name (blocks DNS rebinding) and a cross-origin
    ``Origin`` is rejected (blocks CSRF from other web pages).
  * LAN mode (any other host, e.g. 0.0.0.0 to open it from a phone): a random
    token is generated at start and required on every ``/api/*`` request as
    ``?token=...`` or ``X-Token`` header. The printed URL carries the token.

The server only reads the bus and publishes commands; it never touches the game.
"""

from __future__ import annotations

import dataclasses
import json
import math
import queue
import secrets
import socket
import socketserver
import sys
import threading
import time
from datetime import date, datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePath
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

from ..bus import EventBus
from ..config import UIConfig

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"analyze", "scout", "shop", "toggle_auto", "ask", "dismiss", "set_field", "set_comp", "new_game"}
)
MAX_BODY_BYTES = 16 * 1024
KEEPALIVE_S = 15.0
PORT_ATTEMPTS = 11  # the configured port plus the next 10
MAX_SSE_CLIENTS = 32
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_MAX_STR = 500  # longest free-text argument (questions)
_MAX_SHORT_STR = 200
_MAX_EXTRA_ARGS = 8

_FALLBACK_HTML = (
    "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>TFT 实时助手</title></head>"
    "<body style=\"background:#111;color:#eee;font-family:sans-serif\"><h1>TFT 实时助手</h1>"
    "<p>找不到看板页面文件 (ui/static/index.html)，请重新安装。接口 /api/snapshot 仍可用。</p></body></html>"
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert anything the bus may carry into plain JSON types.

    Handles pydantic models, enums, sets/tuples, dataclasses, datetimes, paths,
    bytes, non-string dict keys and non-finite floats (NaN / inf become null,
    because browsers' ``JSON.parse`` rejects them).
    """
    if _depth > 60:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int)):
        return obj.value if isinstance(obj, Enum) else obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return jsonable(obj.value, _depth + 1)
    if isinstance(obj, str):
        return str(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return jsonable(obj.model_dump(mode="json"), _depth + 1)
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, Enum):
                k = k.value
            key = k if isinstance(k, str) else (json.dumps(k) if isinstance(k, (int, float, bool)) or k is None else str(k))
            out[key] = jsonable(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [jsonable(v, _depth + 1) for v in obj]
        try:
            return sorted(items)  # stable output when comparable
        except TypeError:
            return items
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj), _depth + 1)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<{len(obj)} bytes>"
    return str(obj)


def _json_default(obj: Any) -> Any:  # safety net for json.dumps
    return jsonable(obj)


def dumps(obj: Any) -> str:
    """Compact JSON (UTF-8 text, single line) that browsers can always parse."""
    return json.dumps(jsonable(obj), ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False)


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------


class CommandError(ValueError):
    """Invalid command payload; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _is_scalar(v: Any) -> bool:
    if isinstance(v, float):
        return math.isfinite(v)
    return v is None or isinstance(v, (str, int, bool))


def _opt_str(payload: dict[str, Any], key: str, limit: int = _MAX_SHORT_STR) -> Optional[str]:
    v = payload.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise CommandError(f"参数 {key} 必须是文本")
    v = v.strip()[:limit]
    return v or None


def _req_str(payload: dict[str, Any], key: str, limit: int = _MAX_SHORT_STR) -> str:
    v = _opt_str(payload, key, limit)
    if not v:
        raise CommandError(f"缺少参数 {key}")
    return v


def validate_command(payload: Any) -> tuple[str, dict[str, Any]]:
    """Validate a ``POST /api/command`` body. Returns ``(cmd, kwargs)``.

    Known arguments are type checked and trimmed; unknown extra arguments are
    forwarded only when they are short scalars with identifier-like keys.
    """
    if not isinstance(payload, dict):
        raise CommandError("请求体必须是 JSON 对象")
    cmd = payload.get("cmd")
    if not isinstance(cmd, str) or cmd not in ALLOWED_COMMANDS:
        raise CommandError(f"未知命令: {str(cmd)[:40]}")
    args: dict[str, Any] = {}
    if cmd == "scout":
        args["player"] = _opt_str(payload, "player", 64)
    elif cmd == "ask":
        args["question"] = _req_str(payload, "question", _MAX_STR)
    elif cmd == "dismiss":
        args["id"] = _req_str(payload, "id", 128)
    elif cmd == "set_field":
        args["field"] = _req_str(payload, "field", 32)
        if "value" not in payload:
            raise CommandError("缺少参数 value")
        value = payload.get("value")
        if not _is_scalar(value):
            raise CommandError("参数 value 必须是数字或文本")
        args["value"] = value.strip()[:32] if isinstance(value, str) else value
    elif cmd == "set_comp":
        args["comp"] = _opt_str(payload, "comp", _MAX_SHORT_STR) or ""
    extras = 0
    for key, value in payload.items():
        if key in ("cmd", "token") or key in args:
            continue
        if not isinstance(key, str) or not key.isidentifier() or len(key) > 32 or key.startswith("_"):
            continue
        if not _is_scalar(value) or extras >= _MAX_EXTRA_ARGS:
            continue
        args[key] = value[:_MAX_SHORT_STR] if isinstance(value, str) else value
        extras += 1
    return cmd, args


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def is_local_host(host: str) -> bool:
    return (host or "").strip().lower().strip("[]") in LOCAL_HOSTS


def _guess_lan_ip() -> Optional[str]:
    """Best-effort LAN address of this PC (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip != "0.0.0.0":
                return ip
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _host_header_name(value: str) -> str:
    """Hostname part of a Host / Origin netloc (lower case, no port, no brackets)."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    if value.count(":") > 1:  # bare IPv6 without port
        return value
    return value.split(":", 1)[0]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second process bind a port that is already
    # in use, which would defeat the "port busy -> try the next one" logic.
    allow_reuse_address = sys.platform != "win32"
    request_queue_size = 32
    dashboard: "DashboardServer"

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        # HTTPServer.server_bind() also calls socket.getfqdn(), a reverse DNS
        # lookup that can stall startup for seconds on some Windows networks.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        try:
            self.dashboard.log(f"看板请求出错: {exc!r}")
        except Exception:
            pass


class _V6Server(_Server):
    address_family = socket.AF_INET6


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "TFTAdvisor/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.0"  # one request per connection; SSE ends on close
    timeout = 30  # socket timeout: a stuck client cannot pin a thread forever
    server: _Server

    # ---- plumbing ---------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - silence access log
        pass

    @property
    def dash(self) -> "DashboardServer":
        return self.server.dashboard

    def _split(self) -> tuple[str, dict[str, list[str]]]:
        parts = urlsplit(self.path)
        path = parts.path or "/"
        if len(path) > 1:
            path = path.rstrip("/")
        return path, parse_qs(parts.query, keep_blank_values=True)

    def _begin(self, status: int) -> None:
        self._response_started = True
        self.send_response(status)

    def _send_bytes(self, status: int, body: bytes, content_type: str, extra: Optional[dict[str, str]] = None) -> None:
        self._begin(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, obj: Any) -> None:
        self._send_bytes(status, dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    # ---- access control ---------------------------------------------------
    def _host_allowed(self) -> bool:
        """Localhost mode: Host must be a loopback name (DNS rebinding guard)."""
        if self.dash.lan_mode:
            return True
        host = self.headers.get("Host")
        if not host:  # HTTP/1.0 clients may omit it
            return True
        return _host_header_name(host) in LOCAL_HOSTS

    def _origin_allowed(self) -> bool:
        """Reject requests whose Origin is another site (CSRF guard)."""
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True  # same-origin GETs and non-browser clients send no Origin
        if origin == "null":
            return False  # sandboxed iframes, file:// pages
        try:
            netloc = urlsplit(origin).netloc.lower()
        except ValueError:
            return False
        host = (self.headers.get("Host") or "").lower()
        if netloc == host:
            return True
        # Same loopback machine reached via a different loopback name.
        return not self.dash.lan_mode and _host_header_name(netloc) in LOCAL_HOSTS and _host_header_name(host) in LOCAL_HOSTS

    def _token_ok(self, query: dict[str, list[str]]) -> bool:
        expected = self.dash.token
        if not expected:
            return True
        given = self.headers.get("X-Token") or (query.get("token") or [""])[0]
        return bool(given) and secrets.compare_digest(given.encode("utf-8", "replace"), expected.encode("utf-8"))

    def _guard(self, path: str, query: dict[str, list[str]], *, check_origin: bool) -> bool:
        if not self._host_allowed():
            self._error(403, "拒绝访问：Host 不是本机地址")
            return False
        if check_origin and not self._origin_allowed():
            self._error(403, "拒绝访问：跨站请求")
            return False
        if path.startswith("/api/") and not self._token_ok(query):
            self._error(401, "需要令牌：请用启动时打印的完整链接打开看板")
            return False
        return True

    # ---- verbs ----------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self._safely(self._get)

    def do_GET(self) -> None:  # noqa: N802
        self._safely(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._safely(self._post)

    def _safely(self, fn: Callable[[], None]) -> None:
        self._response_started = False
        try:
            fn()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except Exception as exc:  # a bug in one route must not kill the server thread silently
            try:
                self.dash.log(f"看板请求出错: {exc!r}")
            except Exception:
                pass
            if not self._response_started:
                try:
                    self._error(500, "服务器内部错误")
                except Exception:
                    pass
            self.close_connection = True

    def _get(self) -> None:
        path, query = self._split()
        if not self._guard(path, query, check_origin=path.startswith("/api/")):
            return
        if path in ("/", "/index.html"):
            self._send_bytes(
                200,
                self.dash.index_html(),
                "text/html; charset=utf-8",
                {
                    "Content-Security-Policy": (
                        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
                    ),
                    "X-Frame-Options": "DENY",
                },
            )
        elif path == "/api/health":
            self._send_json(200, self.dash.health())
        elif path == "/api/snapshot":
            self._send_json(200, self.dash.bus.snapshot())
        elif path == "/api/events":
            if self.command == "HEAD":
                self._send_bytes(200, b"", "text/event-stream; charset=utf-8")
            else:
                self._stream_events()
        elif path == "/favicon.ico":
            self._begin(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._error(404, "没有这个页面")

    def _post(self) -> None:
        path, query = self._split()
        if not self._guard(path, query, check_origin=True):
            self._drain_body()
            return
        if path != "/api/command":
            self._drain_body()
            self._error(404, "没有这个接口")
            return
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self._error(411, "需要 Content-Length")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._error(400, "Content-Length 无效")
            return
        if length < 0:
            self._error(400, "Content-Length 无效")
            return
        if length > MAX_BODY_BYTES:
            self._drain_body(length)
            self._error(413, f"请求体过大（上限 {MAX_BODY_BYTES // 1024} KB）")
            return
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "JSON 格式错误")
            return
        try:
            cmd, args = validate_command(payload)
        except CommandError as exc:
            self._error(exc.status, str(exc))
            return
        self.dash.bus.command(cmd, **args)
        self._send_json(200, {"ok": True, "cmd": cmd})

    def _drain_body(self, length: Optional[int] = None) -> None:
        """Read (and discard) a request body so closing the socket does not RST the reply."""
        if length is None:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                return
        remaining = min(max(length, 0), 1024 * 1024)
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass

    # ---- Server-Sent Events -------------------------------------------------
    def _stream_events(self) -> None:
        dash = self.dash
        if not dash._sse_enter():
            self._error(503, "连接数过多，请关闭多余的看板页面")
            return
        q = dash.bus.open_queue()
        try:
            self._begin(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self._write("retry: 3000\n\n")
            self._write(_sse_frame("snapshot", dash.bus.snapshot()))
            last_write = time.monotonic()
            while not dash.stopping:
                try:
                    topic, payload = q.get(timeout=min(0.5, dash.keepalive_s))
                except queue.Empty:
                    if time.monotonic() - last_write >= dash.keepalive_s:
                        self._write(": keepalive\n\n")
                        last_write = time.monotonic()
                    continue
                self._write(_sse_frame(topic, payload))
                last_write = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
            pass  # client went away
        finally:
            dash.bus.close_queue(q)
            dash._sse_exit()
            self.close_connection = True

    def _write(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()


def _sse_topic(topic: Any) -> str:
    name = "".join(ch for ch in str(topic) if ch.isalnum() or ch in "_-.")
    return name or "message"


def _sse_frame(topic: Any, payload: Any) -> str:
    try:
        data = dumps(payload)
    except Exception as exc:  # never let one bad payload kill the stream
        data = dumps({"error": f"无法序列化: {exc!r}"})
    # dumps() never emits raw newlines (they are escaped inside JSON strings).
    return f"event: {_sse_topic(topic)}\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# Public server
# ---------------------------------------------------------------------------


class DashboardServer:
    """HTTP dashboard running in a daemon thread.

    ``start()`` returns the URL to open on this PC (with ``?token=`` in LAN
    mode). ``lan_url`` holds the address a phone on the same network can use.
    """

    def __init__(
        self,
        bus: EventBus,
        cfg: UIConfig,
        log: Callable[[str], Any] = print,
        *,
        keepalive_s: float = KEEPALIVE_S,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.log = log
        self.keepalive_s = max(0.05, float(keepalive_s))
        self.lan_mode = not is_local_host(cfg.host)
        self.token: Optional[str] = None
        self.url: Optional[str] = None
        self.lan_url: Optional[str] = None
        self.port: Optional[int] = None
        self._httpd: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._sse_clients = 0
        self._started_at: Optional[float] = None

    # ---- lifecycle ------------------------------------------------------------
    def start(self) -> str:
        if self._httpd is not None and self.url:
            return self.url
        host = (self.cfg.host or "").strip()
        bind_host = host.strip("[]")
        if bind_host.lower() == "localhost":
            bind_host = "127.0.0.1"
        server_cls = _V6Server if ":" in bind_host else _Server
        base_port = int(self.cfg.port or 0)
        ports = [0] if base_port <= 0 else [p for p in range(base_port, base_port + PORT_ATTEMPTS) if p <= 65535]
        httpd: Optional[_Server] = None
        last_exc: Optional[OSError] = None
        for port in ports:
            try:
                httpd = server_cls((bind_host, port), _Handler)
                break
            except OSError as exc:
                last_exc = exc
        if httpd is None:
            raise OSError(
                f"看板端口 {base_port} 到 {ports[-1] if ports else base_port} 都被占用，请在配置里换一个端口 ([ui] port)"
            ) from last_exc
        httpd.dashboard = self
        self._stopping.clear()
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        if base_port > 0 and self.port != base_port:
            self.log(f"端口 {base_port} 已被占用，改用 {self.port}")

        if self.lan_mode:
            self.token = secrets.token_urlsafe(16)
            suffix = f"/?token={self.token}"
            wildcard = bind_host in ("", "0.0.0.0", "::")
            local_host = ("[::1]" if bind_host == "::" else "127.0.0.1") if wildcard else _url_host(bind_host)
            lan_ip = _guess_lan_ip() if wildcard else bind_host
            self.url = f"http://{local_host}:{self.port}{suffix}"
            self.lan_url = f"http://{_url_host(lan_ip)}:{self.port}{suffix}" if lan_ip else self.url
        else:
            self.token = None
            self.url = f"http://{_url_host(bind_host or '127.0.0.1')}:{self.port}/"
            self.lan_url = None

        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="dashboard", daemon=True)
        self._thread.start()
        self._started_at = time.time()
        self.log(f"看板已启动: {self.url}")
        if self.lan_mode:
            self.log(f"手机或局域网访问: {self.lan_url}")
            self.log("令牌只在本次运行有效，不要把链接发给别人。手机打不开时，请在 Windows 防火墙里允许 Python 访问专用网络。")
        return self.url

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._stopping.set()
        if httpd is None:
            return
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        if thread is not None:
            thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None

    # ---- state ------------------------------------------------------------------
    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    @property
    def running(self) -> bool:
        return self._httpd is not None and not self._stopping.is_set()

    @property
    def sse_clients(self) -> int:
        with self._lock:
            return self._sse_clients

    def _sse_enter(self) -> bool:
        with self._lock:
            if self._sse_clients >= MAX_SSE_CLIENTS:
                return False
            self._sse_clients += 1
            return True

    def _sse_exit(self) -> None:
        with self._lock:
            self._sse_clients = max(0, self._sse_clients - 1)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ts": time.time(),
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "lan": self.lan_mode,
            "sse_clients": self.sse_clients,
            "port": self.port,
        }

    def index_html(self) -> bytes:
        try:
            from importlib import resources

            return resources.files("tft_advisor.ui").joinpath("static", "index.html").read_bytes()
        except Exception:
            return _FALLBACK_HTML.encode("utf-8")


__all__ = [
    "ALLOWED_COMMANDS",
    "CommandError",
    "DashboardServer",
    "KEEPALIVE_S",
    "MAX_BODY_BYTES",
    "dumps",
    "is_local_host",
    "jsonable",
    "validate_command",
]
