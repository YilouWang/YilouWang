"""Browser dashboard: static page + JSON snapshot + Server-Sent Events + commands.

Routes::

    GET  /                 static/index.html (the dashboard, Simplified Chinese)
    GET  /api/health       {"ok": true, ...}
    GET  /api/snapshot     JSON of ``bus.snapshot()`` (latest payload per topic + log)
    GET  /api/events       text/event-stream: first ``event: snapshot``, then every bus
                           event as ``event: <topic>``, keepalive comments every 15 s
                           (each followed by a tiny ``event: ping`` for the page's
                           dead-stream watchdog). A client too slow to keep up gets a
                           fresh ``snapshot`` instead of the events it would lose.
    POST /api/command      {"cmd": "...", ...} -> ``bus.command(cmd, **args)``

Security model:
  * localhost mode (host 127.0.0.1 / localhost / ::1): no token, but the Host
    header must be a loopback name (blocks DNS rebinding) and an ``Origin``
    other than the dashboard itself (scheme, loopback host AND port) is
    rejected, so pages served from other local ports (dev servers, Jupyter)
    cannot send commands either. ``Sec-Fetch-Site`` other than same-origin /
    none is rejected on ``/api/*``.
  * LAN mode (any other host, e.g. 0.0.0.0 to open it from a phone): a random
    token is required on every ``/api/*`` request as ``?token=...`` or
    ``X-Token`` header. The printed URL carries the token. The token is saved
    in ``token_file`` (default ``~/.tft_advisor/dashboard_token``) so a phone
    bookmark keeps working across runs; delete the file (or pass
    ``new_token=True``) to rotate it.
  * ``POST /api/command`` needs ``Content-Type: application/json`` (a "simple"
    cross-site request cannot send that without a CORS preflight, which this
    server never approves).
  * Connections are capped (in total and per LAN client) and the request line,
    headers and body must arrive within a total deadline, so a slow client on
    the Wi-Fi cannot pin an unbounded number of threads.

``POST /api/command`` answers ``{"ok": false, "rejected": true, "error": ...}``
(HTTP 400) when the app refused the command (for example a manual correction
out of range), and passes along warnings the app logged while handling it.

The server only reads the bus and publishes commands; it never touches the game.
"""

from __future__ import annotations

import dataclasses
import errno
import io
import ipaddress
import json
import math
import os
import queue
import re
import secrets
import socket
import socketserver
import sys
import threading
import time
from datetime import date, datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePath
from typing import Any, Callable, Optional, Union
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
MAX_CONNECTIONS = 64  # handler threads in total (SSE streams included)
MAX_CONNECTIONS_PER_IP = 16  # per non-loopback client (one phone needs a handful)
HEADER_TIMEOUT_S = 10.0  # total time for the request line + headers
BODY_TIMEOUT_S = 10.0  # total time for a (small) request body after the headers
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_TOKEN_FILE = "~/.tft_advisor/dashboard_token"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{22,128}")
_JSON_TYPES = frozenset({"application/json"})
_SAME_ORIGIN_FETCH = frozenset({"same-origin", "none"})
# Commands the app handles synchronously and purely (validation only): any
# warning it logs while handling them means the command was refused.
_STRICT_COMMANDS = frozenset({"set_field"})
# How the app words a refused command ("命令 set_field 失败: ...", "修正失败：...").
_FAILED_PREFIX_RE = re.compile(r"^(?:命令\s*\S+\s*失败|修正失败)\s*[:：]\s*")

_MAX_STR = 500  # longest free-text argument (questions)
_MAX_SHORT_STR = 200
_MAX_EXTRA_ARGS = 8
_MAX_ABS_NUMBER = 10**9  # manual corrections are small numbers; bigger ones are garbage
# Keyword names that ``EventBus.command(self, cmd, **kwargs)`` cannot take as extras.
_RESERVED_ARGS = frozenset({"cmd", "token", "self"})
# errno values meaning "this address does not exist here" (retrying other ports is pointless).
_ADDR_NOT_AVAILABLE = frozenset({errno.EADDRNOTAVAIL, 10049})

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
    # numpy scalars (np.int64, np.bool_, ...) and arrays, without importing numpy.
    if type(obj).__module__ == "numpy":
        try:
            if hasattr(obj, "tolist"):
                return jsonable(obj.tolist(), _depth + 1)
            return jsonable(obj.item(), _depth + 1)
        except Exception:
            return str(obj)
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
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v) and abs(v) <= _MAX_ABS_NUMBER
    return False


# C0/C1 controls (terminal escapes), bidi overrides; newlines only where allowed.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def _clean_str(v: str, multiline: bool = False) -> str:
    if not multiline:
        v = v.replace("\n", " ").replace("\t", " ")
    return _CONTROL_CHARS.sub("", v.replace("\r", ""))


def _opt_str(payload: dict[str, Any], key: str, limit: int = _MAX_SHORT_STR, multiline: bool = False) -> Optional[str]:
    v = payload.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise CommandError(f"参数 {key} 必须是文本")
    v = _clean_str(v, multiline).strip()[:limit]
    return v or None


def _req_str(payload: dict[str, Any], key: str, limit: int = _MAX_SHORT_STR, multiline: bool = False) -> str:
    v = _opt_str(payload, key, limit, multiline)
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
        args["question"] = _req_str(payload, "question", _MAX_STR, multiline=True)
    elif cmd == "dismiss":
        args["id"] = _req_str(payload, "id", 128)
    elif cmd == "set_field":
        args["field"] = _req_str(payload, "field", 32)
        if "value" not in payload:
            raise CommandError("缺少参数 value")
        value = payload.get("value")
        if not _is_scalar(value):
            raise CommandError("参数 value 必须是数字或文本")
        args["value"] = _clean_str(value).strip()[:32] if isinstance(value, str) else value
    elif cmd == "set_comp":
        args["comp"] = _opt_str(payload, "comp", _MAX_SHORT_STR) or ""
    extras = 0
    for key, value in payload.items():
        if key in _RESERVED_ARGS or key in args:
            continue
        if not isinstance(key, str) or not key.isidentifier() or len(key) > 32 or key.startswith("_"):
            continue
        if not _is_scalar(value) or extras >= _MAX_EXTRA_ARGS:
            continue
        args[key] = _clean_str(value)[:_MAX_SHORT_STR] if isinstance(value, str) else value
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


def _is_loopback_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(str(ip).split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return bool(addr.is_loopback or (mapped is not None and mapped.is_loopback))


def load_or_create_token(path: Union[str, "os.PathLike[str]"], *, new: bool = False, log: Callable[[str], Any] = print) -> str:
    """The LAN token saved in ``path``; created (0600) when missing, invalid or ``new``.

    Falls back to a token for this run only when the file cannot be written.
    """
    file = Path(os.path.expanduser(str(path)))
    if not new:
        try:
            saved = file.read_text(encoding="utf-8").strip()
            if _TOKEN_RE.fullmatch(saved):
                return saved
        except (OSError, UnicodeDecodeError):
            pass
    token = secrets.token_urlsafe(16)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError as exc:
        log(f"看板令牌无法保存到 {file}（{exc}），本次运行使用临时令牌")
    return token


class _DeadlineReader(socket.SocketIO):
    """Raw socket reader that enforces the handler's *total* read deadline.

    A plain socket timeout applies to each ``recv`` only, so a client sending
    one header byte every few seconds could hold its thread forever.
    """

    def __init__(self, sock: socket.socket, handler: "_Handler") -> None:
        super().__init__(sock, "rb")
        self._handler = handler

    def readinto(self, b: Any) -> Optional[int]:
        deadline = self._handler._read_deadline
        if deadline is None:
            return super().readinto(b)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request too slow")
        normal = self._handler.timeout
        sock = self._sock  # type: ignore[attr-defined]
        sock.settimeout(remaining if normal is None else min(remaining, normal))
        try:
            return super().readinto(b)
        finally:
            try:
                sock.settimeout(normal)  # writes (SSE) keep the normal timeout
            except OSError:
                pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second process bind a port that is already
    # in use, which would defeat the "port busy -> try the next one" logic.
    allow_reuse_address = sys.platform != "win32"
    request_queue_size = 32
    dashboard: "DashboardServer"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._conn_lock = threading.Lock()
        self._conn_total = 0
        self._conn_by_ip: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    # ---- connection caps: one thread per connection, so bound them ---------
    def _admit(self, client_address: Any) -> bool:
        ip = str(client_address[0]) if isinstance(client_address, (tuple, list)) and client_address else ""
        per_ip_cap = not _is_loopback_ip(ip)  # the PC's own browser is trusted
        with self._conn_lock:
            if self._conn_total >= MAX_CONNECTIONS:
                return False
            if per_ip_cap and self._conn_by_ip.get(ip, 0) >= MAX_CONNECTIONS_PER_IP:
                return False
            self._conn_total += 1
            self._conn_by_ip[ip] = self._conn_by_ip.get(ip, 0) + 1
            return True

    def _release(self, client_address: Any) -> None:
        ip = str(client_address[0]) if isinstance(client_address, (tuple, list)) and client_address else ""
        with self._conn_lock:
            self._conn_total = max(0, self._conn_total - 1)
            left = self._conn_by_ip.get(ip, 0) - 1
            if left > 0:
                self._conn_by_ip[ip] = left
            else:
                self._conn_by_ip.pop(ip, None)

    @property
    def open_connections(self) -> int:
        with self._conn_lock:
            return self._conn_total

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._admit(client_address):
            self.shutdown_request(request)  # over the cap: drop it without a thread
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release(client_address)  # the thread never started
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(client_address)

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

    _read_deadline: Optional[float] = None

    # ---- plumbing ---------------------------------------------------------
    def setup(self) -> None:
        self._read_deadline = time.monotonic() + HEADER_TIMEOUT_S
        super().setup()
        try:
            reader = io.BufferedReader(_DeadlineReader(self.connection, self))
        except Exception:  # pragma: no cover - keep the stock reader (per-recv timeout only)
            return
        self.rfile.close()  # only drops the socket's makefile reference
        self.rfile = reader

    def parse_request(self) -> bool:
        ok = super().parse_request()
        # Headers are in; a request body (<= 16 KB) gets its own short deadline.
        self._read_deadline = time.monotonic() + BODY_TIMEOUT_S
        return ok

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
        """Reject requests from any page but the dashboard itself (CSRF guard)."""
        fetch_site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if fetch_site and fetch_site not in _SAME_ORIGIN_FETCH:
            return False  # cross-site, or same-site from another port / subdomain
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True  # same-origin GETs and non-browser clients send no Origin
        if origin == "null":
            return False  # sandboxed iframes, file:// pages
        try:
            parts = urlsplit(origin)
            netloc = parts.netloc.lower()
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            return False
        host = (self.headers.get("Host") or "").lower()
        if self.dash.lan_mode:
            return netloc == host
        # Localhost mode: the dashboard itself (any loopback name), but only on
        # its own scheme and port: another local page (dev server, Jupyter, a
        # downloaded HTML file served on some port) must not send commands.
        return (
            parts.scheme == "http"
            and _host_header_name(netloc) in LOCAL_HOSTS
            and _host_header_name(host) in LOCAL_HOSTS
            and port == self.dash.port
        )

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
        # Forces a CORS preflight for cross-site pages (text/plain bodies need none).
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype not in _JSON_TYPES:
            self._drain_body()
            self._error(415, "请求类型必须是 application/json")
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
        except (ValueError, RecursionError):  # bad UTF-8 / JSON, huge numbers, deep nesting
            self._error(400, "JSON 格式错误")
            return
        try:
            cmd, args = validate_command(payload)
        except CommandError as exc:
            self._error(exc.status, str(exc))
            return
        error, warnings = self.dash.run_command(cmd, args)
        if error:
            self._send_json(400, {"ok": False, "rejected": True, "cmd": cmd, "error": error})
            return
        reply: dict[str, Any] = {"ok": True, "cmd": cmd}
        if warnings:
            reply["warnings"] = warnings
        self._send_json(200, reply)

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
                if q.full():
                    # The bus drops events for a full queue (slow client, e.g. a
                    # sleeping phone). Newer payloads may be lost, so throw the
                    # backlog away and resend the whole current state instead.
                    _drain(q)
                    self._write(_sse_frame("snapshot", dash.bus.snapshot()))
                    last_write = time.monotonic()
                    continue
                try:
                    topic, payload = q.get(timeout=min(0.5, dash.keepalive_s))
                except queue.Empty:
                    if time.monotonic() - last_write >= dash.keepalive_s:
                        # The comment keeps proxies / NAT from closing the idle stream;
                        # the "ping" event lets the page notice a dead stream (EventSource
                        # hides comments from JavaScript).
                        self._write(": keepalive\n\n" + _sse_frame("ping", {"ts": time.time()}))
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


def _drain(q: "queue.Queue[Any]") -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


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

    ``token_file``: where the LAN token is kept between runs (``None`` = a new
    token every run). ``new_token=True`` replaces the saved token (rotation).
    """

    def __init__(
        self,
        bus: EventBus,
        cfg: UIConfig,
        log: Callable[[str], Any] = print,
        *,
        keepalive_s: float = KEEPALIVE_S,
        token_file: Optional[Union[str, "os.PathLike[str]"]] = DEFAULT_TOKEN_FILE,
        new_token: bool = False,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.log = log
        self.keepalive_s = max(0.05, float(keepalive_s))
        self.lan_mode = not is_local_host(cfg.host)
        self.token_file = token_file
        self.new_token = new_token
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
                if isinstance(exc, socket.gaierror) or exc.errno in _ADDR_NOT_AVAILABLE:
                    break  # the host itself is wrong: other ports will not help
        if httpd is None:
            if last_exc is not None and (isinstance(last_exc, socket.gaierror) or last_exc.errno in _ADDR_NOT_AVAILABLE):
                raise OSError(
                    f"看板无法监听地址 {host or '0.0.0.0'}（{last_exc}），请检查配置 [ui] host，"
                    "本机用 127.0.0.1，手机访问用 0.0.0.0"
                ) from last_exc
            raise OSError(
                f"看板端口 {base_port} 到 {ports[-1] if ports else base_port} 都无法使用（{last_exc}），"
                "请在配置里换一个端口 ([ui] port)"
            ) from last_exc
        httpd.dashboard = self
        self._stopping.clear()
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        if base_port > 0 and self.port != base_port:
            self.log(f"端口 {base_port} 已被占用，改用 {self.port}")

        if self.lan_mode:
            if self.token_file:
                self.token = load_or_create_token(self.token_file, new=self.new_token, log=self.log)
            else:
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
            if self.token_file:
                where = os.path.expanduser(str(self.token_file))
                self.log(f"令牌保存在 {where}，以后启动不变，手机可以直接收藏这个链接；想换新令牌就删除这个文件再启动。不要把链接发给别人。")
            else:
                self.log("令牌只在本次运行有效，不要把链接发给别人。")
            self.log("手机打不开时：Windows 防火墙要允许 Python（需要管理员），Wi-Fi 设成「专用网络」，关掉 VPN 或加速器（详见 README「手机看板打不开」）。")
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

    def run_command(self, cmd: str, args: dict[str, Any]) -> tuple[Optional[str], list[str]]:
        """Publish a command and report what the app said about it: ``(error, warnings)``.

        ``EventBus.publish`` runs handlers synchronously in this thread, so a
        warning the app logs from this thread while handling the command belongs
        to it. The app reports a failed command as ``命令 <cmd> 失败: <reason>``;
        for ``set_field`` (pure validation) any warning means it was refused.
        Other warnings (for example a failed screenshot before an analysis, or
        a hint about the target comp) are passed along; the command still ran.
        """
        me = threading.get_ident()
        said: list[str] = []

        def capture(_topic: str, payload: Any) -> None:
            if threading.get_ident() != me or not isinstance(payload, dict):
                return
            if payload.get("level") in ("warn", "error"):
                said.append(str(payload.get("text") or "").strip())

        unsubscribe = self.bus.subscribe("log", capture)
        try:
            self.bus.command(cmd, **args)
        finally:
            unsubscribe()
        said = [s for s in said if s]
        failed = [s for s in said if _FAILED_PREFIX_RE.match(s)]
        if failed or (said and cmd in _STRICT_COMMANDS):
            reason = _FAILED_PREFIX_RE.sub("", (failed or said)[0], count=1).strip()
            return (reason or "命令未执行")[:_MAX_SHORT_STR], []
        return None, [s[:_MAX_SHORT_STR] for s in said[:3]]

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
    "load_or_create_token",
    "validate_command",
]
