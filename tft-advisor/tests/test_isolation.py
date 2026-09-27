"""The suite is hermetic by construction (tests/conftest.py ``isolation``).

It runs on the user's Windows gaming PC, where ~/.tft_advisor holds real game
logs, ANTHROPIC_API_KEY is set and a match may be serving the Live Client API.
"""

from __future__ import annotations

import errno
import os
import socket
from pathlib import Path

import pytest

from tft_advisor.config import Config, load_config
from tft_advisor.llm import has_credentials
from tft_advisor.vision.liveclient import LiveClient

from .conftest import _PROXY_VARS, LIVECLIENT_PORT

# Collected before any fixture runs: the environment pytest was started with.
_STARTING_HOME = Path(os.path.expanduser("~")).resolve()


def _same(a: object, b: object) -> bool:
    return Path(str(a)).resolve() == Path(str(b)).resolve()


def test_home_config_and_credentials_are_isolated(isolation, tmp_path):
    home = isolation.home
    for var in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):  # Windows expanduser reads USERPROFILE
        assert _same(os.environ[var], home), var
    assert _same(os.path.expanduser("~"), home) and _same(Path.home(), home) and _same(Path.cwd(), home)
    assert not _same(home, _STARTING_HOME)
    assert not _same(home, tmp_path)  # tests counting files in tmp_path stay exact
    assert Config().cache_dir.resolve().is_relative_to(home.resolve())  # game logs / review history
    assert load_config().source_path is None
    assert not has_credentials()
    assert [k for k in os.environ if k.startswith(("ANTHROPIC_", "TFT_ADVISOR_")) or k in _PROXY_VARS] == []


def test_network_egress_is_blocked_and_fails_the_test(isolation):
    for connect in ("connect", "connect_ex"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if connect == "connect":
                with pytest.raises(OSError, match="egress blocked"):
                    s.connect(("192.0.2.1", 443))  # TEST-NET-1
            else:
                assert s.connect_ex(("192.0.2.1", 443)) == errno.ENETUNREACH
    assert isolation.egress == [("192.0.2.1", 443)] * 2
    isolation.egress.clear()  # expected here; anywhere else it fails the test at teardown


def test_loopback_and_udp_connects_still_work(isolation):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        with socket.create_connection(srv.getsockname(), timeout=1):
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:  # the dashboard's LAN IP guess
        udp.connect(("10.254.254.254", 1))
    assert isolation.egress == []


def test_a_running_match_on_the_liveclient_port_is_invisible(isolation):
    """Something listening on 127.0.0.1:2999 (a live match) must not change results."""
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", LIVECLIENT_PORT))
        listener.listen()
    except OSError:  # a real game already holds the port: the guard must hide it all the same
        listener.close()
        listener = None
    try:
        with socket.socket() as s, pytest.raises(ConnectionRefusedError):
            s.connect(("127.0.0.1", LIVECLIENT_PORT))
        assert LiveClient().fetch() is None
    finally:
        if listener is not None:
            listener.close()
    assert isolation.egress == []
