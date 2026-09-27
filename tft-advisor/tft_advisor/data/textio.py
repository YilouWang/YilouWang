"""Reading the player's own data files (comps, mechanics overrides).

Windows users create them with Notepad ("UTF-8 with BOM") or PowerShell 5.1
(``Set-Content -Encoding UTF8`` writes a BOM), so a BOM is accepted, and every
error names the file in Chinese.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Optional


def read_user_text(path: Path, what: str) -> str:
    """UTF-8 text with or without a BOM; ``what`` names the file kind ("阵容文件")."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{what} {path} 读取失败: {exc}") from None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{what} {path} 不是 UTF-8 编码，请用记事本另存为 UTF-8") from None


def read_user_data(path: Path, what: str, fmt: Optional[str] = None) -> Any:
    """Parse a JSON or TOML user file (``fmt`` "json" / "toml", default by
    suffix); errors name the file."""
    text = read_user_text(path, what)
    fmt = fmt or ("toml" if path.suffix.lower() == ".toml" else "json")
    try:
        if fmt == "toml":
            return tomllib.loads(text)
        return json.loads(text)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise ValueError(f"{what} {path} 格式错误: {exc}") from None
