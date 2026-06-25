"""Resolves the local store location.

The SQLite database lives inside a hidden ``.agentlens/`` directory (like
``.git``) instead of a bare ``agentlens.db`` in the working directory. Users are
far less likely to mistake a dotfolder for junk and delete it, and it keeps the
project root clean.

A one-time migration moves a legacy ``./agentlens.db`` into the new folder the
first time the path is resolved, so existing local data is preserved.
"""
from __future__ import annotations

import os
import shutil
import sys

STORE_DIR = ".agentlens"
STORE_FILE = "agentlens.db"
LEGACY_DB = "agentlens.db"


def _hide(path: str) -> None:
    """Set the hidden attribute on Windows; dotfolders are already hidden on
    POSIX so nothing to do there."""
    if sys.platform == "win32":
        try:
            import ctypes

            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
        except Exception:
            pass


def default_store_path(base: str | None = None) -> str:
    """Return ``<base>/.agentlens/agentlens.db``, creating (and hiding) the
    directory if needed and migrating any legacy ``<base>/agentlens.db``."""
    base = base or os.getcwd()
    store_dir = os.path.join(base, STORE_DIR)
    target = os.path.join(store_dir, STORE_FILE)

    created = not os.path.isdir(store_dir)
    os.makedirs(store_dir, exist_ok=True)
    if created:
        _hide(store_dir)

    legacy = os.path.join(base, LEGACY_DB)
    if os.path.isfile(legacy) and not os.path.exists(target):
        try:
            shutil.move(legacy, target)
        except Exception:
            return legacy
    return target


def resolve(db_path: str | None) -> str:
    """Resolve an explicit path, or the hidden default when ``db_path`` is
    falsy (None / empty string)."""
    return db_path or default_store_path()
