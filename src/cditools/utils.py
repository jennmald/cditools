from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def makedirs(path: str, mode: int = 0o777) -> list[str]:
    """Recursively make directories and set permissions"""
    # Permissions not working with os.makedirs -
    # See: http://stackoverflow.com/questions/5231901
    if not path or Path(path).exists():
        return []

    head, tail = os.path.split(path)
    ret = makedirs(head, mode)
    try:
        Path(path).mkdir()
    except OSError as ex:
        if "File exists" not in str(ex):
            raise

    Path(path).chmod(mode)
    ret.append(path)
    return ret


def ordered_dict_move_to_beginning(od: dict[str, Any], key: str) -> None:
    if key not in od:
        return

    value = od[key]
    items = [(k, v) for k, v in od.items() if k != key]
    od.clear()
    od[key] = value
    od.update(items)
