from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REQUIRED_RUN_FILES = (
    "coords_3d.npy",
    "ids.npy",
    "params.json",
    "reconstruction_summary.json",
    "run_log.txt",
)


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_params_hash(data: Dict[str, Any], length: int = 10) -> str:
    digest = hashlib.sha1(canonical_json(data).encode("utf-8")).hexdigest()
    return digest[:length]


def write_json(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_log(path: os.PathLike[str] | str, lines: List[str]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def run_is_complete(run_dir: os.PathLike[str] | str) -> bool:
    run_path = Path(run_dir)
    return all((run_path / name).exists() for name in REQUIRED_RUN_FILES)


def save_npy(path: os.PathLike[str] | str, array: np.ndarray) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    np.save(path_obj, array)
