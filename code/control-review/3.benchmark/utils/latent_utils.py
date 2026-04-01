from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from utils.io_utils import ensure_dir, save_npy, write_json


def load_config(config_path: str | Path) -> Dict[str, Any]:
    raw_text = Path(config_path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(raw_text)
    except Exception:
        data = json.loads(raw_text)
    return data


def load_array(path: str | Path, mmap_mode: str | None = "r", file_format: str | None = None) -> np.ndarray:
    path_obj = Path(path)
    resolved_format = (file_format or path_obj.suffix.lstrip('.')).lower()
    if resolved_format == 'npy':
        return np.load(path_obj, mmap_mode=mmap_mode)
    try:
        return joblib.load(path_obj, mmap_mode=mmap_mode)
    except TypeError:
        return joblib.load(path_obj)


def load_joblib_array(path: str | Path, mmap_mode: str | None = "r") -> np.ndarray:
    return load_array(path, mmap_mode=mmap_mode, file_format='joblib')


def resolve_feature_spec(config: Dict[str, Any], feature_group: str) -> Dict[str, Any]:
    if feature_group not in config["feature_groups"]:
        supported = ", ".join(sorted(config["feature_groups"]))
        raise KeyError(f"Unsupported feature_group={feature_group!r}. Supported: {supported}")
    spec = dict(config["feature_groups"][feature_group])
    spec["feature_group"] = feature_group
    return spec


def parse_chunk_sort_key(filename: str) -> Tuple[int, int]:
    start_text, end_text = filename.split("-")
    return int(start_text), int(end_text)


def get_chunk_files(feature_spec: Dict[str, Any]) -> List[Path]:
    latent_source = feature_spec["latent_source"]
    if latent_source["type"] != "chunk_dir":
        return []
    chunk_dir = Path(latent_source["dir"])
    files = [p for p in chunk_dir.glob(latent_source.get("chunk_pattern", "*")) if p.is_file()]
    return sorted(files, key=lambda path: parse_chunk_sort_key(path.name))


def get_chunk_metadata(feature_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata_key = "_chunk_metadata_cache"
    if metadata_key in feature_spec:
        return feature_spec[metadata_key]

    metadata: List[Dict[str, Any]] = []
    start = 0
    for chunk_file in get_chunk_files(feature_spec):
        chunk = load_joblib_array(chunk_file)
        n_rows = int(chunk.shape[0])
        metadata.append({
            "path": chunk_file,
            "start": start,
            "end": start + n_rows,
            "n_rows": n_rows,
        })
        start += n_rows

    feature_spec[metadata_key] = metadata
    return metadata


def get_total_size(feature_spec: Dict[str, Any]) -> int:
    latent_source = feature_spec["latent_source"]
    if latent_source["type"] == "single_file":
        index_ids_path = latent_source.get("index_ids_path")
        if index_ids_path:
            return int(np.load(index_ids_path).shape[0])
        latent = load_array(latent_source["path"], mmap_mode="r", file_format=latent_source.get("file_format"))
        return int(latent.shape[0])
    metadata = get_chunk_metadata(feature_spec)
    if not metadata:
        raise FileNotFoundError(f"No chunk files found for {feature_spec['feature_group']}")
    return int(metadata[-1]["end"])


def get_source_latent_dim(feature_spec: Dict[str, Any]) -> int:
    return int(feature_spec["latent_source"]["latent_dim"])


def get_full_ids(total_size: int) -> np.ndarray:
    return np.arange(total_size, dtype=np.int64)


def subset_cache_dir(base_dir: str | Path, feature_group: str, subset_name: str) -> Path:
    return ensure_dir(Path(base_dir) / "outputs" / "_subsets" / feature_group / subset_name)


def get_or_create_subset_ids(base_dir: str | Path, feature_group: str, subset_name: str, total_size: int, subset_policy: Dict[str, Any]) -> np.ndarray:
    cache_dir = subset_cache_dir(base_dir, feature_group, subset_name)
    ids_path = cache_dir / "ids.npy"
    meta_path = cache_dir / "metadata.json"

    if ids_path.exists():
        return np.load(ids_path)

    if subset_name == "full":
        ids = get_full_ids(total_size)
    else:
        size = int(subset_policy["sizes"][subset_name])
        if size > total_size:
            raise ValueError(f"Subset {subset_name} requests {size} samples, but only {total_size} available.")
        rng = np.random.default_rng(int(subset_policy["seed"]))
        ids = np.sort(rng.choice(total_size, size=size, replace=False).astype(np.int64))

    save_npy(ids_path, ids)
    write_json(meta_path, {
        "feature_group": feature_group,
        "subset_name": subset_name,
        "n_samples": int(ids.shape[0]),
        "seed": int(subset_policy["seed"]),
        "source": "deterministic_sampling",
        "total_size": int(total_size)
    })
    return ids


def load_subset_latent(feature_spec: Dict[str, Any], subset_ids: np.ndarray) -> np.ndarray:
    latent_source = feature_spec["latent_source"]
    if latent_source["type"] == "single_file":
        latent = load_array(latent_source["path"], mmap_mode="r", file_format=latent_source.get("file_format"))
        index_ids_path = latent_source.get("index_ids_path")
        if index_ids_path:
            source_ids = np.load(index_ids_path)
            return np.asarray(latent[source_ids[subset_ids]], dtype=np.float32)
        return np.asarray(latent[subset_ids], dtype=np.float32)

    metadata = get_chunk_metadata(feature_spec)
    latent_dim = get_source_latent_dim(feature_spec)
    output = np.empty((subset_ids.shape[0], latent_dim), dtype=np.float32)
    positions = np.arange(subset_ids.shape[0], dtype=np.int64)

    for chunk_info in metadata:
        start = chunk_info["start"]
        end = chunk_info["end"]
        mask = (subset_ids >= start) & (subset_ids < end)
        if not np.any(mask):
            continue
        chunk = np.asarray(load_joblib_array(chunk_info["path"]), dtype=np.float32)
        local_ids = subset_ids[mask] - start
        output[positions[mask]] = chunk[local_ids]

    return output


def source_latent_descriptor(feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    latent_source = feature_spec["latent_source"]
    if latent_source["type"] == "single_file":
        return {
            "type": "single_file",
            "path": latent_source["path"],
            "file_format": latent_source.get("file_format", "joblib"),
            "index_ids_path": latent_source.get("index_ids_path"),
            "latent_dim": int(latent_source["latent_dim"]),
            "note": latent_source.get("note")
        }
    metadata = get_chunk_metadata(feature_spec)
    return {
        "type": "chunk_dir",
        "dir": latent_source["dir"],
        "latent_dim": int(latent_source["latent_dim"]),
        "n_chunks": len(metadata),
        "total_size": int(metadata[-1]["end"]) if metadata else 0,
        "note": latent_source.get("note")
    }


def parameter_grid(grid_spec: Dict[str, Any]):
    keys = sorted(grid_spec.keys())
    values = [grid_spec[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))
