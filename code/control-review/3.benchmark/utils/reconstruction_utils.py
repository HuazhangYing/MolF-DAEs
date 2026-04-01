from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.io_utils import ensure_dir, read_json, utc_timestamp, write_json
from utils.latent_utils import get_source_latent_dim, source_latent_descriptor


def reconstruction_cache_dir(base_dir: str | Path, feature_group: str, subset_name: str) -> Path:
    return ensure_dir(Path(base_dir) / "outputs" / "_reconstruction" / feature_group / subset_name)


def compute_reconstruction_summary(*, base_dir: str | Path, feature_spec: Dict[str, Any], subset_name: str, subset_ids, force: bool = False) -> Dict[str, Any]:
    cache_dir = reconstruction_cache_dir(base_dir, feature_spec["feature_group"], subset_name)
    summary_path = cache_dir / "reconstruction_summary.json"

    if summary_path.exists() and not force:
        return read_json(summary_path)

    training_summary = None
    training_summary_path = feature_spec.get("training_summary_path")
    if training_summary_path and Path(training_summary_path).exists():
        training_df = pd.read_csv(training_summary_path)
        if not training_df.empty:
            training_summary = training_df.iloc[0].to_dict()

    summary = {
        "feature_group": feature_spec["feature_group"],
        "subset_name": subset_name,
        "n_samples": int(subset_ids.shape[0]),
        "timestamp": utc_timestamp(),
        "reference_notebook": feature_spec["reference_notebook"],
        "source_latent": source_latent_descriptor(feature_spec),
        "source_latent_dim": get_source_latent_dim(feature_spec),
        "reconstruction_rate": None,
        "mean_mse": None,
        "mean_mae": None,
        "status": "latent_reuse_only",
        "note": "当前阶段直接复用已保存 AE latent，仅比较后续降维效果，不重新生成 decoder reconstruction。",
        "training_summary": training_summary
    }
    write_json(summary_path, summary)
    return summary
