from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from utils.io_utils import ensure_dir, save_npy, utc_timestamp, write_json, write_log
from utils.molai_utils import load_molai_resources, preprocess_smiles_batch, smiles_to_latent

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path("/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv")
DEFAULT_MODEL_DIR = Path("/data/yinghuazhang/MolF-DAEs/code/control-review/code/MolAI-Publication/models_MolAI")
DEFAULT_OUTPUT_DIR = Path("/data/yinghuazhang/MolF-DAEs/code/control-review/result/3.benchmark/molai_pca3d")
PHASE_NAMES = ("fit_pca", "transform")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MolAI smi2lat embeddings and PCA-3D coordinates in original CSV order.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="包含 SMILES 的输入 CSV。")
    parser.add_argument("--smiles-column", default="Smiles", help="SMILES 列名。")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="MolAI model directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV 分块大小。")
    parser.add_argument("--predict-batch-size", type=int, default=2048, help="MolAI encoder predict batch size.")
    parser.add_argument("--pca-batch-size", type=int, default=16384, help="IncrementalPCA partial_fit batch size.")
    parser.add_argument("--force", action="store_true", help="如果目标结果已存在则覆盖。")
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于调试。限制读取前 N 行。")
    parser.add_argument("--progress-interval-seconds", type=int, default=3600, help="写入阶段性进度摘要的时间间隔，默认每小时。")
    return parser.parse_args()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


class RunLogger:
    def __init__(self, output_dir: Path, total_rows: int, progress_interval_seconds: int) -> None:
        self.output_dir = output_dir
        self.log_path = output_dir / "run_log.txt"
        self.progress_path = output_dir / "progress.json"
        self.total_rows = int(total_rows)
        self.progress_interval_seconds = int(progress_interval_seconds)
        self.start_time = time.time()
        self.last_progress_log_time = self.start_time
        self.phase = "setup"
        self.phase_rows_done = 0
        self.valid_rows_done = 0
        self.invalid_rows_done = 0
        self.chunk_id = -1
        self.message = "initialized"

    def _append(self, message: str) -> None:
        timestamp = utc_timestamp()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def write_initial(self, lines: List[str]) -> None:
        write_log(self.log_path, [f"[{utc_timestamp()}] {line}" for line in lines])
        self.message = "started"
        self.write_progress(force=True)

    def set_phase(self, phase: str, message: str) -> None:
        self.phase = phase
        self.message = message
        self._append(message)
        self.write_progress(force=True)

    def update_chunk(self, *, phase: str, chunk_id: int, stop_row: int, valid_rows: int, invalid_rows: int, note: str) -> None:
        self.phase = phase
        self.chunk_id = int(chunk_id)
        self.phase_rows_done = int(stop_row)
        self.valid_rows_done += int(valid_rows)
        self.invalid_rows_done += int(invalid_rows)
        self.message = note
        self._append(note)
        self.write_progress(force=False)

    def write_progress(self, *, force: bool) -> None:
        now = time.time()
        elapsed = now - self.start_time
        if (not force) and (now - self.last_progress_log_time < self.progress_interval_seconds):
            self._write_progress_json(elapsed)
            return

        pct = (self.phase_rows_done / self.total_rows * 100.0) if self.total_rows else 0.0
        rate = (self.phase_rows_done / elapsed) if elapsed > 0 else 0.0
        eta_seconds = ((self.total_rows - self.phase_rows_done) / rate) if rate > 0 else None
        summary = (
            f"Progress phase={self.phase} chunk={self.chunk_id} rows={self.phase_rows_done}/{self.total_rows} "
            f"({pct:.2f}%) valid_seen={self.valid_rows_done} invalid_seen={self.invalid_rows_done} "
            f"elapsed_sec={elapsed:.0f} eta_sec={eta_seconds:.0f}" if eta_seconds is not None else
            f"Progress phase={self.phase} chunk={self.chunk_id} rows={self.phase_rows_done}/{self.total_rows} "
            f"({pct:.2f}%) valid_seen={self.valid_rows_done} invalid_seen={self.invalid_rows_done} elapsed_sec={elapsed:.0f}"
        )
        self._append(summary)
        self.last_progress_log_time = now
        self._write_progress_json(elapsed, eta_seconds)

    def _write_progress_json(self, elapsed_seconds: float, eta_seconds: float | None = None) -> None:
        pct = (self.phase_rows_done / self.total_rows * 100.0) if self.total_rows else 0.0
        payload = {
            "timestamp": utc_timestamp(),
            "phase": self.phase,
            "chunk_id": self.chunk_id,
            "rows_done_in_phase": int(self.phase_rows_done),
            "total_rows": int(self.total_rows),
            "percent_done_in_phase": pct,
            "valid_rows_seen": int(self.valid_rows_done),
            "invalid_rows_seen": int(self.invalid_rows_done),
            "elapsed_seconds": float(elapsed_seconds),
            "eta_seconds": None if eta_seconds is None else float(eta_seconds),
            "message": self.message,
        }
        write_json(self.progress_path, payload)

    def finish(self, message: str) -> None:
        self.message = message
        self._append(message)
        self.write_progress(force=True)


def iter_smiles_chunks(csv_path: Path, smiles_column: str, chunksize: int, max_rows: int | None) -> Iterator[Dict[str, object]]:
    rows_seen = 0
    reader = pd.read_csv(csv_path, usecols=[smiles_column], chunksize=chunksize)
    for chunk_id, chunk in enumerate(reader):
        if max_rows is not None and rows_seen >= max_rows:
            break

        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        n_rows = int(chunk.shape[0])
        if n_rows == 0:
            continue

        start = rows_seen
        stop = rows_seen + n_rows
        rows_seen = stop
        yield {
            "chunk_id": chunk_id,
            "start": start,
            "stop": stop,
            "indices": np.arange(start, stop, dtype=np.int64),
            "smiles": chunk[smiles_column].tolist(),
        }


def iter_embedding_batches(latent: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, latent.shape[0], batch_size):
        yield latent[start : start + batch_size]


def flush_ipca_buffer(ipca: IncrementalPCA, pending: np.ndarray | None, batch_size: int, *, final: bool = False) -> np.ndarray | None:
    if pending is None or pending.shape[0] == 0:
        return None

    min_batch = max(int(ipca.n_components), 1)
    step = max(batch_size, min_batch)

    if final:
        if pending.shape[0] >= min_batch:
            ipca.partial_fit(pending)
            return None
        return pending

    while pending is not None and pending.shape[0] >= step + min_batch:
        ipca.partial_fit(pending[:step])
        pending = pending[step:]
        if pending.shape[0] == 0:
            pending = None

    return pending


def plot_explained_variance(explained_ratio: np.ndarray, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    components = np.arange(1, explained_ratio.shape[0] + 1)
    cumulative = np.cumsum(explained_ratio)

    ax1.bar(components, explained_ratio, color="#4C78A8", alpha=0.85)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="#F58518", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative Explained Variance Ratio")
    ax2.set_ylim(0.0, 1.0)

    fig.suptitle("MolAI PCA 3D Explained Variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_can_write_outputs(output_dir: Path, force: bool) -> None:
    required = [
        output_dir / "coords_3d.npy",
        output_dir / "ids.npy",
        output_dir / "pca_metrics.csv",
        output_dir / "run_summary.json",
        output_dir / "run_log.txt",
    ]
    if any(path.exists() for path in required) and not force:
        raise FileExistsError(f"Output already exists in {output_dir}. Re-run with --force to overwrite.")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_can_write_outputs(args.output_dir, args.force)

    total_rows = count_csv_rows(args.dataset)
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)

    logger = RunLogger(args.output_dir, total_rows=total_rows, progress_interval_seconds=args.progress_interval_seconds)
    logger.write_initial([
        "Start MolAI PCA 3D run.",
        f"dataset={args.dataset}",
        f"smiles_column={args.smiles_column}",
        f"model_dir={args.model_dir}",
        f"output_dir={args.output_dir}",
        f"chunksize={args.chunksize}",
        f"predict_batch_size={args.predict_batch_size}",
        f"pca_batch_size={args.pca_batch_size}",
        f"max_rows={args.max_rows}",
        f"total_rows={total_rows}",
        f"progress_interval_seconds={args.progress_interval_seconds}",
    ])

    resources = load_molai_resources(args.model_dir)
    latent_dim = int(resources.smiles_to_latent_model.output_shape[-1])
    ipca = IncrementalPCA(n_components=3, batch_size=args.pca_batch_size)

    preprocessing_summary = {
        "valid": 0,
        "missing": 0,
        "too_short": 0,
        "too_long": 0,
        "unsupported_characters": 0,
    }

    logger.set_phase("fit_pca", "Entering PCA fitting pass.")
    first_pass_valid = 0
    pending_latent: np.ndarray | None = None

    for payload in iter_smiles_chunks(args.dataset, args.smiles_column, args.chunksize, args.max_rows):
        valid_local_ids, normalized_smiles, batch_summary = preprocess_smiles_batch(payload["smiles"])
        for key, value in batch_summary.items():
            preprocessing_summary[key] += int(value)

        chunk_rows = int(payload["stop"] - payload["start"])
        invalid_rows = chunk_rows - int(valid_local_ids.size)
        if valid_local_ids.size != 0:
            latent = smiles_to_latent(normalized_smiles, resources, args.predict_batch_size)
            for latent_batch in iter_embedding_batches(latent, args.pca_batch_size):
                if pending_latent is None:
                    pending_latent = latent_batch
                else:
                    pending_latent = np.concatenate([pending_latent, latent_batch], axis=0)
                pending_latent = flush_ipca_buffer(ipca, pending_latent, args.pca_batch_size)
            first_pass_valid += int(valid_local_ids.size)

        logger.update_chunk(
            phase="fit_pca",
            chunk_id=int(payload["chunk_id"]),
            stop_row=int(payload["stop"]),
            valid_rows=int(valid_local_ids.size),
            invalid_rows=int(invalid_rows),
            note=(
                f"First pass chunk={payload['chunk_id']} rows={payload['start']}:{payload['stop']} "
                f"valid={int(valid_local_ids.size)} invalid={int(invalid_rows)}"
            ),
        )

    if first_pass_valid == 0:
        raise RuntimeError("No valid SMILES remained after MolAI preprocessing. PCA cannot be fitted.")
    if first_pass_valid < ipca.n_components:
        raise RuntimeError(f"Need at least {ipca.n_components} valid SMILES, but only found {first_pass_valid}.")

    pending_latent = flush_ipca_buffer(ipca, pending_latent, args.pca_batch_size, final=True)
    if pending_latent is not None and pending_latent.shape[0] > 0:
        raise RuntimeError("IncrementalPCA still has an unflushed latent buffer; adjust batch sizing.")

    coords_memmap_path = args.output_dir / "coords_3d.memmap"
    coords_memmap = np.memmap(coords_memmap_path, mode="w+", dtype=np.float32, shape=(total_rows, 3))
    coords_memmap[:] = np.nan

    valid_mask = np.zeros(total_rows, dtype=bool)
    valid_latent_sum = np.zeros(latent_dim, dtype=np.float64)
    valid_latent_sq_sum = np.zeros(latent_dim, dtype=np.float64)
    second_pass_valid = 0

    logger.set_phase("transform", "Entering PCA transform pass.")
    for payload in iter_smiles_chunks(args.dataset, args.smiles_column, args.chunksize, args.max_rows):
        global_ids = payload["indices"]
        valid_local_ids, normalized_smiles, _ = preprocess_smiles_batch(payload["smiles"])
        chunk_rows = int(payload["stop"] - payload["start"])
        invalid_rows = chunk_rows - int(valid_local_ids.size)
        if valid_local_ids.size != 0:
            latent = smiles_to_latent(normalized_smiles, resources, args.predict_batch_size)
            coords = ipca.transform(latent).astype(np.float32, copy=False)
            valid_global_ids = global_ids[valid_local_ids]

            coords_memmap[valid_global_ids] = coords
            valid_mask[valid_global_ids] = True

            valid_latent_sum += latent.sum(axis=0, dtype=np.float64)
            valid_latent_sq_sum += np.square(latent, dtype=np.float64).sum(axis=0, dtype=np.float64)
            second_pass_valid += int(valid_local_ids.size)

        logger.update_chunk(
            phase="transform",
            chunk_id=int(payload["chunk_id"]),
            stop_row=int(payload["stop"]),
            valid_rows=int(valid_local_ids.size),
            invalid_rows=int(invalid_rows),
            note=(
                f"Second pass chunk={payload['chunk_id']} rows={payload['start']}:{payload['stop']} "
                f"valid={int(valid_local_ids.size)} invalid={int(invalid_rows)}"
            ),
        )

    coords_memmap.flush()
    coords_array = np.array(coords_memmap)
    save_npy(args.output_dir / "coords_3d.npy", coords_array)
    del coords_memmap
    coords_memmap_path.unlink(missing_ok=True)

    ids = np.arange(total_rows, dtype=np.int64)
    save_npy(args.output_dir / "ids.npy", ids)
    save_npy(args.output_dir / "valid_mask.npy", valid_mask)
    joblib.dump(ipca, args.output_dir / "pca3d_model.joblib")

    explained_ratio = np.asarray(ipca.explained_variance_ratio_, dtype=np.float64)
    explained_variance = np.asarray(ipca.explained_variance_, dtype=np.float64)
    singular_values = np.asarray(ipca.singular_values_, dtype=np.float64)
    cumulative_explained_ratio = float(np.sum(explained_ratio))

    valid_count = int(valid_mask.sum())
    if valid_count != second_pass_valid:
        raise RuntimeError(f"Valid count mismatch between mask ({valid_count}) and transform pass ({second_pass_valid}).")

    latent_mean = valid_latent_sum / max(valid_count, 1)
    latent_var = valid_latent_sq_sum / max(valid_count, 1) - np.square(latent_mean)
    latent_var = np.clip(latent_var, a_min=0.0, a_max=None)

    coord_valid = coords_array[valid_mask]
    pca_metrics = pd.DataFrame(
        {
            "component": [1, 2, 3],
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance_ratio": np.cumsum(explained_ratio),
            "singular_value": singular_values,
        }
    )
    pca_metrics.to_csv(args.output_dir / "pca_metrics.csv", index=False)

    dataset_summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": total_rows},
            {"metric": "valid_rows", "value": valid_count},
            {"metric": "invalid_rows", "value": total_rows - valid_count},
            {"metric": "missing_rows", "value": preprocessing_summary["missing"]},
            {"metric": "too_short_rows", "value": preprocessing_summary["too_short"]},
            {"metric": "too_long_rows", "value": preprocessing_summary["too_long"]},
            {"metric": "unsupported_character_rows", "value": preprocessing_summary["unsupported_characters"]},
            {"metric": "valid_ratio", "value": valid_count / total_rows if total_rows else math.nan},
            {"metric": "cumulative_explained_variance_ratio_3d", "value": cumulative_explained_ratio},
            {"metric": "coord1_mean", "value": float(np.nanmean(coord_valid[:, 0])) if valid_count else math.nan},
            {"metric": "coord2_mean", "value": float(np.nanmean(coord_valid[:, 1])) if valid_count else math.nan},
            {"metric": "coord3_mean", "value": float(np.nanmean(coord_valid[:, 2])) if valid_count else math.nan},
            {"metric": "coord1_std", "value": float(np.nanstd(coord_valid[:, 0])) if valid_count else math.nan},
            {"metric": "coord2_std", "value": float(np.nanstd(coord_valid[:, 1])) if valid_count else math.nan},
            {"metric": "coord3_std", "value": float(np.nanstd(coord_valid[:, 2])) if valid_count else math.nan},
            {"metric": "latent_feature_mean_mean", "value": float(np.mean(latent_mean)) if valid_count else math.nan},
            {"metric": "latent_feature_std_mean", "value": float(np.mean(np.sqrt(latent_var))) if valid_count else math.nan},
        ]
    )
    dataset_summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)

    plot_explained_variance(explained_ratio, args.output_dir / "pca_explained_variance.png")

    run_summary = {
        "timestamp": utc_timestamp(),
        "dataset": str(args.dataset),
        "smiles_column": args.smiles_column,
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "encoder_model": str(args.model_dir / "smi2lat_epoch_6.h5"),
        "total_rows": int(total_rows),
        "valid_rows": int(valid_count),
        "invalid_rows": int(total_rows - valid_count),
        "preprocessing_summary": {k: int(v) for k, v in preprocessing_summary.items()},
        "latent_dim": latent_dim,
        "projection_method": "IncrementalPCA",
        "projection_components": 3,
        "cumulative_explained_variance_ratio_3d": cumulative_explained_ratio,
        "pca_metrics_csv": str(args.output_dir / "pca_metrics.csv"),
        "dataset_summary_csv": str(args.output_dir / "dataset_summary.csv"),
        "coords_path": str(args.output_dir / "coords_3d.npy"),
        "ids_path": str(args.output_dir / "ids.npy"),
        "valid_mask_path": str(args.output_dir / "valid_mask.npy"),
        "plot_path": str(args.output_dir / "pca_explained_variance.png"),
    }
    write_json(args.output_dir / "run_summary.json", run_summary)

    logger.finish(
        f"Finished successfully. first_pass_valid={first_pass_valid} second_pass_valid={second_pass_valid} "
        f"cumulative_explained_variance_ratio_3d={cumulative_explained_ratio:.8f}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    raise SystemExit(main())
