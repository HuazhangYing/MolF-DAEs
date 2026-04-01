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
import umap
from numpy.lib.format import open_memmap
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import trustworthiness

from utils.io_utils import ensure_dir, save_npy, utc_timestamp, write_json, write_log
from utils.mol2vec_utils import (
    MOL2VEC_DEFAULT_MODEL_PATH,
    load_mol2vec_resources,
    preprocess_smiles_batch,
    sentences_to_vectors,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = Path("/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv")
DEFAULT_OUTPUT_ROOT = Path("/data/yinghuazhang/MolF-DAEs/code/control-review/result/3.benchmark/mol2vec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mol2vec embeddings and 3D PCA/UMAP coordinates in original CSV order.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Input CSV containing SMILES.")
    parser.add_argument("--smiles-column", default="Smiles", help="SMILES column name.")
    parser.add_argument("--model-path", type=Path, default=MOL2VEC_DEFAULT_MODEL_PATH, help="Pretrained mol2vec Word2Vec model path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory. Artifacts are written into a mol2vec subfolder layout here.")
    parser.add_argument("--run-name", default=None, help="Optional run label. Defaults to full or rowsN.")
    parser.add_argument("--chunksize", type=int, default=20000, help="CSV chunk size.")
    parser.add_argument("--radius", type=int, default=1, help="Morgan radius used by mol2vec sentence generation.")
    parser.add_argument("--unseen-token", default="UNK", help="Token used for unseen identifiers. Set empty string to disable.")
    parser.add_argument("--pca-batch-size", type=int, default=16384, help="IncrementalPCA partial_fit batch size.")
    parser.add_argument("--umap-fit-rows", type=int, default=300000, help="Number of valid embeddings used to fit UMAP. Full valid set is used when smaller.")
    parser.add_argument("--umap-transform-batch-size", type=int, default=50000, help="Batch size for UMAP transform over the full dataset.")
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap-metric", default="cosine", help="UMAP metric.")
    parser.add_argument("--metrics-sample-size", type=int, default=10000, help="Sample size for trustworthiness and plotting diagnostics.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug mode: only process the first N rows.")
    parser.add_argument("--progress-interval-seconds", type=int, default=1800, help="Progress snapshot interval in seconds.")
    return parser.parse_args()


class RunLogger:
    def __init__(self, output_root: Path, total_rows: int, progress_interval_seconds: int) -> None:
        self.output_root = output_root
        self.total_rows = int(total_rows)
        self.progress_interval_seconds = int(progress_interval_seconds)
        self.log_path = output_root / "run_log.txt"
        self.progress_path = output_root / "progress.json"
        self.start_time = time.time()
        self.last_progress = self.start_time
        self.phase = "setup"
        self.rows_done = 0
        self.valid_rows = 0
        self.invalid_rows = 0
        self.message = "initialized"
        self.chunk_id = -1

    def write_initial(self, lines: List[str]) -> None:
        write_log(self.log_path, [f"[{utc_timestamp()}] {line}" for line in lines])
        self.message = "started"
        self.write_progress(force=True)

    def append(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_timestamp()}] {message}\n")

    def set_phase(self, phase: str, message: str) -> None:
        self.phase = phase
        self.message = message
        self.append(message)
        self.write_progress(force=True)

    def update_chunk(self, chunk_id: int, stop_row: int, valid_rows: int, invalid_rows: int, note: str) -> None:
        self.chunk_id = int(chunk_id)
        self.rows_done = int(stop_row)
        self.valid_rows += int(valid_rows)
        self.invalid_rows += int(invalid_rows)
        self.message = note
        self.append(note)
        self.write_progress(force=False)

    def write_progress(self, force: bool) -> None:
        now = time.time()
        elapsed = now - self.start_time
        pct = (self.rows_done / self.total_rows * 100.0) if self.total_rows else 0.0
        rate = self.rows_done / elapsed if elapsed > 0 else 0.0
        eta_seconds = (self.total_rows - self.rows_done) / rate if rate > 0 else None
        payload = {
            "timestamp": utc_timestamp(),
            "phase": self.phase,
            "chunk_id": self.chunk_id,
            "rows_done": int(self.rows_done),
            "total_rows": int(self.total_rows),
            "percent_done": float(pct),
            "valid_rows_seen": int(self.valid_rows),
            "invalid_rows_seen": int(self.invalid_rows),
            "elapsed_seconds": float(elapsed),
            "eta_seconds": None if eta_seconds is None else float(eta_seconds),
            "message": self.message,
        }
        write_json(self.progress_path, payload)
        if force or (now - self.last_progress >= self.progress_interval_seconds):
            eta_text = f" eta_sec={eta_seconds:.0f}" if eta_seconds is not None else ""
            self.append(
                f"Progress phase={self.phase} chunk={self.chunk_id} rows={self.rows_done}/{self.total_rows} "
                f"({pct:.2f}%) valid_seen={self.valid_rows} invalid_seen={self.invalid_rows} elapsed_sec={elapsed:.0f}{eta_text}"
            )
            self.last_progress = now

    def finish(self, message: str) -> None:
        self.message = message
        self.append(message)
        self.write_progress(force=True)


class ReservoirSampler:
    def __init__(self, capacity: int, dim: int, seed: int) -> None:
        self.capacity = max(int(capacity), 1)
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((self.capacity, self.dim), dtype=np.float32)
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.count = 0
        self.seen = 0

    def add_batch(self, ids: np.ndarray, values: np.ndarray) -> None:
        for row_id, row in zip(ids, values):
            self.seen += 1
            if self.count < self.capacity:
                self.data[self.count] = row
                self.ids[self.count] = row_id
                self.count += 1
                continue
            replace_idx = int(self.rng.integers(0, self.seen))
            if replace_idx < self.capacity:
                self.data[replace_idx] = row
                self.ids[replace_idx] = row_id

    def export(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.empty((0, self.dim), dtype=np.float32), np.empty((0,), dtype=np.int64)
        order = np.argsort(self.ids[: self.count])
        return self.data[: self.count][order].copy(), self.ids[: self.count][order].copy()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


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


def ensure_can_write(output_root: Path, force: bool) -> None:
    required = [
        output_root / "embeddings_full" / "embeddings.npy",
        output_root / "pca3d_full" / "coords_3d.npy",
        output_root / "umap3d_full" / "coords_3d.npy",
        output_root / "run_summary.json",
        output_root / "run_log.txt",
    ]
    if any(path.exists() for path in required) and not force:
        raise FileExistsError(f"Output already exists in {output_root}. Re-run with --force to overwrite.")


def effective_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.max_rows is not None:
        return f"rows{args.max_rows}"
    return "full"


def plot_embedding_norms(norm_sample: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(norm_sample, bins=50, color="#4C78A8", alpha=0.9)
    ax.set_title("mol2vec Embedding Norm Distribution")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


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
    fig.suptitle("mol2vec PCA 3D Explained Variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_coord_pairs(coords: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    pairs = [
        (0, 1, "coord1", "coord2"),
        (0, 2, "coord1", "coord3"),
        (1, 2, "coord2", "coord3"),
    ]
    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(coords[:, i], coords[:, j], s=2, alpha=0.25, linewidths=0, color="#4C78A8")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_metric_table(rows: List[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    run_name = effective_run_name(args)
    output_root = ensure_dir(args.output_root)
    ensure_can_write(output_root, args.force)

    embedding_dir = ensure_dir(output_root / f"embeddings_{run_name}")
    pca_dir = ensure_dir(output_root / f"pca3d_{run_name}")
    umap_dir = ensure_dir(output_root / f"umap3d_{run_name}")

    total_rows = count_csv_rows(args.dataset)
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)

    logger = RunLogger(output_root=output_root, total_rows=total_rows, progress_interval_seconds=args.progress_interval_seconds)
    logger.write_initial([
        "Start mol2vec 3D run.",
        f"dataset={args.dataset}",
        f"smiles_column={args.smiles_column}",
        f"model_path={args.model_path}",
        f"output_root={output_root}",
        f"run_name={run_name}",
        f"chunksize={args.chunksize}",
        f"radius={args.radius}",
        f"pca_batch_size={args.pca_batch_size}",
        f"umap_fit_rows={args.umap_fit_rows}",
        f"umap_transform_batch_size={args.umap_transform_batch_size}",
        f"max_rows={args.max_rows}",
        f"total_rows={total_rows}",
    ])

    resources = load_mol2vec_resources(args.model_path)
    model = resources["model"]
    sentence_builder = resources["mol2alt_sentence"]
    vector_size = int(resources["vector_size"])
    vocab = resources["vocab"]
    unseen_token = args.unseen_token if args.unseen_token else None

    embedding_path = embedding_dir / "embeddings.npy"
    embeddings_memmap = open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(total_rows, vector_size))
    embeddings_memmap[:] = np.nan
    valid_mask = np.zeros(total_rows, dtype=bool)
    ids = np.arange(total_rows, dtype=np.int64)

    ipca = IncrementalPCA(n_components=3, batch_size=args.pca_batch_size)
    pending_latent: np.ndarray | None = None
    sampler = ReservoirSampler(capacity=min(args.umap_fit_rows, max(total_rows, 1)), dim=vector_size, seed=args.random_state)

    preprocessing_summary = {
        "valid": 0,
        "missing": 0,
        "invalid_smiles": 0,
        "empty_sentence": 0,
    }
    token_total = 0.0
    recognized_total = 0.0
    oov_total = 0.0
    zero_vector_rows = 0.0
    sum_sentence_length = 0.0
    max_sentence_length = 0.0
    norm_sum = 0.0
    norm_sq_sum = 0.0
    first_pass_valid = 0

    logger.set_phase("embed_fit_pca", "Generating mol2vec embeddings and fitting PCA.")
    for payload in iter_smiles_chunks(args.dataset, args.smiles_column, args.chunksize, args.max_rows):
        global_ids = payload["indices"]
        valid_local_ids, sentences, batch_summary = preprocess_smiles_batch(payload["smiles"], args.radius, sentence_builder)
        for key, value in batch_summary.items():
            preprocessing_summary[key] += int(value)

        chunk_rows = int(payload["stop"] - payload["start"])
        invalid_rows = chunk_rows - int(valid_local_ids.size)
        if valid_local_ids.size:
            embeddings, embedding_metrics = sentences_to_vectors(sentences, model, vocab, unseen_token=unseen_token)
            valid_global_ids = global_ids[valid_local_ids]
            embeddings_memmap[valid_global_ids] = embeddings
            valid_mask[valid_global_ids] = True
            sampler.add_batch(valid_global_ids, embeddings)

            token_total += embedding_metrics["total_tokens"]
            recognized_total += embedding_metrics["recognized_tokens"]
            oov_total += embedding_metrics["oov_tokens"]
            zero_vector_rows += embedding_metrics["zero_vector_rows"]
            sum_sentence_length += embedding_metrics["mean_sentence_length"] * embeddings.shape[0]
            max_sentence_length = max(max_sentence_length, embedding_metrics["max_sentence_length"])

            norms = np.linalg.norm(embeddings, axis=1)
            norm_sum += float(norms.sum(dtype=np.float64))
            norm_sq_sum += float(np.square(norms, dtype=np.float64).sum(dtype=np.float64))

            for start in range(0, embeddings.shape[0], args.pca_batch_size):
                latent_batch = embeddings[start : start + args.pca_batch_size]
                if pending_latent is None:
                    pending_latent = latent_batch
                else:
                    pending_latent = np.concatenate([pending_latent, latent_batch], axis=0)
                pending_latent = flush_ipca_buffer(ipca, pending_latent, args.pca_batch_size)
            first_pass_valid += int(valid_local_ids.size)

        logger.update_chunk(
            chunk_id=int(payload["chunk_id"]),
            stop_row=int(payload["stop"]),
            valid_rows=int(valid_local_ids.size),
            invalid_rows=int(invalid_rows),
            note=(
                f"Embed pass chunk={payload['chunk_id']} rows={payload['start']}:{payload['stop']} "
                f"valid={int(valid_local_ids.size)} invalid={int(invalid_rows)}"
            ),
        )

    if first_pass_valid == 0:
        raise RuntimeError("No valid SMILES remained after mol2vec preprocessing.")
    if first_pass_valid < ipca.n_components:
        raise RuntimeError(f"Need at least {ipca.n_components} valid SMILES, but only found {first_pass_valid}.")

    pending_latent = flush_ipca_buffer(ipca, pending_latent, args.pca_batch_size, final=True)
    if pending_latent is not None and pending_latent.shape[0] > 0:
        raise RuntimeError("IncrementalPCA still has an unflushed buffer; adjust batch sizes.")

    embeddings_memmap.flush()
    valid_row_ids = np.flatnonzero(valid_mask).astype(np.int64)
    save_npy(embedding_dir / "ids.npy", ids)
    save_npy(embedding_dir / "valid_mask.npy", valid_mask)
    save_npy(embedding_dir / "valid_row_ids.npy", valid_row_ids)

    sampled_embeddings, sampled_ids = sampler.export()
    if sampled_embeddings.shape[0] == 0:
        raise RuntimeError("UMAP fit sample is empty.")

    sampled_norms = np.linalg.norm(sampled_embeddings, axis=1)
    plot_embedding_norms(sampled_norms, embedding_dir / "embedding_norm_hist.png")

    valid_count = int(valid_mask.sum())
    mean_sentence_length = sum_sentence_length / max(valid_count, 1)
    mean_norm = norm_sum / max(valid_count, 1)
    norm_var = max(norm_sq_sum / max(valid_count, 1) - mean_norm * mean_norm, 0.0)

    write_metric_table(
        [
            {"metric": "total_rows", "value": total_rows},
            {"metric": "valid_rows", "value": valid_count},
            {"metric": "invalid_rows", "value": total_rows - valid_count},
            {"metric": "missing_rows", "value": preprocessing_summary["missing"]},
            {"metric": "invalid_smiles_rows", "value": preprocessing_summary["invalid_smiles"]},
            {"metric": "empty_sentence_rows", "value": preprocessing_summary["empty_sentence"]},
            {"metric": "valid_ratio", "value": valid_count / total_rows if total_rows else math.nan},
            {"metric": "embedding_dim", "value": vector_size},
            {"metric": "total_tokens", "value": token_total},
            {"metric": "recognized_tokens", "value": recognized_total},
            {"metric": "oov_tokens", "value": oov_total},
            {"metric": "oov_ratio", "value": oov_total / token_total if token_total else math.nan},
            {"metric": "zero_vector_rows", "value": zero_vector_rows},
            {"metric": "mean_sentence_length", "value": mean_sentence_length},
            {"metric": "max_sentence_length", "value": max_sentence_length},
            {"metric": "embedding_norm_mean", "value": mean_norm},
            {"metric": "embedding_norm_std", "value": math.sqrt(norm_var)},
            {"metric": "umap_fit_sample_rows", "value": int(sampled_embeddings.shape[0])},
        ],
        embedding_dir / "embedding_summary.csv",
    )

    logger.set_phase("pca_transform", "Transforming full valid embedding set with PCA.")
    pca_coords_path = pca_dir / "coords_3d.npy"
    pca_coords = open_memmap(pca_coords_path, mode="w+", dtype=np.float32, shape=(total_rows, 3))
    pca_coords[:] = np.nan
    for start in range(0, total_rows, args.chunksize):
        stop = min(start + args.chunksize, total_rows)
        mask = valid_mask[start:stop]
        if np.any(mask):
            rows = np.asarray(embeddings_memmap[start:stop][mask], dtype=np.float32)
            chunk_coords = np.full((stop - start, 3), np.nan, dtype=np.float32)
            chunk_coords[mask] = ipca.transform(rows).astype(np.float32, copy=False)
            pca_coords[start:stop] = chunk_coords
    pca_coords.flush()
    pca_coords_array = np.load(pca_coords_path, mmap_mode="r")
    save_npy(pca_dir / "ids.npy", ids)
    save_npy(pca_dir / "valid_mask.npy", valid_mask)
    joblib.dump(ipca, pca_dir / "pca3d_model.joblib")

    explained_ratio = np.asarray(ipca.explained_variance_ratio_, dtype=np.float64)
    explained_variance = np.asarray(ipca.explained_variance_, dtype=np.float64)
    singular_values = np.asarray(ipca.singular_values_, dtype=np.float64)
    cumulative_explained_ratio = float(np.sum(explained_ratio))

    pca_valid_coords = np.asarray(pca_coords_array[valid_mask], dtype=np.float32)
    write_metric_table(
        [
            {
                "component": i + 1,
                "explained_variance": float(explained_variance[i]),
                "explained_variance_ratio": float(explained_ratio[i]),
                "cumulative_explained_variance_ratio": float(np.cumsum(explained_ratio)[i]),
                "singular_value": float(singular_values[i]),
            }
            for i in range(3)
        ],
        pca_dir / "pca_metrics.csv",
    )
    write_metric_table(
        [
            {"metric": "total_rows", "value": total_rows},
            {"metric": "valid_rows", "value": valid_count},
            {"metric": "invalid_rows", "value": total_rows - valid_count},
            {"metric": "cumulative_explained_variance_ratio_3d", "value": cumulative_explained_ratio},
            {"metric": "coord1_mean", "value": float(np.nanmean(pca_valid_coords[:, 0]))},
            {"metric": "coord2_mean", "value": float(np.nanmean(pca_valid_coords[:, 1]))},
            {"metric": "coord3_mean", "value": float(np.nanmean(pca_valid_coords[:, 2]))},
            {"metric": "coord1_std", "value": float(np.nanstd(pca_valid_coords[:, 0]))},
            {"metric": "coord2_std", "value": float(np.nanstd(pca_valid_coords[:, 1]))},
            {"metric": "coord3_std", "value": float(np.nanstd(pca_valid_coords[:, 2]))},
        ],
        pca_dir / "dataset_summary.csv",
    )
    plot_explained_variance(explained_ratio, pca_dir / "pca_explained_variance.png")
    plot_sample_size = min(args.metrics_sample_size, pca_valid_coords.shape[0])
    plot_rng = np.random.default_rng(args.random_state)
    pca_plot_ids = plot_rng.choice(pca_valid_coords.shape[0], size=plot_sample_size, replace=False)
    plot_coord_pairs(pca_valid_coords[pca_plot_ids], pca_dir / "pca_coord_pairs.png", "mol2vec PCA 3D Coordinate Pairs")

    logger.set_phase("umap_fit", "Fitting UMAP on sampled mol2vec embeddings.")
    umap_model = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
        transform_seed=args.random_state,
        verbose=False,
    )
    umap_fit_coords = umap_model.fit_transform(sampled_embeddings)

    logger.set_phase("umap_transform", "Transforming full valid embedding set with fitted UMAP model.")
    umap_coords_path = umap_dir / "coords_3d.npy"
    umap_coords = open_memmap(umap_coords_path, mode="w+", dtype=np.float32, shape=(total_rows, 3))
    umap_coords[:] = np.nan
    for start in range(0, total_rows, args.umap_transform_batch_size):
        stop = min(start + args.umap_transform_batch_size, total_rows)
        mask = valid_mask[start:stop]
        if np.any(mask):
            rows = np.asarray(embeddings_memmap[start:stop][mask], dtype=np.float32)
            chunk_coords = np.full((stop - start, 3), np.nan, dtype=np.float32)
            chunk_coords[mask] = umap_model.transform(rows).astype(np.float32, copy=False)
            umap_coords[start:stop] = chunk_coords
    umap_coords.flush()
    umap_coords_array = np.load(umap_coords_path, mmap_mode="r")
    save_npy(umap_dir / "ids.npy", ids)
    save_npy(umap_dir / "valid_mask.npy", valid_mask)
    joblib.dump(umap_model, umap_dir / "umap3d_model.joblib")

    umap_valid_coords = np.asarray(umap_coords_array[valid_mask], dtype=np.float32)
    trust_n = min(args.metrics_sample_size, sampled_embeddings.shape[0])
    trust_ids = plot_rng.choice(sampled_embeddings.shape[0], size=trust_n, replace=False)
    trust_score = float(trustworthiness(sampled_embeddings[trust_ids], umap_fit_coords[trust_ids], n_neighbors=min(15, max(trust_n - 1, 1))))

    write_metric_table(
        [
            {"metric": "fit_rows", "value": int(sampled_embeddings.shape[0])},
            {"metric": "transformed_valid_rows", "value": valid_count},
            {"metric": "n_neighbors", "value": args.umap_n_neighbors},
            {"metric": "min_dist", "value": args.umap_min_dist},
            {"metric": "metric", "value": args.umap_metric},
            {"metric": "trustworthiness_sample_size", "value": trust_n},
            {"metric": "trustworthiness", "value": trust_score},
            {"metric": "coord1_mean", "value": float(np.nanmean(umap_valid_coords[:, 0]))},
            {"metric": "coord2_mean", "value": float(np.nanmean(umap_valid_coords[:, 1]))},
            {"metric": "coord3_mean", "value": float(np.nanmean(umap_valid_coords[:, 2]))},
            {"metric": "coord1_std", "value": float(np.nanstd(umap_valid_coords[:, 0]))},
            {"metric": "coord2_std", "value": float(np.nanstd(umap_valid_coords[:, 1]))},
            {"metric": "coord3_std", "value": float(np.nanstd(umap_valid_coords[:, 2]))},
        ],
        umap_dir / "umap_metrics.csv",
    )
    write_metric_table(
        [
            {"metric": "total_rows", "value": total_rows},
            {"metric": "valid_rows", "value": valid_count},
            {"metric": "invalid_rows", "value": total_rows - valid_count},
            {"metric": "fit_rows", "value": int(sampled_embeddings.shape[0])},
            {"metric": "trustworthiness", "value": trust_score},
        ],
        umap_dir / "dataset_summary.csv",
    )
    plot_coord_pairs(np.asarray(umap_fit_coords[trust_ids], dtype=np.float32), umap_dir / "umap_coord_pairs.png", "mol2vec UMAP 3D Coordinate Pairs")

    run_summary = {
        "timestamp": utc_timestamp(),
        "dataset": str(args.dataset),
        "smiles_column": args.smiles_column,
        "model_path": str(args.model_path),
        "output_root": str(output_root),
        "run_name": run_name,
        "total_rows": int(total_rows),
        "valid_rows": int(valid_count),
        "invalid_rows": int(total_rows - valid_count),
        "embedding_dim": vector_size,
        "radius": int(args.radius),
        "unseen_token": unseen_token,
        "preprocessing_summary": {k: int(v) for k, v in preprocessing_summary.items()},
        "embedding_summary_csv": str(embedding_dir / "embedding_summary.csv"),
        "embedding_path": str(embedding_path),
        "embedding_valid_mask_path": str(embedding_dir / "valid_mask.npy"),
        "embedding_valid_row_ids_path": str(embedding_dir / "valid_row_ids.npy"),
        "pca_coords_path": str(pca_coords_path),
        "pca_metrics_csv": str(pca_dir / "pca_metrics.csv"),
        "umap_coords_path": str(umap_coords_path),
        "umap_metrics_csv": str(umap_dir / "umap_metrics.csv"),
        "pca_cumulative_explained_variance_ratio_3d": cumulative_explained_ratio,
        "umap_fit_rows": int(sampled_embeddings.shape[0]),
        "umap_trustworthiness": trust_score,
        "note": "UMAP is fitted on a deterministic valid-embedding sample, then transformed over all valid rows to preserve original CSV order.",
    }
    write_json(output_root / "run_summary.json", run_summary)
    logger.finish(
        f"Finished successfully. valid_rows={valid_count} pca_cumulative_explained_variance_ratio_3d={cumulative_explained_ratio:.8f} "
        f"umap_fit_rows={sampled_embeddings.shape[0]} trustworthiness={trust_score:.8f}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    raise SystemExit(main())
