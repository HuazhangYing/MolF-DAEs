from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors


DEFAULT_LABELS = [
    "kinase",
    "Protease",
    "G-protein coupled receptor",
    "Nuclear receptor",
    "Remainder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick clustering preservation evaluation.")
    parser.add_argument("--labels", default="/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv")
    parser.add_argument("--data2", default="/data/yinghuazhang/MolF-DAEs/dataset/PharmacoPFP_molecule3.data2") # 特征对应
    parser.add_argument("--dae", default="/data/yinghuazhang/MolF-DAEs/result/pharmachopfp/test1_data_best_pharmacopfp/test1_ME_pharmacopfp.csv") # 特征对应
    parser.add_argument("--pca", default="/data/yinghuazhang/MolF-DAEs/result/comparison/PCA/PCA_2_ME.csv")
    parser.add_argument("--umap", default="/data/yinghuazhang/MolF-DAEs/result/comparison/UMAP/UMAP_2_ME.csv")
    parser.add_argument("--sample-per-label", type=int, default=300)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--shepard-pairs", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", default="/data/yinghuazhang/MolF-DAEs/code/control-review/1.clustering/output_quick")
    return parser.parse_args()


def load_projection(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["orig_idx"] = np.arange(len(df))
    return df[["orig_idx", "X", "Y", "Z"]].rename(columns={"X": f"{name}_X", "Y": f"{name}_Y", "Z": f"{name}_Z"})


def build_sample(labels_path: str, projections: dict[str, str], sample_per_label: int, random_state: int) -> pd.DataFrame:
    df_label = pd.read_csv(labels_path).copy()
    df_label["orig_idx"] = np.arange(len(df_label))

    label_cols = ["Protease", "Nuclear receptor", "kinase", "G-protein coupled receptor"]
    label_count = (df_label[label_cols] == 1).sum(axis=1)
    df_label["label"] = "Remainder"
    single_mask = label_count == 1
    multi_mask = label_count > 1
    df_label.loc[single_mask, "label"] = df_label.loc[single_mask, label_cols].idxmax(axis=1)
    df_label.loc[multi_mask, "label"] = "multiple"

    merged = df_label
    for name, path in projections.items():
        merged = merged.merge(load_projection(path, name), on="orig_idx", how="left")

    sample_parts = []
    for cls in DEFAULT_LABELS:
        subset = merged[merged["label"] == cls]
        n_take = min(sample_per_label, len(subset))
        if n_take > 0:
            sample_parts.append(subset.sample(n=n_take, random_state=random_state))
    return pd.concat(sample_parts, axis=0).reset_index(drop=True)


def load_binary_fingerprints(data2_path: str, sample_idx: np.ndarray) -> np.ndarray:
    x = load(data2_path)
    x_sample = np.asarray(x[sample_idx]).reshape(len(sample_idx), -1)
    return (x_sample > 0.5).astype(bool)


def get_projection_arrays(df_sample: pd.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    return {
        name: df_sample[[f"{name}_X", f"{name}_Y", f"{name}_Z"]].to_numpy(dtype=np.float32)
        for name in names
    }


def fit_knn(data: np.ndarray, k: int, metric: str) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute" if metric == "jaccard" else "auto",
        n_jobs=-1 if metric == "jaccard" else None,
    )
    model.fit(data)
    return model.kneighbors(return_distance=False)[:, 1:]


def knn_preservation(orig_knn: np.ndarray, emb_knn: np.ndarray, k: int) -> float:
    scores = []
    for row_o, row_e in zip(orig_knn, emb_knn):
        scores.append(len(set(row_o.tolist()) & set(row_e.tolist())) / k)
    return float(np.mean(scores))


def continuity_score(orig_knn: np.ndarray, coords: np.ndarray, k: int, chunk_size: int) -> float:
    n = len(coords)
    emb_knn = fit_knn(coords, k, "euclidean")
    emb_knn_sets = [set(row.tolist()) for row in emb_knn]
    penalty = 0.0

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = coords[start:stop]
        dists = pairwise_distances(block, coords, metric="euclidean")
        for local_i, global_i in enumerate(range(start, stop)):
            row = dists[local_i]
            row[global_i] = np.inf
            missing = [j for j in orig_knn[global_i] if j not in emb_knn_sets[global_i]]
            if not missing:
                continue
            valid = np.arange(n) != global_i
            row_valid = row[valid]
            for j in missing:
                rank = 1 + np.count_nonzero(row_valid < row[j])
                penalty += rank - k

    normalizer = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - normalizer * penalty)


def neighborhood_target_consistency(labels: np.ndarray, emb_knn: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    same = labels[emb_knn] == labels[:, None]
    scores = same.mean(axis=1).astype(float)
    per_label = []
    for label in DEFAULT_LABELS:
        mask = labels == label
        if mask.any():
            vals = scores[mask]
            per_label.append(
                {
                    "label": label,
                    "n_samples": int(mask.sum()),
                    "target_consistency_mean": float(vals.mean()),
                    "target_consistency_std": float(vals.std()),
                    "target_consistency_q25": float(np.quantile(vals, 0.25)),
                    "target_consistency_median": float(np.quantile(vals, 0.50)),
                    "target_consistency_q75": float(np.quantile(vals, 0.75)),
                }
            )
    return scores, pd.DataFrame(per_label)


def generalized_tanimoto_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    denom = float(np.dot(a, a) + np.dot(b, b) - dot)
    return 0.0 if denom <= 0 else 1.0 - dot / denom


def cluster_metrics(x_bin: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    used_labels = [lab for lab in DEFAULT_LABELS if lab in set(labels.tolist())]
    orig_centroids = []
    emb_centroids = []
    for lab in used_labels:
        mask = labels == lab
        orig_centroids.append(x_bin[mask].mean(axis=0, dtype=np.float32))
        emb_centroids.append(coords[mask].mean(axis=0, dtype=np.float32))
    orig_centroids = np.asarray(orig_centroids, dtype=np.float32)
    emb_centroids = np.asarray(emb_centroids, dtype=np.float32)

    n_clusters = len(used_labels)
    d_orig = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    d_emb = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d_orig[i, j] = d_orig[j, i] = generalized_tanimoto_distance(orig_centroids[i], orig_centroids[j])
            d_emb[i, j] = d_emb[j, i] = float(np.linalg.norm(emb_centroids[i] - emb_centroids[j]))

    tri = np.triu_indices(n_clusters, k=1)
    orig_upper = d_orig[tri]
    emb_upper = d_emb[tri]
    cluster_spearman = float(spearmanr(orig_upper, emb_upper).statistic)
    orig_norm = (orig_upper - orig_upper.mean()) / (orig_upper.std() + 1e-12)
    emb_norm = (emb_upper - emb_upper.mean()) / (emb_upper.std() + 1e-12)
    cluster_stress = float(np.sqrt(np.mean((orig_norm - emb_norm) ** 2)))

    return {
        "cluster_distance_spearman": cluster_spearman,
        "cluster_distance_stress": cluster_stress,
        "cluster_labels": used_labels,
        "cluster_matrix_original": d_orig,
        "cluster_matrix_embedding": d_emb,
    }


def shepard_metrics(x_bin: np.ndarray, coords: np.ndarray, max_pairs: int, random_state: int) -> dict[str, object]:
    n = len(coords)
    rng = np.random.default_rng(random_state)
    idx_i = rng.integers(0, n, size=max_pairs, endpoint=False)
    idx_j = rng.integers(0, n, size=max_pairs, endpoint=False)
    valid = idx_i != idx_j
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    a = x_bin[idx_i]
    b = x_bin[idx_j]
    inter = np.logical_and(a, b).sum(axis=1)
    union = np.logical_or(a, b).sum(axis=1)
    d_orig = 1.0 - (inter / np.clip(union, 1, None))
    d_emb = np.linalg.norm(coords[idx_i] - coords[idx_j], axis=1)

    return {
        "shepard_spearman": float(spearmanr(d_orig, d_emb).statistic),
        "shepard_original": d_orig,
        "shepard_embedding": d_emb,
    }


def mean_offdiag(block: np.ndarray) -> float:
    if block.shape[0] <= 1:
        return 0.0
    tri = np.triu_indices(block.shape[0], k=1)
    vals = block[tri]
    return float(vals.mean()) if len(vals) else 0.0


def cluster_diagnostics(x_bin: np.ndarray, coords: np.ndarray, labels: np.ndarray, cluster: dict[str, object]) -> dict[str, object]:
    used_labels = list(cluster["cluster_labels"])
    orig_dist = pairwise_distances(x_bin.astype(np.uint8), metric="jaccard", n_jobs=-1)
    emb_dist = pairwise_distances(coords, metric="euclidean")

    silhouette_orig = float(silhouette_score(orig_dist, labels, metric="precomputed"))
    silhouette_emb = float(silhouette_score(coords, labels, metric="euclidean"))
    sil_orig_samples = silhouette_samples(orig_dist, labels, metric="precomputed")
    sil_emb_samples = silhouette_samples(coords, labels, metric="euclidean")

    profile_rows = []
    for lab in used_labels:
        mask = labels == lab
        inv_mask = ~mask
        within_orig = mean_offdiag(orig_dist[np.ix_(mask, mask)])
        within_emb = mean_offdiag(emb_dist[np.ix_(mask, mask)])
        between_orig = float(orig_dist[np.ix_(mask, inv_mask)].mean())
        between_emb = float(emb_dist[np.ix_(mask, inv_mask)].mean())
        profile_rows.append(
            {
                "label": lab,
                "n_samples": int(mask.sum()),
                "within_orig": within_orig,
                "between_orig": between_orig,
                "within_between_ratio_orig": within_orig / max(between_orig, 1e-12),
                "within_emb": within_emb,
                "between_emb": between_emb,
                "within_between_ratio_emb": within_emb / max(between_emb, 1e-12),
                "ratio_distortion": (within_emb / max(between_emb, 1e-12)) - (within_orig / max(between_orig, 1e-12)),
                "silhouette_orig": float(sil_orig_samples[mask].mean()),
                "silhouette_emb": float(sil_emb_samples[mask].mean()),
                "silhouette_drop": float(sil_emb_samples[mask].mean() - sil_orig_samples[mask].mean()),
            }
        )
    profile_df = pd.DataFrame(profile_rows).sort_values("ratio_distortion", ascending=False).reset_index(drop=True)

    d_orig = np.asarray(cluster["cluster_matrix_original"])
    d_emb = np.asarray(cluster["cluster_matrix_embedding"])
    tri = np.triu_indices(len(used_labels), k=1)
    orig_upper = d_orig[tri]
    emb_upper = d_emb[tri]
    scale = float(np.median(orig_upper) / max(np.median(emb_upper), 1e-12))
    emb_scaled = d_emb * scale

    pair_rows = []
    for i in range(len(used_labels)):
        for j in range(i + 1, len(used_labels)):
            pair_rows.append(
                {
                    "pair": f"{used_labels[i]} vs {used_labels[j]}",
                    "label_i": used_labels[i],
                    "label_j": used_labels[j],
                    "orig_distance": float(d_orig[i, j]),
                    "emb_distance": float(d_emb[i, j]),
                    "emb_distance_scaled": float(emb_scaled[i, j]),
                    "distance_error": float(emb_scaled[i, j] - d_orig[i, j]),
                    "abs_distance_error": float(abs(emb_scaled[i, j] - d_orig[i, j])),
                    "distance_ratio": float(emb_scaled[i, j] / max(d_orig[i, j], 1e-12)),
                }
            )
    pair_df = pd.DataFrame(pair_rows).sort_values("abs_distance_error", ascending=False).reset_index(drop=True)

    nearest_rows = []
    for i, lab in enumerate(used_labels):
        orig_candidates = d_orig[i].copy()
        emb_candidates = d_emb[i].copy()
        orig_candidates[i] = np.inf
        emb_candidates[i] = np.inf
        orig_nearest_idx = int(np.argmin(orig_candidates))
        emb_nearest_idx = int(np.argmin(emb_candidates))
        nearest_rows.append(
            {
                "label": lab,
                "orig_nearest_cluster": used_labels[orig_nearest_idx],
                "emb_nearest_cluster": used_labels[emb_nearest_idx],
                "nearest_cluster_preserved": used_labels[orig_nearest_idx] == used_labels[emb_nearest_idx],
            }
        )
    nearest_df = pd.DataFrame(nearest_rows)

    return {
        "profile_df": profile_df,
        "pair_df": pair_df,
        "nearest_df": nearest_df,
        "silhouette_orig": silhouette_orig,
        "silhouette_emb": silhouette_emb,
        "silhouette_drop": silhouette_emb - silhouette_orig,
        "nearest_cluster_preservation_rate": float(nearest_df["nearest_cluster_preserved"].mean()),
    }


def save_summary_plot(df_metrics: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    higher = ["trustworthiness", "continuity", "knn_preservation", "cluster_distance_spearman", "shepard_spearman"]
    x = np.arange(len(df_metrics))
    width = 0.15
    for idx, col in enumerate(higher):
        axes[0].bar(x + idx * width, df_metrics[col], width=width, label=col)
    axes[0].set_xticks(x + width * (len(higher) - 1) / 2)
    axes[0].set_xticklabels(df_metrics["method"], rotation=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Higher is better")

    axes[1].bar(df_metrics["method"], df_metrics["cluster_distance_stress"], color="#d95f02")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_title("Lower is better: cluster_distance_stress")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_method_plots(name: str, cluster: dict[str, object], shepard: dict[str, object], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title in zip(
        axes,
        [cluster["cluster_matrix_original"], cluster["cluster_matrix_embedding"]],
        ["Original centroid distances", f"{name} centroid distances"],
    ):
        im = ax.imshow(matrix, cmap="magma")
        ax.set_xticks(np.arange(len(cluster["cluster_labels"])))
        ax.set_yticks(np.arange(len(cluster["cluster_labels"])))
        ax.set_xticklabels(cluster["cluster_labels"], rotation=30, ha="right")
        ax.set_yticklabels(cluster["cluster_labels"])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / f"cluster_distance_heatmap_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    hb = ax.hexbin(shepard["shepard_original"], shepard["shepard_embedding"], gridsize=40, mincnt=1, cmap="viridis")
    ax.set_title(f"{name} Shepard Spearman={shepard['shepard_spearman']:.4f}")
    ax.set_xlabel("Original Jaccard distance")
    ax.set_ylabel("3D Euclidean distance")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / f"shepard_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_target_consistency_plots(df_target: pd.DataFrame, per_label_tables: dict[str, pd.DataFrame], outdir: Path, k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(df_target["method"], df_target["target_consistency_mean"], color=["#1b9e77", "#7570b3", "#d95f02"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(f"Neighborhood target consistency (k={k})")
    axes[0].set_ylabel("Same-label neighbor ratio")
    axes[0].tick_params(axis="x", rotation=15)

    methods = list(per_label_tables.keys())
    labels = list(per_label_tables[methods[0]]["label"]) if methods else []
    x = np.arange(len(labels))
    width = 0.25
    for idx, method in enumerate(methods):
        tbl = per_label_tables[method]
        axes[1].bar(x + idx * width, tbl["target_consistency_mean"], width=width, label=method)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f"Per-label neighborhood target consistency (k={k})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"target_consistency_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_molf_dae_diagnostics(diag: dict[str, object], outdir: Path) -> None:
    profile_df = diag["profile_df"]
    pair_df = diag["pair_df"]
    nearest_df = diag["nearest_df"]

    profile_df.to_csv(outdir / "molf_dae_cluster_profile.csv", index=False)
    pair_df.to_csv(outdir / "molf_dae_cluster_pair_distortion.csv", index=False)
    nearest_df.to_csv(outdir / "molf_dae_nearest_cluster_check.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(profile_df))
    width = 0.35
    axes[0].bar(x - width / 2, profile_df["within_between_ratio_orig"], width=width, label="Original")
    axes[0].bar(x + width / 2, profile_df["within_between_ratio_emb"], width=width, label="MolF-DAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(profile_df["label"], rotation=25, ha="right")
    axes[0].set_title("Within/Between ratio by cluster")
    axes[0].legend()

    axes[1].bar(x - width / 2, profile_df["silhouette_orig"], width=width, label="Original")
    axes[1].bar(x + width / 2, profile_df["silhouette_emb"], width=width, label="MolF-DAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(profile_df["label"], rotation=25, ha="right")
    axes[1].set_title("Mean silhouette by cluster")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "molf_dae_cluster_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_pairs = pair_df.head(min(10, len(pair_df))).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(top_pairs["pair"], top_pairs["distance_error"], color=["#d95f02" if v > 0 else "#1b9e77" for v in top_pairs["distance_error"]])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("MolF-DAE top centroid-pair distance distortions")
    ax.set_ylabel("Scaled embedding distance - original distance")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(outdir / "molf_dae_pair_distortion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        f"MolF-DAE overall silhouette: {diag['silhouette_emb']:.4f}",
        f"Original silhouette: {diag['silhouette_orig']:.4f}",
        f"Silhouette drop: {diag['silhouette_drop']:.4f}",
        f"Nearest-cluster preservation rate: {diag['nearest_cluster_preservation_rate']:.2%}",
        "Top clusters with the strongest compactness/separation distortion:",
    ]
    for _, row in profile_df.head(3).iterrows():
        summary_lines.append(
            f"- {row['label']}: ratio {row['within_between_ratio_orig']:.4f} -> {row['within_between_ratio_emb']:.4f}, silhouette {row['silhouette_orig']:.4f} -> {row['silhouette_emb']:.4f}"
        )
    summary_lines.append("Top distorted centroid pairs:")
    for _, row in pair_df.head(5).iterrows():
        summary_lines.append(
            f"- {row['pair']}: original {row['orig_distance']:.4f}, scaled MolF-DAE {row['emb_distance_scaled']:.4f}, error {row['distance_error']:.4f}"
        )
    (outdir / "molf_dae_why_not_lossless.txt").write_text("\n".join(summary_lines))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    projections = {"MolF-DAE": args.dae, "PCA": args.pca, "UMAP": args.umap}
    df_sample = build_sample(args.labels, projections, args.sample_per_label, args.random_state)
    df_sample.to_csv(outdir / "df_sample_used.csv", index=False)

    sample_idx = df_sample["orig_idx"].to_numpy(dtype=np.int64)
    x_bin = load_binary_fingerprints(args.data2, sample_idx)
    arrays = get_projection_arrays(df_sample, list(projections))
    labels = df_sample["label"].to_numpy()
    orig_knn = fit_knn(x_bin, args.neighbors, "jaccard")

    rows = []
    target_rows = []
    target_tables = {}
    molf_dae_diag = None
    for name, coords in arrays.items():
        print(f"Running {name} on {len(df_sample)} points ...")
        emb_knn = fit_knn(coords, args.neighbors, "euclidean")
        target_scores, target_per_label = neighborhood_target_consistency(labels, emb_knn)
        target_tables[name] = target_per_label
        target_per_label.to_csv(outdir / f"target_consistency_per_label_{name}.csv", index=False)
        pd.DataFrame({"label": labels, "target_consistency": target_scores}).to_csv(outdir / f"target_consistency_per_sample_{name}.csv", index=False)

        cluster = cluster_metrics(x_bin, coords, labels)
        shepard = shepard_metrics(x_bin, coords, args.shepard_pairs, args.random_state)
        save_method_plots(name, cluster, shepard, outdir)

        if name == "MolF-DAE":
            molf_dae_diag = cluster_diagnostics(x_bin, coords, labels, cluster)
            save_molf_dae_diagnostics(molf_dae_diag, outdir)

        target_rows.append(
            {
                "method": name,
                "k": args.neighbors,
                "target_consistency_mean": float(target_scores.mean()),
                "target_consistency_std": float(target_scores.std()),
                "target_consistency_q25": float(np.quantile(target_scores, 0.25)),
                "target_consistency_median": float(np.quantile(target_scores, 0.50)),
                "target_consistency_q75": float(np.quantile(target_scores, 0.75)),
            }
        )

        row = {
            "method": name,
            "n_points": len(df_sample),
            "k": args.neighbors,
            "trustworthiness": float(trustworthiness(x_bin.astype(np.uint8), coords, n_neighbors=args.neighbors, metric="jaccard")),
            "continuity": continuity_score(orig_knn, coords, args.neighbors, args.chunk_size),
            "knn_preservation": knn_preservation(orig_knn, emb_knn, args.neighbors),
            "cluster_distance_spearman": cluster["cluster_distance_spearman"],
            "cluster_distance_stress": cluster["cluster_distance_stress"],
            "shepard_spearman": shepard["shepard_spearman"],
            "target_consistency_mean": float(target_scores.mean()),
            "target_consistency_median": float(np.quantile(target_scores, 0.50)),
            "cluster_silhouette": np.nan,
            "nearest_cluster_preservation_rate": np.nan,
        }
        if name == "MolF-DAE" and molf_dae_diag is not None:
            row["cluster_silhouette"] = molf_dae_diag["silhouette_emb"]
            row["nearest_cluster_preservation_rate"] = molf_dae_diag["nearest_cluster_preservation_rate"]
        rows.append(row)

    df_metrics = pd.DataFrame(rows)
    df_target = pd.DataFrame(target_rows)
    df_target.to_csv(outdir / "target_consistency_summary.csv", index=False)
    save_target_consistency_plots(df_target, target_tables, outdir, args.neighbors)

    df_metrics["overall_score"] = (
        df_metrics["trustworthiness"]
        + df_metrics["continuity"]
        + df_metrics["knn_preservation"]
        + df_metrics["cluster_distance_spearman"]
        + df_metrics["shepard_spearman"]
        + df_metrics["target_consistency_mean"]
        - df_metrics["cluster_distance_stress"]
    ) / 6.0
    df_metrics["appears_lossless"] = (
        (df_metrics["trustworthiness"] > 0.99)
        & (df_metrics["continuity"] > 0.99)
        & (df_metrics["knn_preservation"] > 0.99)
        & (df_metrics["cluster_distance_spearman"] > 0.99)
        & (df_metrics["shepard_spearman"] > 0.99)
        & (df_metrics["target_consistency_mean"] > 0.99)
        & (df_metrics["cluster_distance_stress"] < 0.02)
    )
    df_metrics = df_metrics.sort_values("overall_score", ascending=False).reset_index(drop=True)
    df_metrics.to_csv(outdir / "projection_metrics.csv", index=False)
    save_summary_plot(df_metrics, outdir / "metric_summary.png")

    print(df_metrics.to_string(index=False))
    print("\nTarget consistency summary")
    print(df_target.to_string(index=False))
    if molf_dae_diag is not None:
        print("\nMolF-DAE diagnostics")
        print(molf_dae_diag["profile_df"].to_string(index=False))
        print(molf_dae_diag["nearest_df"].to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
