from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

MODULE_PATH = Path(__file__).with_name('1.clustering_quick_eval.py')
spec = importlib.util.spec_from_file_location('clustering_quick_eval_mod', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

build_projection_specs = module.build_projection_specs
build_sample = module.build_sample
continuity_score = module.continuity_score
fit_knn = module.fit_knn
get_projection_arrays = module.get_projection_arrays
load_binary_fingerprints = module.load_binary_fingerprints
neighborhood_target_consistency = module.neighborhood_target_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep local metrics over different k values.")
    parser.add_argument("--labels", default="/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv")
    parser.add_argument("--data2", default="/data/yinghuazhang/MolF-DAEs/dataset/PharmacoPFP_molecule3.data2")
    parser.add_argument("--dae", default="/data/yinghuazhang/MolF-DAEs/result/pharmachopfp/test1_data_best_pharmacopfp/test1_ME_pharmacopfp.csv")
    parser.add_argument("--pca", default="/data/yinghuazhang/MolF-DAEs/result/comparison/PCA_2_ME.csv")
    parser.add_argument("--umap", default="/data/yinghuazhang/MolF-DAEs/result/comparison/UMAP_2_ME.csv")
    parser.add_argument("--match-column", default="ChEMBL ID")
    parser.add_argument("--add-mol2vec", action="store_true")
    parser.add_argument("--mol2vec-mode", choices=["umap3d", "pca3d"], default="umap3d")
    parser.add_argument("--mol2vec-pca-coords", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/mol2vec/pca3d_full/coords_3d.npy")
    parser.add_argument("--mol2vec-pca-ids", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/mol2vec/pca3d_full/ids.npy")
    parser.add_argument("--mol2vec-umap-coords", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/mol2vec/umap3d_full/coords_3d.npy")
    parser.add_argument("--mol2vec-umap-ids", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/mol2vec/umap3d_full/ids.npy")
    parser.add_argument("--add-molai-pca3d", action="store_true")
    parser.add_argument("--molai-pca3d-coords", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/molai_pca3d_full/coords_3d.npy")
    parser.add_argument("--molai-pca3d-ids", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/molai_pca3d_full/ids.npy")
    parser.add_argument("--add-pacmap-pubchem", action="store_true")
    parser.add_argument("--pacmap-pubchem-coords", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_020c271bf2/coords_3d.npy")
    parser.add_argument("--pacmap-pubchem-ids", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_020c271bf2/ids.npy")
    parser.add_argument("--add-phate-pubchem", action="store_true")
    parser.add_argument("--phate-pubchem-coords", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_6826002097/coords_3d.npy")
    parser.add_argument("--phate-pubchem-ids", default="/data/yinghuazhang/MolF-DAEs/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_6826002097/ids.npy")
    parser.add_argument("--sample-per-label", type=int, default=120)
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 15, 30, 50])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", default="/data/yinghuazhang/MolF-DAEs/code/control-review/1.clustering/output_k_sweep")
    return parser.parse_args()


def per_sample_knn_overlap(orig_knn: np.ndarray, emb_knn: np.ndarray, k: int) -> np.ndarray:
    vals = []
    for row_o, row_e in zip(orig_knn, emb_knn):
        vals.append(len(set(row_o.tolist()) & set(row_e.tolist())) / k)
    return np.asarray(vals, dtype=float)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    projections = build_projection_specs(args)
    df_sample, coverage_df = build_sample(
        args.labels,
        projections,
        args.sample_per_label,
        args.random_state,
        args.match_column,
    )
    df_sample.to_csv(outdir / "df_sample_used.csv", index=False)
    coverage_df.to_csv(outdir / "projection_coverage_summary.csv", index=False)

    sample_idx = df_sample["orig_idx"].to_numpy(dtype=np.int64)
    x_bin = load_binary_fingerprints(args.data2, sample_idx)
    arrays = get_projection_arrays(df_sample, list(projections))
    labels = df_sample["label"].to_numpy()

    metric_rows = []
    overlap_rows = []
    target_rows = []
    target_label_rows = []
    for k in args.k_values:
        print(f"Running k={k} ...")
        orig_knn = fit_knn(x_bin, k, "jaccard")
        for method, coords in arrays.items():
            emb_knn = fit_knn(coords, k, "euclidean")
            overlap = per_sample_knn_overlap(orig_knn, emb_knn, k)
            target_scores, target_per_label = neighborhood_target_consistency(labels, emb_knn)
            metric_rows.append(
                {
                    "method": method,
                    "k": k,
                    "trustworthiness": float(trustworthiness(x_bin.astype(np.uint8), coords, n_neighbors=k, metric="jaccard")),
                    "continuity": continuity_score(orig_knn, coords, k, args.chunk_size),
                    "knn_preservation": float(overlap.mean()),
                    "knn_overlap_std": float(overlap.std()),
                    "knn_overlap_q25": float(np.quantile(overlap, 0.25)),
                    "knn_overlap_median": float(np.quantile(overlap, 0.50)),
                    "knn_overlap_q75": float(np.quantile(overlap, 0.75)),
                    "target_consistency_mean": float(target_scores.mean()),
                    "target_consistency_std": float(target_scores.std()),
                    "target_consistency_q25": float(np.quantile(target_scores, 0.25)),
                    "target_consistency_median": float(np.quantile(target_scores, 0.50)),
                    "target_consistency_q75": float(np.quantile(target_scores, 0.75)),
                }
            )
            for value in overlap:
                overlap_rows.append({"method": method, "k": k, "overlap": float(value)})
            for value in target_scores:
                target_rows.append({"method": method, "k": k, "target_consistency": float(value)})
            tmp = target_per_label.copy()
            tmp.insert(0, "k", k)
            tmp.insert(0, "method", method)
            target_label_rows.append(tmp)

    df_metrics = pd.DataFrame(metric_rows).sort_values(["method", "k"]).reset_index(drop=True)
    df_overlap = pd.DataFrame(overlap_rows)
    df_target = pd.DataFrame(target_rows)
    df_target_label = pd.concat(target_label_rows, axis=0).reset_index(drop=True)
    df_metrics.to_csv(outdir / "local_metrics_k_sweep.csv", index=False)
    df_overlap.to_csv(outdir / "knn_overlap_distribution.csv", index=False)
    df_target.to_csv(outdir / "target_consistency_distribution.csv", index=False)
    df_target_label.to_csv(outdir / "target_consistency_per_label_k_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for method in df_metrics["method"].unique():
        sub = df_metrics[df_metrics["method"] == method]
        axes[0].plot(sub["k"], sub["trustworthiness"], marker="o", label=method)
        axes[1].plot(sub["k"], sub["continuity"], marker="o", label=method)
        axes[2].plot(sub["k"], sub["knn_preservation"], marker="o", label=method)
        axes[3].plot(sub["k"], sub["target_consistency_mean"], marker="o", label=method)
    axes[0].set_title("Trustworthiness vs k")
    axes[1].set_title("Continuity vs k")
    axes[2].set_title("kNN Preservation vs k")
    axes[3].set_title("Target Consistency vs k")
    for ax in axes:
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "local_metrics_k_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    methods = list(df_overlap["method"].unique())
    k_values = list(sorted(df_overlap["k"].unique()))
    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 3.5 * len(methods)), squeeze=False)
    for row_idx, method in enumerate(methods):
        overlap_data = [df_overlap[(df_overlap["method"] == method) & (df_overlap["k"] == k)]["overlap"].to_numpy() for k in k_values]
        target_data = [df_target[(df_target["method"] == method) & (df_target["k"] == k)]["target_consistency"].to_numpy() for k in k_values]
        axes[row_idx, 0].boxplot(overlap_data, tick_labels=k_values, showfliers=False)
        axes[row_idx, 0].set_title(f"{method}: per-sample kNN overlap")
        axes[row_idx, 0].set_xlabel("k")
        axes[row_idx, 0].set_ylabel("Overlap ratio")
        axes[row_idx, 0].set_ylim(-0.02, 1.02)
        axes[row_idx, 1].boxplot(target_data, tick_labels=k_values, showfliers=False)
        axes[row_idx, 1].set_title(f"{method}: per-sample target consistency")
        axes[row_idx, 1].set_xlabel("k")
        axes[row_idx, 1].set_ylabel("Same-label neighbor ratio")
        axes[row_idx, 1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outdir / "local_distribution_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 3.5 * len(methods)), squeeze=False)
    for ax, method in zip(axes[:, 0], methods):
        sub = df_target_label[df_target_label["method"] == method]
        for label in sub["label"].unique():
            part = sub[sub["label"] == label].sort_values("k")
            ax.plot(part["k"], part["target_consistency_mean"], marker="o", label=label)
        ax.set_title(f"{method}: target consistency by label")
        ax.set_xlabel("k")
        ax.set_ylabel("Same-label neighbor ratio")
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "target_consistency_by_label_k_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = []
    for method in df_metrics["method"].unique():
        sub = df_metrics[df_metrics["method"] == method].sort_values("k")
        summary.append(
            {
                "method": method,
                "trustworthiness_range": float(sub["trustworthiness"].max() - sub["trustworthiness"].min()),
                "continuity_range": float(sub["continuity"].max() - sub["continuity"].min()),
                "knn_preservation_range": float(sub["knn_preservation"].max() - sub["knn_preservation"].min()),
                "target_consistency_range": float(sub["target_consistency_mean"].max() - sub["target_consistency_mean"].min()),
                "overlap_std_range": float(sub["knn_overlap_std"].max() - sub["knn_overlap_std"].min()),
            }
        )
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(outdir / "local_metrics_variation_summary.csv", index=False)

    print(df_metrics.to_string(index=False))
    print("\nCoverage summary")
    print(coverage_df.to_string(index=False))
    print("\nVariation summary")
    print(df_summary.to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
