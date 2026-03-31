from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background validation with local band-sized neighborhoods ordered along a local direction."
    )
    parser.add_argument("--fp-type", choices=["pubchem", "maccs", "pharma"], default="pubchem")
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--candidate-restarts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to result/clustering/{fp_type}/bg_band_gradient_validation",
    )
    return parser.parse_args()


def resolve_paths(fp_type: str) -> dict[str, str]:
    labels = "/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv"
    pca = "/data/yinghuazhang/MolF-DAEs/result/comparison/PCA/PCA_2_ME.csv"
    umap = "/data/yinghuazhang/MolF-DAEs/result/comparison/UMAP/UMAP_2_ME.csv"
    if fp_type == "pubchem":
        return {
            "labels": labels,
            "data2": "/data/yinghuazhang/MolF-DAEs/dataset/pubchem_molecule3.data2",
            "band": "/data/yinghuazhang/MolF-DAEs/dataset/band1-pubchemfp.xlsx",
            "pca": pca,
            "umap": umap,
        }
    if fp_type == "maccs":
        return {
            "labels": labels,
            "data2": "/data/yinghuazhang/MolF-DAEs/dataset/MACCSFP_molecule3.data2",
            "band": "/data/yinghuazhang/MolF-DAEs/dataset/band5-maccsfp.xlsx",
            "pca": pca,
            "umap": umap,
        }
    return {
        "labels": labels,
        "data2": "/data/yinghuazhang/MolF-DAEs/dataset/PharmacoPFP_molecule3.data2",
        "band": "/data/yinghuazhang/MolF-DAEs/dataset/band6-pharmacopfp.xlsx",
        "pca": pca,
        "umap": umap,
    }


def column_letters_to_index(col_ref: str) -> int:
    value = 0
    for ch in col_ref:
        if not ch.isalpha():
            break
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(name))
    strings = []
    for si in root.findall("main:si", NS):
        parts = []
        for node in si.iter():
            if node.tag.endswith("}t") and node.text is not None:
                parts.append(node.text)
        strings.append("".join(parts))
    return strings


def read_first_sheet_rows(xlsx_path: str) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = read_shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        first_sheet = workbook.find("main:sheets/main:sheet", NS)
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = None
        for rel in rels:
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise ValueError("Cannot find first worksheet target in xlsx")
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        sheet_root = ET.fromstring(zf.read(target))
        rows = []
        sheet_data = sheet_root.find("main:sheetData", NS)
        for row in sheet_data.findall("main:row", NS):
            cells = {}
            max_col = -1
            for c in row.findall("main:c", NS):
                ref = c.attrib.get("r", "A1")
                col_idx = column_letters_to_index(ref)
                max_col = max(max_col, col_idx)
                cell_type = c.attrib.get("t")
                value_node = c.find("main:v", NS)
                value = ""
                if cell_type == "inlineStr":
                    t_node = c.find("main:is/main:t", NS)
                    value = t_node.text if t_node is not None and t_node.text is not None else ""
                elif value_node is not None and value_node.text is not None:
                    raw = value_node.text
                    if cell_type == "s":
                        value = shared[int(raw)]
                    else:
                        value = raw
                cells[col_idx] = value
            row_vals = [""] * (max_col + 1 if max_col >= 0 else 0)
            for idx, val in cells.items():
                row_vals[idx] = val
            rows.append(row_vals)
    return rows


def find_header_row(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows):
        row_norm = [str(x).strip() for x in row[:16]]
        has_id = ("ID" in row_norm) or ("ChEMBL ID" in row_norm)
        has_smiles = ("Smiles" in row_norm) or ("SMILES" in row_norm)
        has_xyz = ("X" in row_norm and "Y" in row_norm and "Z" in row_norm)
        has_posxyz = ("posX" in row_norm and "posY" in row_norm and "posZ" in row_norm)
        if has_id and (has_smiles or has_xyz or has_posxyz):
            return idx
    raise ValueError("Band header row not found")


def read_band_table(xlsx_path: str) -> pd.DataFrame:
    rows = read_first_sheet_rows(xlsx_path)
    header_idx = find_header_row(rows)
    header = [str(x).strip() for x in rows[header_idx]]
    width = len(header)
    body = []
    for row in rows[header_idx + 1 :]:
        vals = row[:width] + [""] * max(0, width - len(row))
        if not any(str(x).strip() for x in vals):
            continue
        body.append(vals[:width])
    df = pd.DataFrame(body, columns=header)
    df = df.replace({"": np.nan}).dropna(how="all")
    if "ID" in df.columns and "ChEMBL ID" not in df.columns:
        df = df.rename(columns={"ID": "ChEMBL ID"})
    df["ChEMBL ID"] = df["ChEMBL ID"].astype(str).str.strip()
    numeric_mask = df["ChEMBL ID"].str.fullmatch(r"\d+")
    df.loc[numeric_mask, "ChEMBL ID"] = "CHEMBL" + df.loc[numeric_mask, "ChEMBL ID"]
    df = df[df["ChEMBL ID"].str.startswith("CHEMBL", na=False)].reset_index(drop=True)
    df["band_order"] = np.arange(len(df))
    return df


def tanimoto_similarity_rows(x_bool: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    a = x_bool[idx_a]
    b = x_bool[idx_b]
    inter = np.logical_and(a, b).sum(axis=1)
    union = np.logical_or(a, b).sum(axis=1)
    return inter / np.clip(union, 1, None)


def summarize_adjacent_gradient(x_bool: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    idx_a = np.arange(0, len(x_bool) - 1)
    idx_b = idx_a + 1
    sim = tanimoto_similarity_rows(x_bool, idx_a, idx_b)
    edge_order = np.arange(1, len(sim) + 1)
    sim_spearman = float(spearmanr(edge_order, sim).statistic) if len(sim) >= 2 else math.nan
    drop_ratio = float(np.mean(sim[1:] <= sim[:-1])) if len(sim) >= 2 else math.nan
    df = pd.DataFrame({"edge_order": edge_order, "adjacent_similarity": sim})
    summary = {
        "adjacent_similarity_mean": float(sim.mean()),
        "adjacent_similarity_median": float(np.median(sim)),
        "adjacent_similarity_std": float(sim.std()),
        "adjacent_similarity_spearman": sim_spearman,
        "adjacent_similarity_drop_ratio": drop_ratio,
    }
    return df, summary


def order_local_neighborhood(coords: np.ndarray, selected_idx: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    selected_coords = coords[selected_idx]
    center = selected_coords.mean(axis=0, keepdims=True)
    centered = selected_coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    proj = centered @ axis
    order = np.argsort(proj)
    ordered_idx = selected_idx[order]
    ordered_coords = coords[ordered_idx]
    step_dist = np.linalg.norm(ordered_coords[1:] - ordered_coords[:-1], axis=1)
    spread = float(np.std(proj))
    return ordered_idx, {
        "local_axis_spread": spread,
        "mean_step_distance": float(step_dist.mean()) if len(step_dist) else math.nan,
        "median_step_distance": float(np.median(step_dist)) if len(step_dist) else math.nan,
        "path_score": float(step_dist.mean() / max(spread, 1e-8)) if len(step_dist) else math.inf,
    }


def sample_local_background_band(
    coords: np.ndarray,
    n_select: int,
    candidate_restarts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    tree = cKDTree(coords)
    best_idx = None
    best_info = None
    for _ in range(candidate_restarts):
        anchor_idx = int(rng.integers(0, len(coords)))
        _, selected_idx = tree.query(coords[anchor_idx], k=n_select, workers=-1)
        selected_idx = np.atleast_1d(selected_idx).astype(int)
        ordered_idx, info = order_local_neighborhood(coords, selected_idx)
        if best_idx is None or info["path_score"] < best_info["path_score"]:
            best_idx = ordered_idx
            best_info = info
    if best_idx is None:
        raise RuntimeError("Unable to sample a local background neighborhood")
    return best_idx, best_info


def empirical_background_summary(background_df: pd.DataFrame, actual: dict[str, float]) -> pd.DataFrame:
    rows = []
    metrics = [
        "adjacent_similarity_mean",
        "adjacent_similarity_median",
        "adjacent_similarity_spearman",
        "adjacent_similarity_drop_ratio",
    ]
    for method in sorted(background_df["method"].unique()):
        sub = background_df[background_df["method"] == method]
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            actual_val = float(actual[metric])
            if metric == "adjacent_similarity_spearman":
                empirical = float(np.mean(vals <= actual_val))
            else:
                empirical = float(np.mean(vals >= actual_val))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "actual_band_value": actual_val,
                    "background_mean": float(vals.mean()),
                    "background_std": float(vals.std()),
                    "background_q25": float(np.quantile(vals, 0.25)),
                    "background_median": float(np.quantile(vals, 0.50)),
                    "background_q75": float(np.quantile(vals, 0.75)),
                    "fold_vs_background_mean": float(actual_val / vals.mean()) if vals.mean() != 0 else math.nan,
                    "empirical_p_value": empirical,
                }
            )
    return pd.DataFrame(rows)


def draw_background_boxplots(background_df: pd.DataFrame, actual: dict[str, float], outpath: Path) -> None:
    metrics = [
        "adjacent_similarity_mean",
        "adjacent_similarity_median",
        "adjacent_similarity_spearman",
        "adjacent_similarity_drop_ratio",
    ]
    methods = sorted(background_df["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        data = [background_df[background_df["method"] == method][metric].dropna().to_numpy() for method in methods]
        ax.boxplot(data, tick_labels=methods, showfliers=False)
        ax.axhline(actual[metric], color="red", linestyle="--", linewidth=1.5, label="MolF-DAE band")
        ax.set_title(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_adjacent_curves(actual_edges: pd.DataFrame, background_edges: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(actual_edges["edge_order"], actual_edges["adjacent_similarity"], color="red", marker="o", label="MolF-DAE band")
    for method in sorted(background_edges["method"].unique()):
        sub = background_edges[background_edges["method"] == method]
        stats = sub.groupby("edge_order")["adjacent_similarity"].agg([
            "mean",
            lambda x: np.quantile(x, 0.25),
            lambda x: np.quantile(x, 0.75),
        ]).reset_index()
        stats.columns = ["edge_order", "mean", "q25", "q75"]
        ax.plot(stats["edge_order"], stats["mean"], label=f"{method} background mean")
        ax.fill_between(stats["edge_order"], stats["q25"], stats["q75"], alpha=0.15)
    ax.set_xlabel("Path edge order")
    ax.set_ylabel("Adjacent high-dimensional Tanimoto similarity")
    ax.set_title("Adjacent similarity along local band-sized backgrounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.fp_type)
    outdir = Path(args.outdir) if args.outdir else Path(f"/data/yinghuazhang/MolF-DAEs/code/control-review/result/clustering/{args.fp_type}/bg_band_gradient_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    df_band = read_band_table(paths["band"])
    df_label = pd.read_csv(paths["labels"]).copy()
    df_label["orig_idx"] = np.arange(len(df_label))
    chembl_to_idx = df_label.set_index("ChEMBL ID")["orig_idx"]
    df_band["orig_idx"] = df_band["ChEMBL ID"].map(chembl_to_idx)
    df_band = df_band.dropna(subset=["orig_idx"]).copy()
    df_band["orig_idx"] = df_band["orig_idx"].astype(int)
    df_band = df_band.drop_duplicates(subset=["ChEMBL ID"]).reset_index(drop=True)

    n_band = len(df_band)
    x_full = load(paths["data2"])
    x_band = np.asarray(x_full[df_band["orig_idx"].to_numpy()]).reshape(n_band, -1)
    x_band = (x_band > 0.5).astype(bool)

    actual_edges, actual_summary = summarize_adjacent_gradient(x_band)
    actual_summary.update({"fp_type": args.fp_type, "n_band_samples": int(n_band)})
    pd.DataFrame([actual_summary]).to_csv(outdir / "actual_band_summary.csv", index=False)
    actual_edges.to_csv(outdir / "actual_band_adjacent_similarity.csv", index=False)

    projections = {
        "PCA": pd.read_csv(paths["pca"])[["X", "Y", "Z"]].to_numpy(dtype=float),
        "UMAP": pd.read_csv(paths["umap"])[["X", "Y", "Z"]].to_numpy(dtype=float),
    }

    rng = np.random.default_rng(args.seed)
    bg_metric_rows = []
    bg_edge_rows = []
    for method, coords in projections.items():
        for repeat in range(args.n_repeats):
            selected_idx, info = sample_local_background_band(coords, n_band, args.candidate_restarts, rng)
            x_sel = np.asarray(x_full[selected_idx]).reshape(len(selected_idx), -1)
            x_sel = (x_sel > 0.5).astype(bool)
            edge_df, metrics = summarize_adjacent_gradient(x_sel)
            bg_metric_rows.append({
                "method": method,
                "repeat": repeat,
                "n_selected": int(len(selected_idx)),
                **info,
                **metrics,
            })
            tmp = edge_df.copy()
            tmp.insert(0, "repeat", repeat)
            tmp.insert(0, "method", method)
            bg_edge_rows.append(tmp)

    background_metrics = pd.DataFrame(bg_metric_rows)
    background_edges = pd.concat(bg_edge_rows, axis=0, ignore_index=True)
    background_metrics.to_csv(outdir / "background_local_path_metrics.csv", index=False)
    background_edges.to_csv(outdir / "background_local_path_adjacent_similarity.csv", index=False)

    comparison = empirical_background_summary(background_metrics, actual_summary)
    comparison.to_csv(outdir / "background_vs_actual_comparison.csv", index=False)
    draw_background_boxplots(background_metrics, actual_summary, outdir / "background_metric_boxplots.png")
    draw_adjacent_curves(actual_edges, background_edges, outdir / "actual_vs_background_adjacent_curves.png")

    print(pd.DataFrame([actual_summary]).to_string(index=False))
    print("\nBackground comparison")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
