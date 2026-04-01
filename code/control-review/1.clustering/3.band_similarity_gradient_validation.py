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
from scipy.stats import spearmanr

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether the band / continuous 3D structure reflects a high-dimensional similarity gradient."
    )
    parser.add_argument("--fp-type", choices=["pubchem", "maccs", "pharma"], default="pubchem")
    parser.add_argument("--max-lag", type=int, default=40)
    parser.add_argument("--random-pairs", type=int, default=20000)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to result/clustering/{fp_type}/band_gradient_validation",
    )
    return parser.parse_args()


def resolve_paths(fp_type: str) -> dict[str, str]:
    labels = "/data/yinghuazhang/MolF-DAEs/dataset/190w_3D_label_dropna.csv"
    if fp_type == "pubchem":
        return {
            "labels": labels,
            "data2": "/data/yinghuazhang/MolF-DAEs/dataset/pubchem_molecule3.data2",
            "dae_3d": "/data/yinghuazhang/MolF-DAEs/result/pubchemfp/test1_best_pubchem/test1_ME.csv",
            "band": "/data/yinghuazhang/MolF-DAEs/dataset/band1-pubchemfp.xlsx",
        }
    if fp_type == "maccs":
        return {
            "labels": labels,
            "data2": "/data/yinghuazhang/MolF-DAEs/dataset/MACCSFP_molecule3.data2",
            "dae_3d": "/data/yinghuazhang/MolF-DAEs/result/maccsfp/test9_data_best/test9_ME.csv",
            "band": "/data/yinghuazhang/MolF-DAEs/dataset/band5-maccsfp.xlsx",
        }
    return {
        "labels": labels,
        "data2": "/data/yinghuazhang/MolF-DAEs/dataset/PharmacoPFP_molecule3.data2",
        "dae_3d": "/data/yinghuazhang/MolF-DAEs/result/pharmachopfp/test1_data_best_pharmacopfp/test1_ME_pharmacopfp.csv",
        "band": "/data/yinghuazhang/MolF-DAEs/dataset/band6-pharmacopfp.xlsx",
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
    if 'ID' in df.columns and 'ChEMBL ID' not in df.columns:
        df = df.rename(columns={'ID': 'ChEMBL ID'})
    if 'SMILES' in df.columns and 'Smiles' not in df.columns:
        df = df.rename(columns={'SMILES': 'Smiles'})

    keep_cols = [col for col in ['ChEMBL ID', 'type', 'Smiles', 'Protease', 'Nuclear receptor', 'kinase', 'G-protein coupled receptor', 'X', 'Y', 'Z', 'posX', 'posY', 'posZ'] if col in df.columns]
    df = df[keep_cols].copy()

    df['ChEMBL ID'] = df['ChEMBL ID'].astype(str).str.strip()
    numeric_mask = df['ChEMBL ID'].str.fullmatch(r'\d+')
    df.loc[numeric_mask, 'ChEMBL ID'] = 'CHEMBL' + df.loc[numeric_mask, 'ChEMBL ID']
    df = df[df['ChEMBL ID'].str.startswith('CHEMBL', na=False)].reset_index(drop=True)
    df['band_order'] = np.arange(len(df))
    return df


def tanimoto_similarity_matrix_rows(x_bool: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    a = x_bool[idx_a]
    b = x_bool[idx_b]
    inter = np.logical_and(a, b).sum(axis=1)
    union = np.logical_or(a, b).sum(axis=1)
    return inter / np.clip(union, 1, None)


def summarize_by_lag(x_bool: np.ndarray, max_lag: int) -> pd.DataFrame:
    rows = []
    n = len(x_bool)
    upper_lag = min(max_lag, n - 1)
    for lag in range(1, upper_lag + 1):
        idx_a = np.arange(0, n - lag)
        idx_b = idx_a + lag
        sim = tanimoto_similarity_matrix_rows(x_bool, idx_a, idx_b)
        rows.append(
            {
                "lag": lag,
                "n_pairs": len(sim),
                "mean_similarity": float(sim.mean()),
                "median_similarity": float(np.median(sim)),
                "q25_similarity": float(np.quantile(sim, 0.25)),
                "q75_similarity": float(np.quantile(sim, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def sample_random_pairs(n: int, n_pairs: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs, endpoint=False)
    j = rng.integers(0, n, size=n_pairs, endpoint=False)
    valid = i != j
    return i[valid], j[valid]


def pairwise_band_distance_summary(x_bool: np.ndarray, random_pairs: int) -> tuple[pd.DataFrame, dict[str, float]]:
    n = len(x_bool)
    adjacent_i = np.arange(0, n - 1)
    adjacent_j = adjacent_i + 1
    near_i = np.arange(0, max(0, n - 3))
    near_j = near_i + 3
    far_i = np.arange(0, max(0, n - 20))
    far_j = far_i + 20
    rnd_i, rnd_j = sample_random_pairs(n, random_pairs)

    groups = {
        "adjacent_lag1": tanimoto_similarity_matrix_rows(x_bool, adjacent_i, adjacent_j),
        "near_lag3": tanimoto_similarity_matrix_rows(x_bool, near_i, near_j) if len(near_i) else np.array([]),
        "far_lag20": tanimoto_similarity_matrix_rows(x_bool, far_i, far_j) if len(far_i) else np.array([]),
        "random": tanimoto_similarity_matrix_rows(x_bool, rnd_i, rnd_j),
    }

    rows = []
    for group, vals in groups.items():
        if len(vals) == 0:
            continue
        rows.append(
            {
                "group": group,
                "n_pairs": len(vals),
                "mean_similarity": float(vals.mean()),
                "median_similarity": float(np.median(vals)),
                "q25_similarity": float(np.quantile(vals, 0.25)),
                "q75_similarity": float(np.quantile(vals, 0.75)),
            }
        )
    stats = {
        "adjacent_minus_random": float(groups["adjacent_lag1"].mean() - groups["random"].mean()),
        "adjacent_minus_far": float(groups["adjacent_lag1"].mean() - groups["far_lag20"].mean()) if len(groups["far_lag20"]) else math.nan,
    }
    return pd.DataFrame(rows), stats


def compute_3d_path_metrics(coords: np.ndarray) -> dict[str, float]:
    step = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    axis_ranges = coords.max(axis=0) - coords.min(axis=0)
    return {
        "mean_adjacent_3d_step": float(step.mean()),
        "median_adjacent_3d_step": float(np.median(step)),
        "3d_path_length": float(step.sum()),
        "3d_bbox_x": float(axis_ranges[0]),
        "3d_bbox_y": float(axis_ranges[1]),
        "3d_bbox_z": float(axis_ranges[2]),
        "3d_cumulative_end": float(cumulative[-1]),
    }


def draw_band_path(coords: np.ndarray, df_band: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    color = np.arange(len(coords))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, cmap='viridis', s=20)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.8, alpha=0.6)
    ax.set_title('Band order in MolF-DAE 3D space')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def draw_lag_curve(df_lag: pd.DataFrame, corr: float, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_lag['lag'], df_lag['mean_similarity'], marker='o', label='Mean Tanimoto similarity')
    ax.fill_between(df_lag['lag'], df_lag['q25_similarity'], df_lag['q75_similarity'], alpha=0.2)
    ax.set_xlabel('Band index lag')
    ax.set_ylabel('High-dimensional Tanimoto similarity')
    ax.set_title(f'Similarity gradient along the band (Spearman={corr:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def draw_group_bars(df_groups: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_groups['group'], df_groups['mean_similarity'], color=['#1b9e77', '#66a61e', '#d95f02', '#7570b3'][: len(df_groups)])
    ax.set_ylabel('Mean Tanimoto similarity')
    ax.set_title('Adjacent vs near vs far vs random similarity')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.fp_type)
    outdir = Path(args.outdir) if args.outdir else Path(f'/data/yinghuazhang/MolF-DAEs/code/control-review/result/clustering/{args.fp_type}/band_gradient_validation')
    outdir.mkdir(parents=True, exist_ok=True)

    df_band = read_band_table(paths['band'])
    df_label = pd.read_csv(paths['labels']).copy()
    df_label['orig_idx'] = np.arange(len(df_label))
    chembl_to_idx = df_label.set_index('ChEMBL ID')['orig_idx']

    df_band['orig_idx'] = df_band['ChEMBL ID'].map(chembl_to_idx)
    df_band = df_band.dropna(subset=['orig_idx']).copy()
    df_band['orig_idx'] = df_band['orig_idx'].astype(int)
    df_band = df_band.drop_duplicates(subset=['ChEMBL ID']).reset_index(drop=True)

    df_dae = pd.read_csv(paths['dae_3d']).copy()
    x_full = load(paths['data2'])
    x_band = np.asarray(x_full[df_band['orig_idx'].to_numpy()]).reshape(len(df_band), -1)
    x_band = (x_band > 0.5).astype(bool)
    coords = df_dae.loc[df_band['orig_idx'], ['X', 'Y', 'Z']].to_numpy(dtype=float)

    df_band['dae_X'] = coords[:, 0]
    df_band['dae_Y'] = coords[:, 1]
    df_band['dae_Z'] = coords[:, 2]
    df_band.to_csv(outdir / 'band_matched_samples.csv', index=False)

    df_lag = summarize_by_lag(x_band, args.max_lag)
    lag_corr = float(spearmanr(df_lag['lag'], df_lag['mean_similarity']).statistic)
    df_lag.to_csv(outdir / 'band_lag_similarity.csv', index=False)

    df_groups, group_stats = pairwise_band_distance_summary(x_band, args.random_pairs)
    df_groups.to_csv(outdir / 'band_group_similarity_summary.csv', index=False)

    path_metrics = compute_3d_path_metrics(coords)
    summary = {
        'fp_type': args.fp_type,
        'n_band_samples': int(len(df_band)),
        'lag_similarity_spearman': lag_corr,
        'adjacent_minus_random_similarity': group_stats['adjacent_minus_random'],
        'adjacent_minus_far_similarity': group_stats['adjacent_minus_far'],
        **path_metrics,
    }
    pd.DataFrame([summary]).to_csv(outdir / 'band_gradient_summary.csv', index=False)

    draw_band_path(coords, df_band, outdir / 'band_3d_path.png')
    draw_lag_curve(df_lag, lag_corr, outdir / 'band_lag_similarity_curve.png')
    draw_group_bars(df_groups, outdir / 'band_similarity_groups.png')

    print(pd.DataFrame([summary]).to_string(index=False))
    print('\nBand group summary')
    print(df_groups.to_string(index=False))
    print(f'\nSaved outputs to: {outdir}')


if __name__ == '__main__':
    main()
