from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

# Preload CUDA libs before importing tensorflow so saved models can use GPU when available.
LIB_PATH = "/home/yinghuazhang/miniconda3/envs/daes/lib"
LIBS = [
    "libcudart.so.11.0",
    "libcublas.so.11",
    "libcublasLt.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.11",
    "libcudnn.so.8",
]
for lib in LIBS:
    full_path = os.path.join(LIB_PATH, lib)
    if os.path.exists(full_path):
        try:
            ctypes.CDLL(full_path)
        except OSError:
            pass

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load

DIMS = [2, 3, 4, 8, 16, 32]
BASE_DIR = Path('/data/yinghuazhang/MolF-DAEs')
MODEL_DIR = BASE_DIR / 'code' / 'control-review' / 'model'
RESULT_COMPARISON_DIR = BASE_DIR / 'result' / 'comparison'

FINGERPRINTS = {
    'MACCSFP': {
        'result_dir': BASE_DIR / 'result' / 'maccsfp',
        'data_path': BASE_DIR / 'dataset' / 'MACCSFP_molecule3.data2',
        'prefix': 'maccsfp',
        'model_prefix': 'maccsfp_autoencoder_dim',
        'seed': 42,
    },
    'PharmacoPFP': {
        'result_dir': BASE_DIR / 'result' / 'pharmachopfp',
        'data_path': BASE_DIR / 'dataset' / 'PharmacoPFP_molecule3.data2',
        'prefix': 'pharmachopfp',
        'model_prefix': 'pharmacopfp_autoencoder_dim',
        'seed': 42,
    },
    'PubChemFP': {
        'result_dir': BASE_DIR / 'result' / 'pubchemfp',
        'data_path': BASE_DIR / 'dataset' / 'pubchem_molecule3.data2',
        'prefix': 'pubchemfp',
        'model_prefix': 'pubchemfp_autoencoder_dim',
        'seed': 42,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Recover latent vectors from saved models and export comparison-style coordinate tables.')
    parser.add_argument('--sample-size', type=int, default=80000)
    parser.add_argument('--dims', type=int, nargs='*', default=DIMS)
    parser.add_argument('--force-recompute', action='store_true', help='Recompute latent vectors even if joblib files already exist.')
    return parser.parse_args()


def coordinate_columns(dim: int) -> list[str]:
    base = ['X', 'Y', 'Z']
    cols = []
    for i in range(dim):
        if i < len(base):
            cols.append(base[i])
        else:
            cols.append(f'C{i + 1}')
    return cols


def build_sample_index_and_matrix(data_path: Path, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    x = load(data_path, mmap_mode='r')
    n_rows = int(x.shape[0])
    if sample_size > n_rows:
        raise ValueError(f'sample_size={sample_size} exceeds row count {n_rows} for {data_path}')
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n_rows, sample_size, replace=False).astype(np.int64)
    x_sample = np.asarray(x[sample_idx]).astype('float32')
    return sample_idx, x_sample, n_rows


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPU detected; export will run on CPU.')
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f'Using {len(gpus)} visible GPU(s).')


def load_or_recompute_latent(model_path: Path, latent_path: Path, x_sample: np.ndarray, force_recompute: bool) -> np.ndarray:
    if latent_path.exists() and not force_recompute:
        return np.asarray(load(latent_path))

    model = tf.keras.models.load_model(model_path)
    if not hasattr(model, 'encoder'):
        raise AttributeError(f'Loaded model at {model_path} does not expose an encoder attribute.')
    latent = np.asarray(model.encoder(x_sample).numpy())
    dump(latent, latent_path)
    return latent


def main() -> None:
    args = parse_args()
    RESULT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    configure_gpu()

    fingerprint_payloads: dict[str, dict[str, object]] = {}
    row_count_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for fingerprint, cfg in FINGERPRINTS.items():
        sample_idx, x_sample, n_rows = build_sample_index_and_matrix(cfg['data_path'], args.sample_size, cfg['seed'])
        fingerprint_payloads[fingerprint] = {
            'sample_idx': sample_idx,
            'x_sample': x_sample,
            'n_rows': n_rows,
        }
        row_count_rows.append({
            'fingerprint': fingerprint,
            'n_rows': n_rows,
            'sample_size': args.sample_size,
        })

    ref_name = next(iter(fingerprint_payloads))
    ref_idx = fingerprint_payloads[ref_name]['sample_idx']
    aligned = all(np.array_equal(ref_idx, payload['sample_idx']) for payload in fingerprint_payloads.values())

    summary_df = pd.DataFrame(row_count_rows)
    summary_df['same_sample_index_as_reference'] = [
        np.array_equal(ref_idx, fingerprint_payloads[name]['sample_idx']) for name in summary_df['fingerprint']
    ]
    summary_df.to_csv(RESULT_COMPARISON_DIR / 'latent_coordinate_export_summary.csv', index=False)

    pd.DataFrame({
        'sample_order': np.arange(args.sample_size, dtype=np.int64),
        'sample_idx': ref_idx,
    }).to_csv(RESULT_COMPARISON_DIR / 'latent_common_sample_idx.csv', index=False)

    for fingerprint, cfg in FINGERPRINTS.items():
        result_dir = cfg['result_dir']
        prefix = cfg['prefix']
        model_prefix = cfg['model_prefix']
        x_sample = fingerprint_payloads[fingerprint]['x_sample']

        for dim in args.dims:
            run_dir = result_dir / f'test9-{dim}D'
            run_dir.mkdir(parents=True, exist_ok=True)
            latent_path = run_dir / 'latent_vectors.joblib'
            model_path = MODEL_DIR / f'{model_prefix}{dim}'
            if not model_path.exists():
                raise FileNotFoundError(f'Missing saved model: {model_path}')

            latent = load_or_recompute_latent(model_path, latent_path, x_sample, args.force_recompute)
            if latent.shape[0] != args.sample_size:
                raise ValueError(f'{latent_path} has {latent.shape[0]} rows, expected {args.sample_size}')
            if latent.ndim != 2 or latent.shape[1] != dim:
                raise ValueError(f'{latent_path} has shape {latent.shape}, expected (*, {dim})')

            cols = coordinate_columns(dim)
            df = pd.DataFrame(latent, columns=cols)

            per_run_csv = run_dir / 'latent_coordinates_ME.csv'
            comparison_csv = RESULT_COMPARISON_DIR / f'{prefix}_latent_{dim}D_ME.csv'
            df.to_csv(per_run_csv)
            df.to_csv(comparison_csv)

            manifest_rows.append({
                'fingerprint': fingerprint,
                'dim': dim,
                'n_rows': latent.shape[0],
                'n_coords': latent.shape[1],
                'aligned_common_sample_idx': aligned,
                'model_path': str(model_path),
                'run_dir': str(run_dir),
                'latent_joblib': str(latent_path),
                'per_run_csv': str(per_run_csv),
                'comparison_csv': str(comparison_csv),
                'columns': '|'.join(cols),
            })

    pd.DataFrame(manifest_rows).sort_values(['fingerprint', 'dim']).to_csv(
        RESULT_COMPARISON_DIR / 'latent_coordinate_manifest.csv',
        index=False,
    )
    print('Exported latent coordinate tables successfully.')
    print(f'Common sample index aligned across fingerprints: {aligned}')


if __name__ == '__main__':
    main()
