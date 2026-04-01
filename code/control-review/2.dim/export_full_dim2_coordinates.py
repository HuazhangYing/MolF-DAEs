from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

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
from joblib import load
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape

BASE_DIR = Path('/data/yinghuazhang/MolF-DAEs')
MODEL_DIR = BASE_DIR / 'code' / 'control-review' / 'model'
OUTDIR = BASE_DIR / 'code' / 'control-review' / 'result' / '2.dim' / 'dim-2'

FINGERPRINTS = {
    'MACCSFP': {
        'data_path': BASE_DIR / 'dataset' / 'MACCSFP_molecule3.data2',
        'checkpoint_path': MODEL_DIR / 'maccsfp_autoencoder_dim2' / 'variables' / 'variables',
        'output_csv': OUTDIR / 'MACCSFP_latent_2D_full.csv',
        'input_shape': (13, 13, 1),
        'encoder_hidden': [512, 1024, 512, 256, 64, 32],
        'decoder_hidden': [32, 64, 256, 512, 1024, 512],
        'decoder_output': 169,
        'decoder_reshape': (13, 13),
    },
    'PharmacoPFP': {
        'data_path': BASE_DIR / 'dataset' / 'PharmacoPFP_molecule3.data2',
        'checkpoint_path': MODEL_DIR / 'pharmacopfp_autoencoder_dim2' / 'variables' / 'variables',
        'output_csv': OUTDIR / 'PharmacoPFP_latent_2D_full.csv',
        'input_shape': (17, 17, 1),
        'encoder_hidden': [512, 1024, 512, 256, 64, 32],
        'decoder_hidden': [32, 64, 256, 512, 1024, 512],
        'decoder_output': 289,
        'decoder_reshape': (17, 17),
    },
    'PubChemFP': {
        'data_path': BASE_DIR / 'dataset' / 'pubchem_molecule3.data2',
        'checkpoint_path': MODEL_DIR / 'pubchemfp_autoencoder_dim2' / 'variables' / 'variables',
        'output_csv': OUTDIR / 'PubChemFP_latent_2D_full.csv',
        'input_shape': (27, 27, 1),
        'encoder_hidden': [1024, 512, 128, 64, 32],
        'decoder_hidden': [32, 64, 128, 512, 1024],
        'decoder_output': 729,
        'decoder_reshape': (27, 27),
    },
}


class Encoder(Model):
    def __init__(self, hidden_units: list[int], dim: int):
        super().__init__()
        self.flatten = Flatten()
        self.hidden_layers = [Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = Dense(dim, activation='relu')

    def call(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class Decoder(Model):
    def __init__(self, hidden_units: list[int], output_units: int, reshape_dims: tuple[int, int]):
        super().__init__()
        self.hidden_layers = [Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = Dense(output_units, activation='sigmoid')
        self.reshape = Reshape(reshape_dims)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.reshape(x)


class Autoencoder(Model):
    def __init__(self, encoder_hidden: list[int], decoder_hidden: list[int], decoder_output: int, decoder_reshape: tuple[int, int], dim: int):
        super().__init__()
        self.encoder = Encoder(encoder_hidden, dim)
        self.decoder = Decoder(decoder_hidden, decoder_output, decoder_reshape)

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export full dim-2 latent coordinates for all fingerprint datasets.')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


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


def build_encoder(cfg: dict):
    model = Autoencoder(
        encoder_hidden=cfg['encoder_hidden'],
        decoder_hidden=cfg['decoder_hidden'],
        decoder_output=cfg['decoder_output'],
        decoder_reshape=cfg['decoder_reshape'],
        dim=2,
    )
    dummy = tf.zeros((1,) + tuple(cfg['input_shape']), dtype=tf.float32)
    _ = model(dummy)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(str(cfg['checkpoint_path'])).expect_partial()
    return model.encoder


def encode_full_dataset(encoder, x, batch_size: int) -> np.ndarray:
    n_rows = int(x.shape[0])
    latent = np.empty((n_rows, 2), dtype=np.float32)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = np.asarray(x[start:end]).astype('float32')
        z = encoder(batch, training=False).numpy().astype(np.float32)
        if z.shape[1] != 2:
            raise ValueError(f'Expected latent dim 2, got shape {z.shape}')
        latent[start:end] = z
        if start == 0 or end == n_rows or ((start // batch_size) + 1) % 20 == 0:
            print(f'  encoded rows {start:,} to {end:,} / {n_rows:,}')
    return latent


def export_one(name: str, cfg: dict, batch_size: int, overwrite: bool) -> dict:
    output_csv = cfg['output_csv']
    if output_csv.exists() and not overwrite:
        total_rows = sum(1 for _ in open(output_csv)) - 1
        print(f'Skipping {name}: existing file found at {output_csv}')
        return {
            'fingerprint': name,
            'n_rows': total_rows,
            'output_csv': str(output_csv),
            'checkpoint_path': str(cfg['checkpoint_path']),
            'data_path': str(cfg['data_path']),
            'status': 'skipped_existing',
            'columns': 'Unnamed: 0|X|Y',
        }

    print(f'Loading data for {name}: {cfg["data_path"]}')
    x = load(cfg['data_path'], mmap_mode='r')
    print(f'  dataset shape={x.shape}, dtype={x.dtype}')

    print(f'Restoring encoder for {name}: {cfg["checkpoint_path"]}')
    encoder = build_encoder(cfg)

    latent = encode_full_dataset(encoder, x, batch_size=batch_size)
    df = pd.DataFrame({
        'Unnamed: 0': np.arange(latent.shape[0], dtype=np.int64),
        'X': latent[:, 0],
        'Y': latent[:, 1],
    })
    df.to_csv(output_csv, index=False)
    print(f'  wrote {output_csv}')

    return {
        'fingerprint': name,
        'n_rows': int(latent.shape[0]),
        'output_csv': str(output_csv),
        'checkpoint_path': str(cfg['checkpoint_path']),
        'data_path': str(cfg['data_path']),
        'status': 'exported',
        'columns': 'Unnamed: 0|X|Y',
    }


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    configure_gpu()

    rows = []
    for name, cfg in FINGERPRINTS.items():
        rows.append(export_one(name, cfg, batch_size=args.batch_size, overwrite=args.overwrite))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTDIR / 'dim2_full_export_manifest.csv', index=False)
    print('Saved manifest to', OUTDIR / 'dim2_full_export_manifest.csv')


if __name__ == '__main__':
    main()
