from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ALLOWED_CHARACTERS = {
    "c", "3", "5", "=", "O", "+", "8", "[", "s", "]", "C", "n", "p", "H",
    "%", "-", "(", "I", "1", "4", "Y", "X", "6", "!", "S", ")", "#", "o",
    "2", "P", "N", "7", "F",
}
MAX_SMI_LEN = 111
MAX_RAW_SMILES_LEN = 109


@dataclass(frozen=True)
class MolAIResources:
    char_to_int: Dict[str, int]
    int_to_char: Dict[int, str]
    char_to_int_len: int
    smiles_to_latent_model: object


def build_legacy_molai_encoder(char_to_int_len: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    encoder_inputs = keras.Input(shape=(110, char_to_int_len), name="encoder_input")
    masked = layers.Masking(mask_value=13, name="masking")(encoder_inputs)
    out1, h1, c1 = layers.LSTM(1024, return_sequences=True, return_state=True, name="encoder_LSTM1")(masked)
    out2, h2, c2 = layers.LSTM(1024, return_sequences=True, return_state=True, name="encoder_LSTM2")(out1)
    _, h3, c3 = layers.LSTM(1024, return_sequences=False, return_state=True, name="encoder_LSTM3")(out2)
    concat = layers.Concatenate(name="encoder_concatenate")([h1, c1, h2, c2, h3, c3])
    latent = layers.Dense(512, activation="tanh", name="latent")(concat)
    return keras.Model(encoder_inputs, latent, name="model_1")


def load_molai_resources(model_dir: str | Path) -> MolAIResources:
    model_dir = Path(model_dir)
    with (model_dir / "char_to_int.pkl").open("rb") as handle:
        char_to_int = pickle.load(handle)
    with (model_dir / "int_to_char.pkl").open("rb") as handle:
        int_to_char = pickle.load(handle)

    model = build_legacy_molai_encoder(len(char_to_int))
    model.load_weights(model_dir / "smi2lat_epoch_6.h5")
    return MolAIResources(
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        char_to_int_len=len(char_to_int),
        smiles_to_latent_model=model,
    )


def _normalize_smiles(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None

    text = str(raw_value).strip()
    if not text:
        return None
    return text.replace("Cl", "X").replace("Br", "Y")


def preprocess_smiles_batch(smiles: Sequence[object]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    escaped_characters = "".join(re.escape(char) for char in sorted(ALLOWED_CHARACTERS))
    invalid_regex = re.compile(f"[^{escaped_characters}]")

    valid_positions: List[int] = []
    normalized_smiles: List[str] = []
    summary = {
        "valid": 0,
        "missing": 0,
        "too_short": 0,
        "too_long": 0,
        "unsupported_characters": 0,
    }

    for idx, raw_value in enumerate(smiles):
        normalized = _normalize_smiles(raw_value)
        if normalized is None:
            summary["missing"] += 1
            continue
        if len(normalized) <= 2:
            summary["too_short"] += 1
            continue
        if len(normalized) > MAX_RAW_SMILES_LEN:
            summary["too_long"] += 1
            continue
        if invalid_regex.search(normalized):
            summary["unsupported_characters"] += 1
            continue

        valid_positions.append(idx)
        normalized_smiles.append(normalized)
        summary["valid"] += 1

    return np.asarray(valid_positions, dtype=np.int64), normalized_smiles, summary


def vectorize_smiles(smiles: Sequence[str], char_to_int: Dict[str, int], char_to_int_len: int) -> np.ndarray:
    one_hot = np.zeros((len(smiles), MAX_SMI_LEN, char_to_int_len), dtype=np.int8)

    for i, smile in enumerate(smiles):
        one_hot[i, 0, char_to_int["!"]] = 1
        for j, char in enumerate(smile):
            one_hot[i, j + 1, char_to_int[char]] = 1
        one_hot[i, len(smile) + 1, char_to_int["$"]] = 1
        one_hot[i, len(smile) + 2 :, char_to_int["%"]] = 1

    return one_hot[:, 1:, :]


def smiles_to_latent(smiles: Sequence[str], resources: MolAIResources, predict_batch_size: int) -> np.ndarray:
    vectorized = vectorize_smiles(smiles, resources.char_to_int, resources.char_to_int_len)
    latent = resources.smiles_to_latent_model.predict(vectorized, batch_size=predict_batch_size, verbose=0)
    return np.asarray(latent, dtype=np.float32)
