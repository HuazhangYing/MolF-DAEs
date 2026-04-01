from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

MOL2VEC_SOURCE_DIR = Path("/data/yinghuazhang/MolF-DAEs/code/control-review/code/mol2vec")
MOL2VEC_DEFAULT_MODEL_PATH = MOL2VEC_SOURCE_DIR / "examples" / "models" / "model_300dim.pkl"

RDLogger.DisableLog("rdApp.warning")


def _ensure_mol2vec_source_on_path() -> None:
    source_root = str(MOL2VEC_SOURCE_DIR)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


def load_mol2vec_resources(model_path: str | Path):
    _ensure_mol2vec_source_on_path()
    try:
        from gensim.models import word2vec
    except ImportError as exc:
        raise ImportError(
            "mol2vec needs gensim to load the pretrained Word2Vec model. "
            "Please install gensim in the runtime environment first."
        ) from exc

    from mol2vec.features import mol2alt_sentence

    model = word2vec.Word2Vec.load(str(model_path))
    return {
        "model": model,
        "mol2alt_sentence": mol2alt_sentence,
        "vector_size": int(model.wv.vector_size),
        "vocab": set(model.wv.key_to_index.keys()) if hasattr(model.wv, "key_to_index") else set(model.wv.vocab.keys()),
    }


def preprocess_smiles_batch(smiles_list: Sequence[object], radius: int, sentence_builder) -> Tuple[np.ndarray, List[List[str]], Dict[str, int]]:
    valid_local_ids: List[int] = []
    sentences: List[List[str]] = []
    summary = {
        "valid": 0,
        "missing": 0,
        "invalid_smiles": 0,
        "empty_sentence": 0,
    }

    for idx, raw_smiles in enumerate(smiles_list):
        if raw_smiles is None:
            summary["missing"] += 1
            continue

        smiles = str(raw_smiles).strip()
        if not smiles or smiles.lower() == "nan":
            summary["missing"] += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            summary["invalid_smiles"] += 1
            continue

        canonical = Chem.MolToSmiles(mol)
        canonical_mol = Chem.MolFromSmiles(canonical)
        if canonical_mol is None:
            summary["invalid_smiles"] += 1
            continue

        sentence = list(sentence_builder(canonical_mol, radius))
        if not sentence:
            summary["empty_sentence"] += 1
            continue

        valid_local_ids.append(idx)
        sentences.append(sentence)
        summary["valid"] += 1

    return np.asarray(valid_local_ids, dtype=np.int64), sentences, summary


def sentences_to_vectors(
    sentences: Sequence[Sequence[str]],
    model,
    vocab: set[str],
    unseen_token: str | None = "UNK",
) -> Tuple[np.ndarray, Dict[str, float]]:
    vector_size = int(model.wv.vector_size)
    embeddings = np.zeros((len(sentences), vector_size), dtype=np.float32)

    unseen_vec = None
    if unseen_token is not None:
        try:
            unseen_vec = np.asarray(model.wv.word_vec(unseen_token), dtype=np.float32)
        except KeyError:
            unseen_vec = None

    total_tokens = 0
    recognized_tokens = 0
    oov_tokens = 0
    zero_vector_rows = 0
    sentence_lengths: List[int] = []

    for row_idx, sentence in enumerate(sentences):
        acc = np.zeros(vector_size, dtype=np.float32)
        sentence_len = len(sentence)
        sentence_lengths.append(sentence_len)
        total_tokens += sentence_len

        for token in sentence:
            if token in vocab:
                acc += np.asarray(model.wv.word_vec(token), dtype=np.float32)
                recognized_tokens += 1
            elif unseen_vec is not None:
                acc += unseen_vec
                oov_tokens += 1
            else:
                oov_tokens += 1

        embeddings[row_idx] = acc
        if not np.any(acc):
            zero_vector_rows += 1

    metrics = {
        "total_tokens": float(total_tokens),
        "recognized_tokens": float(recognized_tokens),
        "oov_tokens": float(oov_tokens),
        "oov_ratio": float(oov_tokens / total_tokens) if total_tokens else 0.0,
        "zero_vector_rows": float(zero_vector_rows),
        "mean_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "max_sentence_length": float(np.max(sentence_lengths)) if sentence_lengths else 0.0,
    }
    return embeddings, metrics
