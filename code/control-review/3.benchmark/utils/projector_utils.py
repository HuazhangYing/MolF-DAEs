from pathlib import Path
import inspect
from typing import Any, Dict

import numpy as np

LOCAL_PHATE_PYTHON_DIR = Path('/data/yinghuazhang/MolF-DAEs/code/control-review/code/PHATE/Python')
LOCAL_PACMAP_SOURCE_DIR = Path('/data/yinghuazhang/MolF-DAEs/code/control-review/code/PaCMAP/source')


def _import_local_pacmap():
    import sys

    local_path = str(LOCAL_PACMAP_SOURCE_DIR)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)
    import pacmap  # type: ignore

    return pacmap


def _import_local_phate():
    import sys

    local_path = str(LOCAL_PHATE_PYTHON_DIR)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)
    import phate  # type: ignore

    return phate


def fit_transform_projection(
    method: str,
    latent: np.ndarray,
    params: Dict[str, Any],
    random_state: int,
) -> np.ndarray:
    if method == "pacmap3d":
        try:
            pacmap = _import_local_pacmap()
        except Exception:
            try:
                import pacmap
            except ImportError as exc:
                raise ImportError("pacmap 无法导入。既没有可用的 pip 包，也无法从本地 PaCMAP 源码目录加载。") from exc

        pacmap_distance = params["distance"]
        if pacmap_distance == "cosine":
            pacmap_distance = "angular"

        pacmap_kwargs = {
            "n_components": 3,
            "n_neighbors": params["n_neighbors"],
            "MN_ratio": params["MN_ratio"],
            "FP_ratio": params["FP_ratio"],
            "distance": pacmap_distance,
            "apply_pca": params["apply_pca"],
            "init": params.get("init"),
            "random_state": random_state,
        }
        supported = inspect.signature(pacmap.PaCMAP.__init__).parameters
        pacmap_kwargs = {k: v for k, v in pacmap_kwargs.items() if k in supported and v is not None}
        projector = pacmap.PaCMAP(**pacmap_kwargs)
        coords = projector.fit_transform(latent)
    elif method == "phate3d":
        try:
            import phate
        except ImportError:
            try:
                phate = _import_local_phate()
            except Exception as exc:
                raise ImportError(
                    "phate 无法导入。既没有可用的 pip 包，也无法从本地 PHATE 源码目录加载依赖。"
                ) from exc

        projector = phate.PHATE(
            n_components=3,
            knn=params["knn"],
            decay=params["decay"],
            gamma=params["gamma"],
            t=params["t"],
            n_landmark=params["n_landmark"],
            knn_dist=params["distance"],
            random_state=random_state,
            verbose=0,
        )
        coords = projector.fit_transform(latent)
    else:
        raise ValueError(f"Unsupported projector method: {method}")

    return np.asarray(coords, dtype=np.float32)
