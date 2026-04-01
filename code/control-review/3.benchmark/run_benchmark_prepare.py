import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from utils.io_utils import ensure_dir, run_is_complete, save_npy, stable_params_hash, utc_timestamp, write_json, write_log
from utils.latent_utils import (
    get_or_create_subset_ids,
    get_total_size,
    load_config,
    load_subset_latent,
    parameter_grid,
    resolve_feature_spec,
    source_latent_descriptor,
)
from utils.projector_utils import fit_transform_projection
from utils.reconstruction_utils import compute_reconstruction_summary

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "benchmark_config.yaml"
SUPPORTED_SUBSETS = ("full", "100k", "300k", "1m")
SUPPORTED_METHODS = ("pacmap3d", "phate3d", "gtm3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare latent-based 3D benchmark artifacts for MolF-DAEs.")
    parser.add_argument("--feature-group", choices=["maccsfp", "pubchemfp", "pharmacopfp", "mol2vec", "all"], default="maccsfp", help="选择要处理的特征组。")
    parser.add_argument("--method", choices=[*SUPPORTED_METHODS, "all"], required=True, help="选择投影方法。gtm3d 当前只创建保留目录骨架。")
    parser.add_argument("--subset", choices=[*SUPPORTED_SUBSETS, "all"], required=True, help="选择样本子集。")
    parser.add_argument("--force", action="store_true", help="即使目标结果已存在也重新生成。")
    return parser.parse_args()


def selected_feature_groups(args: argparse.Namespace) -> List[str]:
    return ["maccsfp", "pubchemfp", "pharmacopfp", "mol2vec"] if args.feature_group == "all" else [args.feature_group]


def selected_methods(args: argparse.Namespace) -> List[str]:
    return list(SUPPORTED_METHODS) if args.method == "all" else [args.method]


def selected_subsets(args: argparse.Namespace) -> List[str]:
    return list(SUPPORTED_SUBSETS) if args.subset == "all" else [args.subset]


def validate_method_subset(method: str, subset: str, config: Dict[str, Any]) -> bool:
    if method == "gtm3d":
        return subset in config["methods"]["gtm3d"]["reserved_subsets"]
    return True


def run_dir_for(method: str, feature_group: str, subset: str, params_hash: str) -> Path:
    return BASE_DIR / "outputs" / method / feature_group / subset / f"params_{params_hash}"


def scaffold_gtm_dirs(feature_group: str, subset: str) -> None:
    run_dir = BASE_DIR / "outputs" / "gtm3d" / feature_group / subset
    ensure_dir(run_dir)
    write_json(
        run_dir / "reservation.json",
        {
            "feature_group": feature_group,
            "method": "gtm3d",
            "subset_name": subset,
            "status": "reserved_only",
            "timestamp": utc_timestamp(),
            "note": "GTM 仅保留目录骨架，当前阶段未实现。",
        },
    )
    write_log(
        run_dir / "run_log.txt",
        [
            f"[{utc_timestamp()}] Reserved GTM folder created.",
            "Status: reserved_only",
            "No coordinates were generated in this stage.",
        ],
    )


def prepare_projection_run(*, method: str, feature_spec: Dict[str, Any], subset_name: str, subset_ids, params: Dict[str, Any], runtime_config: Dict[str, Any], reconstruction_summary: Dict[str, Any], force: bool) -> bool:
    params_hash = stable_params_hash(params)
    run_dir = run_dir_for(method, feature_spec["feature_group"], subset_name, params_hash)
    params_payload = {
        "method": method,
        "feature_group": feature_spec["feature_group"],
        "subset_name": subset_name,
        "n_samples": int(subset_ids.shape[0]),
        "params": params,
        "params_hash": params_hash,
        "timestamp": utc_timestamp(),
        "source_latent": source_latent_descriptor(feature_spec),
        "reference_notebook": feature_spec["reference_notebook"],
    }

    if run_is_complete(run_dir) and not force:
        return False

    ensure_dir(run_dir)
    log_lines = [
        f"[{utc_timestamp()}] Start run.",
        f"feature_group={feature_spec['feature_group']}",
        f"method={method}",
        f"subset_name={subset_name}",
        f"n_samples={subset_ids.shape[0]}",
        f"params_hash={params_hash}",
        f"source_latent={source_latent_descriptor(feature_spec)}",
    ]

    try:
        latent_subset = load_subset_latent(feature_spec, subset_ids)
        coords = fit_transform_projection(method=method, latent=latent_subset, params=params, random_state=int(runtime_config["projector_random_state"]))
        save_npy(run_dir / "coords_3d.npy", coords)
        save_npy(run_dir / "ids.npy", subset_ids)
        write_json(run_dir / "params.json", params_payload)
        write_json(run_dir / "reconstruction_summary.json", reconstruction_summary)
        log_lines.append(f"[{utc_timestamp()}] Finished successfully.")
        write_log(run_dir / "run_log.txt", log_lines)
        return True
    except Exception as exc:
        log_lines.append(f"[{utc_timestamp()}] Failed: {exc}")
        log_lines.append(traceback.format_exc())
        write_json(run_dir / "params.json", params_payload)
        write_json(run_dir / "reconstruction_summary.json", reconstruction_summary)
        write_log(run_dir / "run_log.txt", log_lines)
        raise


def main() -> int:
    args = parse_args()
    config = load_config(CONFIG_PATH)
    subset_policy = config["subset_policy"]
    runtime_config = config["runtime"]

    failures: List[str] = []
    completed = 0
    skipped = 0

    for feature_group in selected_feature_groups(args):
        feature_spec = resolve_feature_spec(config, feature_group)
        total_size = get_total_size(feature_spec)

        for subset_name in selected_subsets(args):
            for method in selected_methods(args):
                if not validate_method_subset(method, subset_name, config):
                    continue

                subset_ids = get_or_create_subset_ids(base_dir=BASE_DIR, feature_group=feature_group, subset_name=subset_name, total_size=total_size, subset_policy=subset_policy)

                if method == "gtm3d":
                    scaffold_gtm_dirs(feature_group, subset_name)
                    continue

                reconstruction_summary = compute_reconstruction_summary(base_dir=BASE_DIR, feature_spec=feature_spec, subset_name=subset_name, subset_ids=subset_ids, force=args.force)

                for params in parameter_grid(config["methods"][method]):
                    try:
                        changed = prepare_projection_run(method=method, feature_spec=feature_spec, subset_name=subset_name, subset_ids=subset_ids, params=params, runtime_config=runtime_config, reconstruction_summary=reconstruction_summary, force=args.force)
                        if changed:
                            completed += 1
                        else:
                            skipped += 1
                    except Exception as exc:
                        failures.append(f"{feature_group}/{method}/{subset_name}/{stable_params_hash(params)}: {exc}")

    print(f"Completed runs: {completed}")
    print(f"Skipped runs: {skipped}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
