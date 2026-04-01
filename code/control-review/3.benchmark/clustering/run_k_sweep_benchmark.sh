#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/yinghuazhang/MolF-DAEs"
SCRIPT="${ROOT}/code/control-review/1.clustering/2.k_sweep_local_metrics.py"
RESULT_ROOT="${ROOT}/code/control-review/result/3.benchmark/clustering/output_k_sweep_500_target"

LABELS="${ROOT}/dataset/190w_3D_label_dropna.csv"
DATA2="${ROOT}/dataset/pubchem_molecule3.data2"
DAE="${ROOT}/result/comparison/pubchemfp_latent_3D_ME.csv"
PCA="${ROOT}/result/comparison/pubchem_PCA_2_ME.csv"
UMAP="${ROOT}/result/comparison/pubchem_UMAP_2_ME.csv"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${path}" >&2
    exit 1
  fi
}

run_eval() {
  local name="$1"
  shift
  local outdir="${RESULT_ROOT}/${name}"
  mkdir -p "${outdir}"
  echo "============================================================"
  echo "[run] ${name}"
  echo "[out] ${outdir}"
  python "${SCRIPT}" \
    --labels "${LABELS}" \
    --data2 "${DATA2}" \
    --dae "${DAE}" \
    --pca "${PCA}" \
    --umap "${UMAP}" \
    --outdir "${outdir}" \
    "$@" 2>&1 | tee "${outdir}/run.log"
}

main() {
  mkdir -p "${RESULT_ROOT}"

  require_file "${SCRIPT}"
  require_file "${LABELS}"
  require_file "${DATA2}"
  require_file "${DAE}"
  require_file "${PCA}"
  require_file "${UMAP}"

  run_eval "mol2vec_pca3d_full" \
    --add-mol2vec \
    --mol2vec-mode pca3d \
    --mol2vec-pca-coords "${ROOT}/code/control-review/3.benchmark/outputs/mol2vec/pca3d_full/coords_3d.npy" \
    --mol2vec-pca-ids "${ROOT}/code/control-review/3.benchmark/outputs/mol2vec/pca3d_full/ids.npy"

  run_eval "mol2vec_umap3d_full" \
    --add-mol2vec \
    --mol2vec-mode umap3d \
    --mol2vec-umap-coords "${ROOT}/code/control-review/3.benchmark/outputs/mol2vec/umap3d_full/coords_3d.npy" \
    --mol2vec-umap-ids "${ROOT}/code/control-review/3.benchmark/outputs/mol2vec/umap3d_full/ids.npy"

  run_eval "molai_pca3d_full" \
    --add-molai-pca3d \
    --molai-pca3d-coords "${ROOT}/code/control-review/3.benchmark/outputs/molai_pca3d_full/coords_3d.npy" \
    --molai-pca3d-ids "${ROOT}/code/control-review/3.benchmark/outputs/molai_pca3d_full/ids.npy"

  run_eval "pacmap3d_8d7919aa25" \
    --add-pacmap-pubchem \
    --pacmap-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_8d7919aa25/coords_3d.npy" \
    --pacmap-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_8d7919aa25/ids.npy"

  run_eval "pacmap3d_020c271bf2" \
    --add-pacmap-pubchem \
    --pacmap-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_020c271bf2/coords_3d.npy" \
    --pacmap-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_020c271bf2/ids.npy"

  run_eval "pacmap3d_73e89ea872" \
    --add-pacmap-pubchem \
    --pacmap-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_73e89ea872/coords_3d.npy" \
    --pacmap-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_73e89ea872/ids.npy"

  run_eval "pacmap3d_03969f1a3d" \
    --add-pacmap-pubchem \
    --pacmap-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_03969f1a3d/coords_3d.npy" \
    --pacmap-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/pacmap3d/pubchemfp/full/params_03969f1a3d/ids.npy"

  run_eval "phate3d_7ed1639a42" \
    --add-phate-pubchem \
    --phate-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_7ed1639a42/coords_3d.npy" \
    --phate-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_7ed1639a42/ids.npy"

  run_eval "phate3d_9ad2278356" \
    --add-phate-pubchem \
    --phate-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_9ad2278356/coords_3d.npy" \
    --phate-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_9ad2278356/ids.npy"

  run_eval "phate3d_6826002097" \
    --add-phate-pubchem \
    --phate-pubchem-coords "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_6826002097/coords_3d.npy" \
    --phate-pubchem-ids "${ROOT}/code/control-review/3.benchmark/outputs/phate3d/pubchemfp/full/params_6826002097/ids.npy"

  echo "============================================================"
  echo "[done] all benchmark k-sweep runs finished"
}

main "$@"
