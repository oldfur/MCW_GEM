#!/usr/bin/env bash
set -euo pipefail

# MP-20 stage-decoupled component ablation launcher.
# Required: CHECKPOINT points to the LF_wrap generator checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON=${PYTHON:-python}
read -r -a PYTHON_CMD <<< "${PYTHON}"
MAIN_SCRIPT=${MAIN_SCRIPT:-"${REPO_ROOT}/main_LF_sample.py"}
SAMPLE_MULTI_GPU_SCRIPT=${SAMPLE_MULTI_GPU_SCRIPT:-"${REPO_ROOT}/scripts/sample_multi_gpu.py"}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-"${REPO_ROOT}/scripts/summarize_component_ablation_table.py"}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the LF_wrap checkpoint path}
LATTICE_CHECKPOINT=${LATTICE_CHECKPOINT:-outputs/train_LatticeGen_mp20/diffusion_L/generative_model_ema.npy}
CONFIG=${CONFIG:-}
GPUS=${GPUS:-}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-2026}
OUT_ROOT=${OUT_ROOT:-outputs/ablation_component_diagnostics}
DEVICE=${DEVICE:-cuda}
DP=${DP:-True}
NUM_WORKERS=${NUM_WORKERS:-0}
DATADIR=${DATADIR:-mp20}
DATASET_FOLDER_PATH=${DATASET_FOLDER_PATH:-mp20/raw}
LATTICE_MODEL=${LATTICE_MODEL:-diffusion_L}
CONDITION_LATTICE_ON_N=${CONDITION_LATTICE_ON_N:-False}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-1000}
N_CORRECTOR_STEPS=${N_CORRECTOR_STEPS:-1}
LAMBDA_SYM=${LAMBDA_SYM:-0.0}
COMPUTE_NOVELTY=${COMPUTE_NOVELTY:-0}
COMPUTE_NOVELTY_EPOCH=${COMPUTE_NOVELTY_EPOCH:-0}
VISUALIZE=${VISUALIZE:-True}
WANDB_USER=${WANDB_USER:-maochenwei-ustc}
DEBUG_ATOM_TYPES=${DEBUG_ATOM_TYPES:-True}
DIAGNOSE_GEOMETRY_BEFORE_CORRECTION=${DIAGNOSE_GEOMETRY_BEFORE_CORRECTION:-True}
SAVE_PRE_CORRECTION_GEOMETRY_NPZ=${SAVE_PRE_CORRECTION_GEOMETRY_NPZ:-True}
SUMMARIZE_AFTER=${SUMMARIZE_AFTER:-True}

if (( NUM_SAMPLES % BATCH_SIZE != 0 )); then
  echo "NUM_SAMPLES (${NUM_SAMPLES}) must be divisible by BATCH_SIZE (${BATCH_SIZE})." >&2
  exit 2
fi
NUM_ROUNDS=$((NUM_SAMPLES / BATCH_SIZE))

COMMON_ARGS=(
  --device "${DEVICE}"
  --dp "${DP}"
  --num_workers "${NUM_WORKERS}"
  --wandb_usr "${WANDB_USER}"
  --no_wandb
  --model DGAP
  --atom_type_pred 1
  --lambda_l 1.0
  --lambda_a 1.0
  --lambda_type 0.1
  --n_corrector_steps "${N_CORRECTOR_STEPS}"
  --lambda_sym "${LAMBDA_SYM}"
  --include_charges False
  --compute_novelty "${COMPUTE_NOVELTY}"
  --compute_novelty_epoch "${COMPUTE_NOVELTY_EPOCH}"
  --visualize "${VISUALIZE}"
  --sample_batch_size "${BATCH_SIZE}"
  --diffusion_steps "${DIFFUSION_STEPS}"
  --probabilistic_model diffusion_LF_wrap
  --sde_type ve
  --datadir "${DATADIR}"
  --dataset_folder_path "${DATASET_FOLDER_PATH}"
  --LatticeGenModel "${LATTICE_MODEL}"
  --pretrained_Lattice_model "${LATTICE_CHECKPOINT}"
  --condition-lattice-on-n "${CONDITION_LATTICE_ON_N}"
  --pretrained_model "${CHECKPOINT}"
  --diagnose-geometry-before-correction "${DIAGNOSE_GEOMETRY_BEFORE_CORRECTION}"
  --save-pre-correction-geometry-npz "${SAVE_PRE_CORRECTION_GEOMETRY_NPZ}"
  --debug-atom-types "${DEBUG_ATOM_TYPES}"
)

if [[ -n "${CONFIG}" ]]; then
  COMMON_ARGS+=(--sampling-config-path "${CONFIG}")
fi

run_one() {
  local config_name=$1
  local geometry_correction=$2
  local atom_decode_mode=$3
  local out_dir="${OUT_ROOT}/${config_name}"

  mkdir -p "${out_dir}"
  echo "[ComponentAblation] ${config_name}: geometry_correction=${geometry_correction}, atom_decode_mode=${atom_decode_mode}"
  if [[ -n "${GPUS}" ]]; then
    "${PYTHON_CMD[@]}" -u "${SAMPLE_MULTI_GPU_SCRIPT}" \
      --gpus "${GPUS}" \
      --num-rounds "${NUM_ROUNDS}" \
      --sample-seed "${SEED}" \
      --save-dir "${out_dir}" \
      --main-script "${MAIN_SCRIPT}" \
      -- \
      "${COMMON_ARGS[@]}" \
      --exp_name "${config_name}" \
      --component-config-name "${config_name}" \
      --geometry-correction "${geometry_correction}" \
      --atom-decode-mode "${atom_decode_mode}" \
      > "${out_dir}/sampling.log" 2>&1
  else
    "${PYTHON_CMD[@]}" -u "${MAIN_SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --num_rounds "${NUM_ROUNDS}" \
      --sample_seed "${SEED}" \
      --exp_name "${config_name}" \
      --save_dir "${out_dir}" \
      --debug-atom-dir "${out_dir}/atom_type_debug" \
      --component-config-name "${config_name}" \
      --geometry-correction "${geometry_correction}" \
      --atom-decode-mode "${atom_decode_mode}" \
      > "${out_dir}/sampling.log" 2>&1
  fi
}

run_one raw_geometry_raw_logits False raw_argmax
run_one raw_geometry_constrained_decode False constrained_search
run_one corrected_geometry_raw_logits True raw_argmax
run_one full_pipeline True constrained_search

if [[ "${SUMMARIZE_AFTER}" == "True" || "${SUMMARIZE_AFTER}" == "true" || "${SUMMARIZE_AFTER}" == "1" ]]; then
  "${PYTHON_CMD[@]}" "${SUMMARY_SCRIPT}" --root "${OUT_ROOT}" > "${OUT_ROOT}/component_ablation_table.log" 2>&1 || {
    echo "[ComponentAblation] warning: table summarization failed; see ${OUT_ROOT}/component_ablation_table.log" >&2
  }
fi

echo "[ComponentAblation] all runs completed under ${OUT_ROOT}"
