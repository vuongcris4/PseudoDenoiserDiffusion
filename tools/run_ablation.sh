#!/usr/bin/env bash
# ===========================================================================
# Systematic ablation study for DifusionDenoiser (D3PM)
#
# Trains all 6 config variants (3 conditioning × 2 noise types), then
# evaluates each checkpoint and aggregates results into a single table.
#
# Usage:
#   # Full study — single GPU
#   bash tools/run_ablation.sh
#
#   # Multi-GPU (4 GPUs per experiment)
#   bash tools/run_ablation.sh --gpus 4
#
#   # Evaluate only (skip training)
#   bash tools/run_ablation.sh --eval-only
#
#   # Specify GPU device(s)
#   CUDA_VISIBLE_DEVICES=0,1 bash tools/run_ablation.sh --gpus 2
#
#   # Custom data root
#   bash tools/run_ablation.sh --data-root /path/to/dataset
#
#   # Custom denoising steps for evaluation
#   bash tools/run_ablation.sh --eval-steps 50
# ===========================================================================
set -euo pipefail

# ========================== Default parameters =============================
GPUS=1
EVAL_ONLY=0
DATA_ROOT=""
WORK_DIR="work_dirs"
EVAL_STEPS=50            # Reverse-diffusion steps at evaluation time
TEMPERATURE=1.0          # Sampling temperature at evaluation time
SEED=42
RESUME=0                 # Set to 1 to resume incomplete experiments

# ========================== Parse CLI arguments ============================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS="$2";          shift 2 ;;
        --eval-only)     EVAL_ONLY=1;        shift   ;;
        --data-root)     DATA_ROOT="$2";     shift 2 ;;
        --work-dir)      WORK_DIR="$2";      shift 2 ;;
        --eval-steps)    EVAL_STEPS="$2";    shift 2 ;;
        --temperature)   TEMPERATURE="$2";   shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --resume)        RESUME=1;           shift   ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | head -n -1
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ========================== Project paths ==================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_DIR}/configs/denoiser"
TRAIN_SCRIPT="${PROJECT_DIR}/tools/train.py"
TEST_SCRIPT="${PROJECT_DIR}/tools/test.py"

# ========================== Experiment matrix ==============================
# 3 conditioning × 2 noise types = 6 base + 3 pretrained backbone = 9 experiments
CONFIGS=(
    "d3pm_concat_uniform_512x512_100k"
    "d3pm_concat_absorbing_512x512_100k"
    "d3pm_crossattn_uniform_512x512_100k"
    "d3pm_crossattn_absorbing_512x512_100k"
    "d3pm_hybrid_uniform_512x512_100k"
    "d3pm_hybrid_absorbing_512x512_100k"
    "d3pm_crossattn_uniform_segformer_512x512_100k"
    "d3pm_crossattn_absorbing_resnet50_512x512_100k"
    "d3pm_hybrid_uniform_resnet101_512x512_100k"
)

# Human-readable labels for the results table
declare -A LABELS=(
    ["d3pm_concat_uniform_512x512_100k"]="Concat    | Uniform   | UNet"
    ["d3pm_concat_absorbing_512x512_100k"]="Concat    | Absorbing | UNet"
    ["d3pm_crossattn_uniform_512x512_100k"]="CrossAttn | Uniform   | UNet"
    ["d3pm_crossattn_absorbing_512x512_100k"]="CrossAttn | Absorbing | UNet"
    ["d3pm_hybrid_uniform_512x512_100k"]="Hybrid    | Uniform   | UNet"
    ["d3pm_hybrid_absorbing_512x512_100k"]="Hybrid    | Absorbing | UNet"
    ["d3pm_crossattn_uniform_segformer_512x512_100k"]="CrossAttn | Uniform   | SegFormer-B2"
    ["d3pm_crossattn_absorbing_resnet50_512x512_100k"]="CrossAttn | Absorbing | ResNet-50"
    ["d3pm_hybrid_uniform_resnet101_512x512_100k"]="Hybrid    | Uniform   | ResNet-101"
)

# ========================== Utility functions ===============================
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log_info()  { echo "[$(timestamp)] [INFO]  $*"; }
log_warn()  { echo "[$(timestamp)] [WARN]  $*"; }
log_error() { echo "[$(timestamp)] [ERROR] $*" >&2; }

# Build the data-root override string if user specified --data-root
cfg_options_str() {
    local opts=""
    if [[ -n "$DATA_ROOT" ]]; then
        opts="--cfg-options data.train.data_root=${DATA_ROOT} "
        opts+="data.val.data_root=${DATA_ROOT} "
        opts+="data.test.data_root=${DATA_ROOT}"
    fi
    echo "$opts"
}

# ========================== Training =======================================
run_training() {
    local config_name="$1"
    local config_path="${CONFIG_DIR}/${config_name}.py"
    local exp_work_dir="${WORK_DIR}/${config_name}"
    local latest_ckpt="${exp_work_dir}/latest.pth"

    # Skip if already trained (latest checkpoint exists) and not resuming
    if [[ -f "$latest_ckpt" && "$RESUME" -eq 0 ]]; then
        log_info "Skipping ${config_name} — latest.pth already exists."
        log_info "  (Use --resume to continue incomplete training.)"
        return 0
    fi

    log_info "=========================================================="
    log_info "Training: ${config_name}"
    log_info "=========================================================="

    local resume_flag=""
    if [[ -f "$latest_ckpt" && "$RESUME" -eq 1 ]]; then
        resume_flag="--resume-from ${latest_ckpt}"
        log_info "Resuming from ${latest_ckpt}"
    fi

    local cmd=""
    if [[ "$GPUS" -gt 1 ]]; then
        cmd="torchrun --nproc_per_node=${GPUS} ${TRAIN_SCRIPT}"
        cmd+=" ${config_path}"
        cmd+=" --work-dir ${exp_work_dir}"
        cmd+=" --seed ${SEED}"
        cmd+=" --launcher pytorch"
        cmd+=" ${resume_flag}"
    else
        cmd="python ${TRAIN_SCRIPT}"
        cmd+=" ${config_path}"
        cmd+=" --work-dir ${exp_work_dir}"
        cmd+=" --seed ${SEED}"
        cmd+=" ${resume_flag}"
    fi

    local train_log="${exp_work_dir}/train.log"
    mkdir -p "${exp_work_dir}"

    log_info "Command: ${cmd}"
    log_info "Log:     ${train_log}"

    # Run training, tee to log file
    if eval "${cmd}" 2>&1 | tee "${train_log}"; then
        log_info "Training completed: ${config_name}"
    else
        log_error "Training FAILED: ${config_name}. See ${train_log}"
        return 1
    fi
}

# ========================== Evaluation =====================================
run_evaluation() {
    local config_name="$1"
    local config_path="${CONFIG_DIR}/${config_name}.py"
    local exp_work_dir="${WORK_DIR}/${config_name}"
    local latest_ckpt="${exp_work_dir}/latest.pth"
    local eval_log="${exp_work_dir}/eval_steps${EVAL_STEPS}.log"

    if [[ ! -f "$latest_ckpt" ]]; then
        log_warn "No checkpoint for ${config_name} — skipping evaluation."
        return 1
    fi

    log_info "=========================================================="
    log_info "Evaluating: ${config_name}  (steps=${EVAL_STEPS}, T=${TEMPERATURE})"
    log_info "=========================================================="

    local cmd="python ${TEST_SCRIPT}"
    cmd+=" ${config_path}"
    cmd+=" ${latest_ckpt}"
    cmd+=" --num-steps ${EVAL_STEPS}"
    cmd+=" --temperature ${TEMPERATURE}"

    log_info "Command: ${cmd}"

    if eval "${cmd}" 2>&1 | tee "${eval_log}"; then
        log_info "Evaluation completed: ${config_name}"
    else
        log_error "Evaluation FAILED: ${config_name}. See ${eval_log}"
        return 1
    fi
}

# ========================== Results aggregation ============================
aggregate_results() {
    local results_file="${WORK_DIR}/ablation_results.txt"
    local results_csv="${WORK_DIR}/ablation_results.csv"

    log_info "=========================================================="
    log_info "Aggregating results → ${results_file}"
    log_info "=========================================================="

    # Header
    local header
    header=$(printf "%-12s | %-10s | %-14s | %-12s | %-14s | %-8s" \
             "Conditioning" "Noise" "Backbone" "Pseudo mIoU" "Denoised mIoU" "Δ mIoU")
    local separator
    separator=$(printf '%0.s-' {1..90})

    {
        echo "============================================================"
        echo "  D3PM Ablation Study Results"
        echo "  Eval steps: ${EVAL_STEPS}  |  Temperature: ${TEMPERATURE}"
        echo "  Date: $(timestamp)"
        echo "============================================================"
        echo ""
        echo "$header"
        echo "$separator"
    } > "$results_file"

    # CSV header
    echo "config,conditioning,noise,backbone,pseudo_miou,denoised_miou,delta_miou" > "$results_csv"

    for config_name in "${CONFIGS[@]}"; do
        local eval_log="${WORK_DIR}/${config_name}/eval_steps${EVAL_STEPS}.log"
        local label="${LABELS[$config_name]}"

        # Parse conditioning and noise from label
        local cond noise backbone
        cond=$(echo "$label" | cut -d'|' -f1 | xargs)
        noise=$(echo "$label" | cut -d'|' -f2 | xargs)
        backbone=$(echo "$label" | cut -d'|' -f3 | xargs)

        if [[ ! -f "$eval_log" ]]; then
            printf "%-12s | %-10s | %-14s | %-12s | %-14s | %-8s\n" \
                "$cond" "$noise" "$backbone" "N/A" "N/A" "N/A" >> "$results_file"
            echo "${config_name},${cond},${noise},${backbone},,,," >> "$results_csv"
            continue
        fi

        # Extract mIoU line from evaluation log
        # Expected format: "mIoU       0.XXXX              0.YYYY              +0.ZZZZ"
        local miou_line
        miou_line=$(grep "^mIoU" "$eval_log" 2>/dev/null || echo "")

        if [[ -n "$miou_line" ]]; then
            local pseudo_miou denoised_miou delta_miou
            pseudo_miou=$(echo "$miou_line" | awk '{print $2}')
            denoised_miou=$(echo "$miou_line" | awk '{print $3}')
            delta_miou=$(echo "$miou_line" | awk '{print $4}')

            printf "%-12s | %-10s | %-14s | %-12s | %-14s | %-8s\n" \
                "$cond" "$noise" "$backbone" "$pseudo_miou" "$denoised_miou" "$delta_miou" \
                >> "$results_file"
            echo "${config_name},${cond},${noise},${backbone},${pseudo_miou},${denoised_miou},${delta_miou}" \
                >> "$results_csv"
        else
            printf "%-12s | %-10s | %-14s | %-12s | %-14s | %-8s\n" \
                "$cond" "$noise" "$backbone" "PARSE_ERR" "PARSE_ERR" "PARSE_ERR" \
                >> "$results_file"
            echo "${config_name},${cond},${noise},${backbone},,,," >> "$results_csv"
        fi
    done

    echo "$separator" >> "$results_file"
    echo "" >> "$results_file"

    # Print to stdout
    echo ""
    cat "$results_file"
    echo ""
    log_info "Plain-text results: ${results_file}"
    log_info "CSV results:        ${results_csv}"
}

# ========================== Main entry point ===============================
main() {
    log_info "D3PM Ablation Study"
    log_info "  Project:     ${PROJECT_DIR}"
    log_info "  GPUs:        ${GPUS}"
    log_info "  Work dir:    ${WORK_DIR}"
    log_info "  Eval steps:  ${EVAL_STEPS}"
    log_info "  Temperature: ${TEMPERATURE}"
    log_info "  Eval only:   ${EVAL_ONLY}"
    log_info "  Resume:      ${RESUME}"
    [[ -n "$DATA_ROOT" ]] && log_info "  Data root:   ${DATA_ROOT}"
    echo ""

    mkdir -p "${WORK_DIR}"

    # ---- Phase 1: Training ----
    if [[ "$EVAL_ONLY" -eq 0 ]]; then
        log_info "===== PHASE 1: TRAINING (${#CONFIGS[@]} experiments) ====="
        echo ""

        local train_failed=0
        for config_name in "${CONFIGS[@]}"; do
            if ! run_training "$config_name"; then
                log_error "Experiment ${config_name} failed. Continuing with next..."
                train_failed=$((train_failed + 1))
            fi
            echo ""
        done

        if [[ "$train_failed" -gt 0 ]]; then
            log_warn "${train_failed}/${#CONFIGS[@]} training runs failed."
        else
            log_info "All ${#CONFIGS[@]} training runs completed successfully."
        fi
        echo ""
    fi

    # ---- Phase 2: Evaluation ----
    log_info "===== PHASE 2: EVALUATION (steps=${EVAL_STEPS}) ====="
    echo ""

    local eval_failed=0
    for config_name in "${CONFIGS[@]}"; do
        if ! run_evaluation "$config_name"; then
            eval_failed=$((eval_failed + 1))
        fi
        echo ""
    done

    if [[ "$eval_failed" -gt 0 ]]; then
        log_warn "${eval_failed}/${#CONFIGS[@]} evaluations skipped or failed."
    fi

    # ---- Phase 3: Aggregate results ----
    aggregate_results

    log_info "Ablation study complete."
}

main
