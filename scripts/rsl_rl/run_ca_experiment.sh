#!/usr/bin/env bash
# Cross-Attention vs Concat baseline — paired fair-comparison chain.
#
# Runs (in order, one pair per seed): concat_s42 -> xattn_s42 -> concat_s43 -> xattn_s43
# -> concat_s44 -> xattn_s44. Each arm trains a fresh Student (3000 iters x 64 envs x 120
# steps, lr 1e-3, 2 epochs — the v1 budget) against the SAME frozen Teacher v4
# (model_19999.pt), then is immediately evaluated with the fixed protocol (deterministic
# mean action, PLAY_TERRAIN_CFG, seed 42, 64 envs x 3 rounds).
#
# Robustness / resumability:
#   * each train/eval runs in its own setsid session; logs go to logs/ca_exp/{run_name}.log
#   * before each run the GPU must be free (<1 GiB used) and disk must have headroom
#   * a failed run is recorded and skipped; the chain never aborts on a single failure
#   * results are written incrementally (logs/ca_exp/results_{arm}_{seed}.json) and the
#     summary markdown is rewritten after every run, so you can truncate at any pair.
#
# Launch (background, own session):
#   setsid bash scripts/rsl_rl/run_ca_experiment.sh > logs/ca_exp/chain.log 2>&1 &
#
# Monitor:
#   tail -f logs/ca_exp/chain.log
#   tail -f logs/ca_exp/concat_s42.log

set -u
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="$HOME/IsaacLab/_isaac_sim/python.sh"
TASK="Unitree-Go2-Depth-Student"
EXP="unitree_go2_student_ca"
OUT_DIR="logs/ca_exp"
TEACHER_CKPT="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt"

NUM_ENVS=64
MAX_ITER=3000
STEPS=120
EPOCHS=2
LR="1e-3"
EVAL_NUM_ENVS=64
EVAL_ROUNDS=3
EVAL_SEED=42
FINAL_ITER=$((MAX_ITER - 1))

# order: "fusion_type:arm:seed" (2 seed pairs max, per user decision 2026-10-11)
RUNS=(
    "concat:concat:42"
    "cross_attention:xattn:42"
    "concat:concat:43"
    "cross_attention:xattn:43"
)

mkdir -p "$OUT_DIR"

gpu_free() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    [ -z "$used" ] && return 0
    [ "$used" -lt 1024 ]
}

wait_gpu_free() {
    local timeout_s="${1:-10800}"
    local waited=0
    while ! gpu_free; do
        sleep 60
        waited=$((waited + 60))
        if [ "$waited" -ge "$timeout_s" ]; then
            echo "[chain] WARN: GPU not free after ${timeout_s}s; proceeding anyway" >> "$OUT_DIR/chain.log"
            return 0
        fi
    done
}

check_disk() {
    local avail
    avail="$(df -P . 2>/dev/null | awk 'NR==2{print $4}')"
    if [ -n "$avail" ] && [ "$avail" -lt 5242880 ]; then
        echo "[chain] ERROR: low disk (${avail} KiB free < 5 GiB)" >> "$OUT_DIR/chain.log"
        return 1
    fi
    return 0
}

write_summary() {
    python3 scripts/rsl_rl/merge_ca_results.py --out_dir "$OUT_DIR" > /dev/null 2>&1 || true
}

echo "[chain] ===== chain start $(date) =====" >> "$OUT_DIR/chain.log"
echo "[chain] python=$PYTHON teacher=$TEACHER_CKPT num_envs=$NUM_ENVS max_iter=$MAX_ITER" >> "$OUT_DIR/chain.log"

for entry in "${RUNS[@]}"; do
    fusion_type="${entry%%:*}"
    rest="${entry#*:}"
    arm="${rest%%:*}"
    seed="${rest##*:}"
    run_name="${arm}_s${seed}"
    log="$OUT_DIR/${run_name}.log"

    echo "[chain] ===== run ${run_name} (fusion_type=${fusion_type}) $(date) =====" | tee -a "$OUT_DIR/chain.log"

    wait_gpu_free
    if ! check_disk; then
        echo "[chain] SKIP ${run_name}: disk check failed" | tee -a "$OUT_DIR/chain.log"
        continue
    fi

    echo "[chain] ${run_name}: training ..." | tee -a "$OUT_DIR/chain.log"
    setsid bash -c "cd \"$ROOT\" && \"$PYTHON\" scripts/rsl_rl/train_distill.py \
        --headless --task \"$TASK\" --num_envs $NUM_ENVS --seed $seed \
        --max_iterations $MAX_ITER --num_steps_per_env $STEPS \
        --num_learning_epochs $EPOCHS --learning_rate $LR \
        --experiment_name \"$EXP\" --run_name \"$run_name\" \
        --fusion_type \"$fusion_type\" --teacher_ckpt \"$TEACHER_CKPT\" \
        --mem_track" > "$log" 2>&1
    train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        echo "[chain] FAIL ${run_name}: train rc=${train_rc}" | tee -a "$OUT_DIR/chain.log"
        continue
    fi

    ckpt_dir="$(ls -dt "logs/rsl_rl/$EXP/"*"_${run_name}" 2>/dev/null | head -1)"
    ckpt="${ckpt_dir}/student_model_${FINAL_ITER}.pt"
    if [ -z "$ckpt_dir" ] || [ ! -s "$ckpt" ]; then
        echo "[chain] FAIL ${run_name}: missing checkpoint ${ckpt}" | tee -a "$OUT_DIR/chain.log"
        continue
    fi

    echo "[chain] ${run_name}: evaluating ${ckpt}" | tee -a "$OUT_DIR/chain.log"
    setsid bash -c "cd \"$ROOT\" && \"$PYTHON\" scripts/rsl_rl/eval_three_way.py \
        --headless --model student --checkpoint \"$ckpt\" \
        --num_envs $EVAL_NUM_ENVS --rounds $EVAL_ROUNDS --seed $EVAL_SEED \
        --teacher_ckpt \"$TEACHER_CKPT\" \
        --out_json \"$OUT_DIR/results_${arm}_${seed}.json\"" >> "$log" 2>&1
    eval_rc=$?
    if [ "$eval_rc" -ne 0 ]; then
        echo "[chain] FAIL ${run_name}: eval rc=${eval_rc}" | tee -a "$OUT_DIR/chain.log"
        continue
    fi

    echo "[chain] DONE ${run_name}" | tee -a "$OUT_DIR/chain.log"
    write_summary
done

write_summary
echo "[chain] ===== chain finished all attempted runs $(date) =====" | tee -a "$OUT_DIR/chain.log"
