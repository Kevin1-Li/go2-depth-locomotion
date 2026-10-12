#!/usr/bin/env bash
# LSTM + Patch Cross-Attention Student — train -> eval chain (target architecture 3).
#
# Single arm: trains the ``lstm_patch_ca`` Student (seed 42, 3000 iters x 64 envs x 120
# steps, lr 1e-3, 2 epochs, dual loss lambda 1.0/1.0) against the frozen Teacher v4
# (model_19999.pt), then immediately evaluates with the fixed protocol (deterministic mean
# action, PLAY_TERRAIN_CFG, seed 42, 64 envs x 3 rounds).
#
# Launch (background, own session):
#   setsid nohup bash scripts/rsl_rl/run_lstm_ca_experiment.sh > logs/lstm_ca_exp/chain.out 2>&1 & disown
#
# Monitor:
#   tail -f logs/lstm_ca_exp/train.log
#   tail -f logs/lstm_ca_exp/chain.log

set -u
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="$HOME/IsaacLab/_isaac_sim/python.sh"
TASK="Unitree-Go2-Depth-Student"
EXP="unitree_go2_student_lstm_ca"
OUT_DIR="logs/lstm_ca_exp"
TEACHER_CKPT="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt"

FUSION="lstm_patch_ca"
NUM_ENVS=64
MAX_ITER=3000
STEPS=120
EPOCHS=2
LR="1e-3"
SEED=42
LAMBDA_REPR=1.0
LAMBDA_ACTION=1.0
EVAL_NUM_ENVS=64
EVAL_ROUNDS=3
EVAL_SEED=42
FINAL_ITER=$((MAX_ITER - 1))
RUN_NAME="lstm_patch_ca_s${SEED}"

mkdir -p "$OUT_DIR"

echo "[chain] ===== lstm_patch_ca chain start $(date) =====" | tee -a "$OUT_DIR/chain.log"
echo "[chain] python=$PYTHON teacher=$TEACHER_CKPT fusion=$FUSION num_envs=$NUM_ENVS max_iter=$MAX_ITER" | tee -a "$OUT_DIR/chain.log"

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
echo "[chain] gpu_used_before=${used}MiB" | tee -a "$OUT_DIR/chain.log"
if [ -n "$used" ] && [ "$used" -ge 1500 ]; then
    echo "[chain] FATAL: GPU not free (${used} MiB used); aborting" | tee -a "$OUT_DIR/chain.log"
    exit 1
fi

echo "[chain] training $(date)" | tee -a "$OUT_DIR/chain.log"
setsid bash -c "cd \"$ROOT\" && \"$PYTHON\" scripts/rsl_rl/train_distill.py \
    --headless --task \"$TASK\" --num_envs $NUM_ENVS --seed $SEED \
    --max_iterations $MAX_ITER --num_steps_per_env $STEPS \
    --num_learning_epochs $EPOCHS --learning_rate $LR \
    --experiment_name \"$EXP\" --run_name \"$RUN_NAME\" \
    --fusion_type \"$FUSION\" --teacher_ckpt \"$TEACHER_CKPT\" \
    --lambda_repr $LAMBDA_REPR --lambda_action $LAMBDA_ACTION \
    --mem_track" > "$OUT_DIR/train.log" 2>&1
train_rc=$?
if [ "$train_rc" -ne 0 ]; then
    echo "[chain] FAIL train rc=${train_rc}" | tee -a "$OUT_DIR/chain.log"
    exit 1
fi

ckpt_dir="$(ls -dt "logs/rsl_rl/$EXP/"*"_${RUN_NAME}" 2>/dev/null | head -1)"
ckpt="${ckpt_dir}/student_model_${FINAL_ITER}.pt"
if [ -z "$ckpt_dir" ] || [ ! -s "$ckpt" ]; then
    echo "[chain] FAIL missing checkpoint ${ckpt}" | tee -a "$OUT_DIR/chain.log"
    exit 1
fi

echo "[chain] evaluating ${ckpt} $(date)" | tee -a "$OUT_DIR/chain.log"
setsid bash -c "cd \"$ROOT\" && \"$PYTHON\" scripts/rsl_rl/eval_three_way.py \
    --headless --model student --checkpoint \"$ckpt\" \
    --num_envs $EVAL_NUM_ENVS --rounds $EVAL_ROUNDS --seed $EVAL_SEED \
    --teacher_ckpt \"$TEACHER_CKPT\" \
    --out_json \"$OUT_DIR/results_${RUN_NAME}.json\"" > "$OUT_DIR/eval.log" 2>&1
eval_rc=$?
if [ "$eval_rc" -ne 0 ]; then
    echo "[chain] FAIL eval rc=${eval_rc}" | tee -a "$OUT_DIR/chain.log"
    exit 1
fi

echo "[chain] DONE $(date)" | tee -a "$OUT_DIR/chain.log"
