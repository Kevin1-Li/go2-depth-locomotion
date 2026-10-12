#!/usr/bin/env bash
# One-shot orchestrator for the cross-attention vs concat experiment.
#
# Sequence (all on GPU, serially, never concurrent with the Teacher training):
#   1. wait for the Teacher v4 training process to exit (poll every 5 min)
#   2. verify model_19999.pt exists, non-empty, iter == 19999
#   3. run the Teacher validity gate (must PASS before distillation)
#   4. GPU smoke for BOTH arms (concat + cross_attention, 4 envs)
#   5. dispatch the paired chain (run_ca_experiment.sh)
#
# Launch once in the background:
#   setsid bash scripts/rsl_rl/run_ca_orchestrator.sh > logs/ca_exp/orchestrator.log 2>&1 &
#
# Monitor:
#   tail -f logs/ca_exp/orchestrator.log
#   tail -f logs/ca_exp/chain.log
#   tail -f logs/ca_exp/gate.log

set -u
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="$HOME/IsaacLab/_isaac_sim/python.sh"
TEACHER_DIR="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10"
TEACHER_CKPT="$TEACHER_DIR/model_19999.pt"
OUT_DIR="logs/ca_exp"
TASK="Unitree-Go2-Depth-Student"

mkdir -p "$OUT_DIR"

echo "[orch] ===== orchestrator start $(date) ====="

# 1. wait for teacher process to exit (poll every 5 min, up to 3 h)
waited=0
while pgrep -f "train.py.*Teacher" > /dev/null 2>&1; do
    echo "[orch] teacher still training (waited ${waited}s) $(date)"
    sleep 300
    waited=$((waited + 300))
    if [ "$waited" -ge 10800 ]; then
        echo "[orch] WARN: teacher not finished after 3h; falling back to latest complete checkpoint"
        TEACHER_CKPT="$(ls -t "$TEACHER_DIR"/model_*.pt | head -1)"
        break
    fi
done
echo "[orch] teacher process gone (waited ${waited}s)"

# 2. verify the final checkpoint
python3 - "$TEACHER_CKPT" <<'PY' || { echo "[orch] FATAL: checkpoint verify failed"; exit 1; }
import os
import sys

import torch

p = sys.argv[1]
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print(f"[orch] FATAL: {p} missing or empty")
    sys.exit(1)
ck = torch.load(p, map_location="cpu", weights_only=False)
it = ck.get("iter")
print(f"[orch] checkpoint {p} iter={it}")
if it is None:
    print("[orch] FATAL: checkpoint has no 'iter' key")
    sys.exit(1)
PY

# 3. teacher validity gate (must PASS)
echo "[orch] running teacher gate $(date)"
"$PYTHON" scripts/rsl_rl/train_distill.py --headless --task "$TASK" --num_envs 16 \
    --gate --gate_steps 300 --teacher_ckpt "$TEACHER_CKPT" > "$OUT_DIR/gate.log" 2>&1
if grep -q "GATE PASSED" "$OUT_DIR/gate.log"; then
    echo "[orch] gate PASSED"
else
    echo "[orch] FATAL: gate did not PASS; aborting (no distillation). See $OUT_DIR/gate.log"
    exit 1
fi

# 4. GPU smoke for both arms
for ft in concat cross_attention; do
    echo "[orch] smoke $ft $(date)"
    "$PYTHON" scripts/rsl_rl/smoke_ca.py --headless --fusion_type "$ft" \
        --teacher_ckpt "$TEACHER_CKPT" > "$OUT_DIR/smoke_${ft}.log" 2>&1
    echo "[orch] smoke $ft done rc=$?"
done

# 5. dispatch the paired chain (blocks until all pairs attempted)
echo "[orch] dispatching chain $(date)"
exec bash scripts/rsl_rl/run_ca_experiment.sh
