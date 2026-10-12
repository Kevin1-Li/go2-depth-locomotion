#!/usr/bin/env python3
"""CPU-only unit tests for the LSTM + Patch Cross-Attention Student (spec D section).

Run (system python3 has torch CPU, no Isaac Lab needed):

    PYTHONPATH=source/unitree_rl_lab python3 scripts/rsl_rl/test_depth_student_lstm_ca.py

Covers the new-spec acceptance items 1-6:
  1. dimension chain (depth / patch / history / Q-K-V / z_s==z_t==128 / action)
  2. patch spatial-order correctness (synthetic left/right and top/bottom patterns)
  3. history window reset fill rule (no cross-episode leakage)
  4. no NaN/Inf; L_repr and L_action each backprop; Teacher frozen (params unchanged)
  5. Teacher forward_with_latent action == forward action (allclose)
  6. checkpoint save/load round-trip; fusion-type mismatch raises; deterministic replay
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab"))

from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy, load_student_policy  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))


def _grad_norm(params) -> float:
    g = [p.grad for p in params if p.grad is not None]
    return sum(x.detach().abs().sum().item() for x in g)


def _make_teacher() -> TeacherMLPPolicy:
    return TeacherMLPPolicy(input_dim=232, hidden_dims=[512, 256, 128], action_dim=12, activation="elu")


def test_dimension_chain() -> None:
    torch.manual_seed(0)
    p = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca")
    depth = torch.randn(2, 1, 32, 48)
    history = torch.randn(2, 21, 45)

    # depth -> patch tokens (B, 96, 16)
    patches = p.depth_patcher(depth)
    check("dim_depth", tuple(depth.shape) == (2, 1, 32, 48))
    check("dim_patch", tuple(patches.shape) == (2, 96, 16), f"got {tuple(patches.shape)}")
    check("dim_history", tuple(history.shape) == (2, 21, 45))

    # Q/K/V projections share embed dim == 128
    e = p.ca_embed_dim
    check("dim_qkv_proj", p.fusion.query_proj.out_features == e and p.fusion.token_proj.out_features == e)

    # LSTM hidden == z_dim == teacher latent 128
    check("dim_lstm", p.proprio_lstm.hidden_dim == 128 and p.proprio_lstm.input_dim == 45)
    check("dim_zs", p.z_dim == 128)

    z_s, a_s = p.forward_with_latent(depth, history)
    check("dim_zs_out", tuple(z_s.shape) == (2, 128), f"got {tuple(z_s.shape)}")
    check("dim_action", tuple(a_s.shape) == (2, 12), f"got {tuple(a_s.shape)}")

    # patch_ca variant
    p2 = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="patch_ca")
    z_s2, a_s2 = p2.forward_with_latent(depth, history[:, -1, :])
    check("dim_patchca_zs", tuple(z_s2.shape) == (2, 128))
    check("dim_patchca_action", tuple(a_s2.shape) == (2, 12))


def test_patch_spatial_order() -> None:
    torch.manual_seed(0)
    p = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="patch_ca")

    # left/right: cols 0..23 = 0.1, cols 24..47 = 0.9 -> tokens col 0..5 vs 6..11
    d = torch.zeros(1, 1, 32, 48)
    d[..., :, :24] = 0.1
    d[..., :, 24:] = 0.9
    patches = p.depth_patcher(d)  # (1, 96, 16)
    col = torch.arange(96) % 12
    means = patches.mean(dim=-1).squeeze(0)  # (96,)
    left = means[col < 6].mean().item()
    right = means[col >= 6].mean().item()
    check("patch_leftright", abs(left - 0.1) < 1e-6 and abs(right - 0.9) < 1e-6, f"left={left} right={right}")

    # top/bottom: rows 0..15 = 0.1, rows 16..31 = 0.9 -> tokens row 0..3 vs 4..7 (token 0-47 vs 48-95)
    d2 = torch.zeros(1, 1, 32, 48)
    d2[..., :16, :] = 0.1
    d2[..., 16:, :] = 0.9
    patches2 = p.depth_patcher(d2)
    row = torch.arange(96) // 12
    means2 = patches2.mean(dim=-1).squeeze(0)
    top = means2[row < 4].mean().item()
    bottom = means2[row >= 4].mean().item()
    check("patch_topbottom", abs(top - 0.1) < 1e-6 and abs(bottom - 0.9) < 1e-6, f"top={top} bottom={bottom}")
    check("patch_token_0_47_vs_48_95", abs(means2[:48].mean().item() - 0.1) < 1e-6
          and abs(means2[48:].mean().item() - 0.9) < 1e-6)


class _MiniHistoryBuffer:
    """Faithful reproduction of Isaac Lab ``CircularBuffer`` reset/append fill rule.

    On reset the affected batch slots are zero-filled and their push counter zeroed; on the
    next append those slots are first-push, so every history entry is filled with that first
    frame (repeat-first-frame). The ``buffer`` getter returns (batch, max_len, ...) ordered
    oldest -> newest.
    """

    def __init__(self, max_len: int, batch_size: int):
        self.max_len = max_len
        self.batch_size = batch_size
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long)
        self._pointer = -1
        self._buffer = None

    def reset(self, batch_ids):
        if batch_ids is None:
            batch_ids = slice(None)
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_len, *data.shape))
        self._pointer = (self._pointer + 1) % self.max_len
        self._buffer[self._pointer] = data
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        self._num_pushes += 1

    def buffer(self):
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_len - self._pointer - 1, dims=0)
        return torch.transpose(buf, 0, 1)


def test_history_reset_fill_rule() -> None:
    torch.manual_seed(0)
    T = 21
    buf = _MiniHistoryBuffer(T, batch_size=2)

    # initial reset + first append -> every slot == first frame (repeat-first-frame)
    buf.reset(None)
    buf.append(torch.tensor([[1.0], [2.0]]))
    b = buf.buffer()  # (2, 21, 1)
    check("reset_first_frame_repeat", bool((b[:, 0, 0] == b[:, -1, 0]).all()
                                            and (b[0, :, 0] == 1.0).all() and (b[1, :, 0] == 2.0).all()))

    # run 25 steps so env 0 cycles past T, then reset env 0 only -> no stale values
    for i in range(25):
        buf.append(torch.tensor([[10.0 + i], [20.0 + i]]))
    buf.reset([0])
    buf.append(torch.tensor([[99.0], [21.0 + 25]]))
    b2 = buf.buffer()
    check("reset_no_leak_env0", bool((b2[0, :, 0] == 99.0).all()), f"env0={b2[0, :, 0].tolist()}")
    # env 1 untouched: newest entry is the last append (46.0), and it must NOT contain 99.0
    check("reset_env1_intact", bool((b2[1, :, 0] != 99.0).all()) and b2[1, -1, 0].item() == 46.0)


def test_lstm_uses_full_window() -> None:
    torch.manual_seed(0)
    p = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca").eval()
    depth = torch.randn(1, 1, 32, 48)
    # identical last frame, very different first frame -> different h_t -> different z_s
    h1 = torch.randn(1, 21, 45)
    h2 = h1.clone()
    h2[:, 0, :] = h2[:, 0, :] + 10.0
    with torch.no_grad():
        z1, _ = p.forward_with_latent(depth, h1)
        z2, _ = p.forward_with_latent(depth, h2)
    check("lstm_window_sensitive", not torch.allclose(z1, z2), "LSTM ignored early frames")


def test_no_nan_and_gradients() -> None:
    torch.manual_seed(0)
    teacher = _make_teacher()
    student = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca")
    depth = torch.randn(2, 1, 32, 48, requires_grad=True)
    history = torch.randn(2, 21, 45, requires_grad=True)
    obs = torch.randn(2, 232)

    with torch.no_grad():
        z_t, a_t = teacher.forward_with_latent(obs)
        z_t = z_t.detach()
        a_t = a_t.detach()

    # L_repr backprop
    z_s, _ = student.forward_with_latent(depth, history)
    lr = F.mse_loss(z_s, z_t)
    lr.backward()
    check("loss_repr_backprop", _grad_norm(student.parameters()) > 0)
    g_lstm = _grad_norm(student.proprio_lstm.parameters())
    g_attn = _grad_norm(student.fusion.parameters())
    g_fusion = _grad_norm(student.fusion_mlp.parameters())
    check("grad_lstm", g_lstm > 0, f"g={g_lstm}")
    check("grad_attention", g_attn > 0, f"g={g_attn}")
    check("grad_fusion", g_fusion > 0, f"g={g_fusion}")

    # L_action backprop (fresh model to avoid stale grads)
    student2 = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca")
    _, a_s = student2.forward_with_latent(depth, history)
    la = F.mse_loss(a_s, a_t)
    la.backward()
    check("loss_action_backprop", _grad_norm(student2.parameters()) > 0)
    check("grad_head_action", _grad_norm(student2.head.parameters()) > 0)

    # no NaN/Inf in outputs and losses
    check("no_nan_output", not (torch.isnan(z_s).any() or torch.isinf(z_s).any()))
    check("no_nan_losses", torch.isfinite(lr).item() and torch.isfinite(la).item())


def test_teacher_frozen() -> None:
    torch.manual_seed(0)
    teacher = _make_teacher()
    student = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    depth = torch.randn(2, 1, 32, 48)
    history = torch.randn(2, 21, 45)
    obs = torch.randn(2, 232)

    before = [p.detach().clone() for p in teacher.parameters()]
    with torch.no_grad():
        z_t, a_t = teacher.forward_with_latent(obs)
    z_s, a_s = student.forward_with_latent(depth, history)
    loss = F.mse_loss(z_s, z_t.detach()) + F.mse_loss(a_s, a_t.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    unchanged = all(torch.equal(b, p.detach()) for b, p in zip(before, teacher.parameters()))
    check("teacher_params_frozen", unchanged)
    no_grad = all(p.grad is None for p in teacher.parameters())
    check("teacher_no_grad", no_grad)
    student_moved = any(p.grad is not None for p in student.parameters())
    check("student_has_grad", student_moved)


def test_teacher_forward_with_latent() -> None:
    torch.manual_seed(0)
    teacher = _make_teacher()
    obs = torch.randn(5, 232)
    z_t, a_t = teacher.forward_with_latent(obs)
    check("teacher_latent_dim", tuple(z_t.shape) == (5, 128), f"got {tuple(z_t.shape)}")
    check("teacher_latent_allclose", torch.allclose(a_t, teacher.forward(obs), atol=1e-6))


def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(0)
    depth = torch.randn(3, 1, 32, 48)
    history = torch.randn(3, 21, 45)
    with tempfile.TemporaryDirectory() as tmp:
        p = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="lstm_patch_ca").eval()
        with torch.no_grad():
            expected = p(depth, history)
        path = os.path.join(tmp, "lstm.pt")
        torch.save(
            {
                "student_state_dict": p.state_dict(),
                "fusion_type": "lstm_patch_ca",
                "model_config": p.to_config(),
                "lambda_repr": 1.0,
                "lambda_action": 1.0,
                "iter": 0,
            },
            path,
        )
        loaded = load_student_policy(path, "cpu").eval()
        with torch.no_grad():
            got = loaded(depth, history)
        check("roundtrip_lstm", torch.allclose(expected, got, atol=1e-6))
        check("roundtrip_ft", loaded.fusion_type == "lstm_patch_ca")

        # deterministic replay
        with torch.no_grad():
            r1 = loaded(depth, history)
            r2 = loaded(depth, history)
        check("deterministic_replay", torch.equal(r1, r2))

        # mismatch: request patch_ca against a lstm_patch_ca checkpoint -> error
        try:
            load_student_policy(path, "cpu", fusion_type="patch_ca")
            check("mismatch_raises", False, "expected ValueError")
        except ValueError:
            check("mismatch_raises", True)

        # patch_ca checkpoint round-trips on its own
        p2 = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type="patch_ca").eval()
        path2 = os.path.join(tmp, "patch.pt")
        torch.save({"student_state_dict": p2.state_dict(), "fusion_type": "patch_ca",
                     "model_config": p2.to_config()}, path2)
        loaded2 = load_student_policy(path2, "cpu")
        check("roundtrip_patchca", loaded2.fusion_type == "patch_ca"
              and torch.allclose(p2(depth, history[:, -1, :]), loaded2(depth, history[:, -1, :]), atol=1e-6))


def main() -> None:
    test_dimension_chain()
    test_patch_spatial_order()
    test_history_reset_fill_rule()
    test_lstm_uses_full_window()
    test_no_nan_and_gradients()
    test_teacher_frozen()
    test_teacher_forward_with_latent()
    test_checkpoint_roundtrip()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n==== {n_pass}/{len(RESULTS)} passed, {n_fail} failed ====")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
