#!/usr/bin/env python3
"""CPU-only unit tests for the ``DepthStudentPolicy`` fusion refactor.

Run (system python3 has torch 2.13 CPU, no Isaac Lab needed):

    PYTHONPATH=source/unitree_rl_lab python3 scripts/rsl_rl/test_depth_student_ca.py

Covers: output shapes for both fusions, gradient flow to the depth CNN / proprio MLP /
attention (three required sites), NaN/Inf checks, save/load round-trip with metadata,
fusion-type mismatch error, legacy v1 checkpoint compatibility, and attention-weight shape.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab"))

from unitree_rl_lab.models import DepthStudentPolicy, ckpt_model_config, load_student_policy  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))


def _grad_norm(params) -> float:
    g = [p.grad for p in params if p.grad is not None]
    return sum(x.detach().abs().sum().item() for x in g)


def test_shapes() -> None:
    torch.manual_seed(0)
    depth = torch.randn(2, 1, 64, 64)
    proprio = torch.randn(2, 45)
    for ft in ("concat", "cross_attention"):
        p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type=ft)
        out = p(depth, proprio)
        check(f"shape_{ft}", tuple(out.shape) == (2, 12), f"got {tuple(out.shape)}")
        check(f"api_{ft}", tuple(p.mean_action(depth, proprio).shape) == (2, 12))
        check(f"std_{ft}", tuple(p.std.shape) == (12,) and tuple(p.log_std.shape) == (12,))
        p.update_distribution(depth, proprio)
        check(f"act_{ft}", tuple(p.act(depth, proprio).shape) == (2, 12))


def test_gradients() -> None:
    torch.manual_seed(0)
    depth = torch.randn(2, 1, 64, 64, requires_grad=True)
    proprio = torch.randn(2, 45, requires_grad=True)

    # concat: depth CNN, proprio MLP, head
    p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="concat")
    out = p(depth, proprio)
    out.sum().backward()
    check("grad_concat_depth_cnn", _grad_norm(p.depth_encoder.conv.parameters()) > 0)
    check("grad_concat_proprio_mlp", _grad_norm(p.proprio_encoder.mlp.parameters()) > 0)
    check("grad_concat_head", _grad_norm(p.head.parameters()) > 0)

    # cross_attention: depth CNN, proprio MLP, attention, head
    p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="cross_attention")
    out = p(depth, proprio)
    out.sum().backward()
    check("grad_xattn_depth_cnn", _grad_norm(p.depth_encoder.conv.parameters()) > 0)
    check("grad_xattn_proprio_mlp", _grad_norm(p.proprio_encoder.mlp.parameters()) > 0)
    check("grad_xattn_attention", _grad_norm(p.fusion.cross_attn.parameters()) > 0)
    check("grad_xattn_query_proj", _grad_norm(p.fusion.query_proj.parameters()) > 0)
    check("grad_xattn_head", _grad_norm(p.head.parameters()) > 0)


def test_nan() -> None:
    torch.manual_seed(0)
    depth = torch.randn(2, 1, 64, 64)
    proprio = torch.randn(2, 45)
    for ft in ("concat", "cross_attention"):
        p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type=ft)
        out = p(depth, proprio)
        check(f"nan_{ft}", not torch.isnan(out).any() and not torch.isinf(out).any())
        out.sum().backward()
        g = _grad_norm(p.parameters())
        check(f"nan_grad_{ft}", g == g, f"grad={g}")  # finite (== itself fails on NaN/inf)


def test_attention_shape() -> None:
    torch.manual_seed(0)
    depth = torch.randn(2, 1, 64, 64)
    proprio = torch.randn(2, 45)
    p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="cross_attention")
    out, attn = p(depth, proprio, return_attention=True)
    check("attn_shape", tuple(attn.shape) == (2, 4, 4, 64), f"got {tuple(attn.shape)}")
    # default off returns a bare tensor
    check("attn_default_off", isinstance(p(depth, proprio), torch.Tensor))


def test_save_load_roundtrip() -> None:
    torch.manual_seed(0)
    depth = torch.randn(3, 1, 64, 64)
    proprio = torch.randn(3, 45)
    with tempfile.TemporaryDirectory() as tmp:
        for ft in ("concat", "cross_attention"):
            p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type=ft)
            p.eval()
            expected = p(depth, proprio)
            path = os.path.join(tmp, f"ckpt_{ft}.pt")
            torch.save(
                {
                    "student_state_dict": p.state_dict(),
                    "fusion_type": ft,
                    "model_config": p.to_config(),
                    "iter": 0,
                },
                path,
            )
            loaded = load_student_policy(path, "cpu")
            loaded.eval()
            got = loaded(depth, proprio)
            check(f"roundtrip_{ft}", torch.allclose(expected, got, atol=1e-6))
            check(f"roundtrip_ft_{ft}", loaded.fusion_type == ft)


def test_mismatch_error() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="cross_attention")
        path = os.path.join(tmp, "xattn.pt")
        torch.save({"student_state_dict": p.state_dict(), "fusion_type": "cross_attention",
                     "model_config": p.to_config()}, path)
        try:
            load_student_policy(path, "cpu", fusion_type="concat")
            check("mismatch_raises", False, "expected ValueError")
        except ValueError:
            check("mismatch_raises", True)


def test_legacy_v1_compat() -> None:
    torch.manual_seed(0)
    depth = torch.randn(3, 1, 64, 64)
    proprio = torch.randn(3, 45)
    with tempfile.TemporaryDirectory() as tmp:
        p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="concat")
        p.eval()
        expected = p(depth, proprio)
        # v1 checkpoint layout: student_state_dict + optimizer_state_dict + iter, NO metadata
        path = os.path.join(tmp, "v1.pt")
        torch.save({"student_state_dict": p.state_dict(), "optimizer_state_dict": {}, "iter": 2999}, path)
        loaded = load_student_policy(path, "cpu")
        check("legacy_ft_concat", loaded.fusion_type == "concat")
        loaded.eval()
        check("legacy_roundtrip", torch.allclose(expected, loaded(depth, proprio), atol=1e-6))


def test_config_roundtrip() -> None:
    p = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type="cross_attention",
                           attn_embed_dim=128, attn_num_queries=4, attn_num_heads=4)
    cfg = p.to_config()
    q = DepthStudentPolicy(**cfg)
    check("config_rebuild", q.fusion_type == "cross_attention" and q.attn_num_queries == 4)
    check("config_meta", ckpt_model_config({"model_config": cfg})["fusion_type"] == "cross_attention")


def main() -> None:
    test_shapes()
    test_gradients()
    test_nan()
    test_attention_shape()
    test_save_load_roundtrip()
    test_mismatch_error()
    test_legacy_v1_compat()
    test_config_roundtrip()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n==== {n_pass}/{len(RESULTS)} passed, {n_fail} failed ====")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
