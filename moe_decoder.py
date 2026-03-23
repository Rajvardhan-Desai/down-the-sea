"""
moe_decoder.py — Soft-routing Mixture-of-Experts decoder

Refines the temporal hidden state (B, D, H, W) into a decoded feature
map (B, D, H, W) using a blend of N expert decoders, where the blend
weights are computed per-pixel from the input features.

Architecture:
    Input: state  (B, D, H, W)

    Router:
        Global average pool → (B, D)
        Linear → (B, n_experts)
        Softmax → routing weights  (B, n_experts)
        # Global routing: one weight set per sample (not per-pixel).
        # Keeps routing stable and interpretable.

    Experts (n_experts=4):
        Each expert is a small residual ConvNet:
            Conv 3×3 → GroupNorm → GELU → Conv 3×3 → GroupNorm
            + residual connection
        Experts share the same architecture but have independent weights.

    Output:
        Weighted sum of expert outputs: Σ w_e * expert_e(state)
        (B, D, H, W)

Why global routing (not per-pixel)?
    Per-pixel routing produces a different expert blend at every spatial
    location. In practice this makes the router unstable — small input
    perturbations cause large routing changes, and experts collapse to
    identical functions (the "expert collapse" problem).
    Global routing (one set of weights per sample) is more stable,
    still regime-adaptive (different samples can route differently),
    and interpretable: you can inspect which expert activates for which
    regime after training.

Why 4 experts?
    Matching the 4 dominant Bay of Bengal oceanographic regimes:
    pre-monsoon (Mar–May), summer monsoon (Jun–Sep),
    post-monsoon (Oct–Nov), winter (Dec–Feb).
    This is a soft prior — the model isn't forced to use one expert per
    regime, but the capacity is there if the routing learns it.

Expert collapse prevention:
    - Load-balancing auxiliary loss (call compute_aux_loss() during training)
      penalises routing weight entropy collapse (all samples → one expert).
    - Router input uses global average pooled features, which are more
      stable than local features.
    - Experts use residual connections so they start near the identity
      and diverge gradually during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Expert block
# ======================================================================

class ExpertBlock(nn.Module):
    """
    Single expert: a residual ConvNet with two Conv3×3 layers.

    Input/output: (B, D, H, W)
    """

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


# ======================================================================
# Router
# ======================================================================

class Router(nn.Module):
    """
    Computes per-sample expert routing weights from global features.

    Input:  state  (B, D, H, W)
    Output: weights  (B, n_experts)   softmax probabilities
    """

    def __init__(self, dim: int, n_experts: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)         # (B, D, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, n_experts),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.pool(x).flatten(1)              # (B, D)
        logits = self.fc(feat)                      # (B, n_experts)
        return logits.softmax(dim=-1)               # (B, n_experts)


# ======================================================================
# Load-balancing auxiliary loss
# ======================================================================

def compute_aux_loss(routing_weights: Tensor) -> Tensor:
    """
    Load-balancing loss to prevent expert collapse.

    Penalises routing distributions where one expert captures all samples.
    Uses the auxiliary loss from Switch Transformer (Fedus et al., 2021):

        L_aux = n_experts * Σ_e (f_e * p_e)

    where:
        f_e = fraction of samples routed to expert e (top-1 hard assignment)
        p_e = mean routing weight for expert e across the batch

    A uniform distribution gives L_aux = 1.0.
    Collapse to one expert gives L_aux = n_experts.

    Args:
        routing_weights: (B, n_experts)  softmax routing probabilities

    Returns:
        Scalar auxiliary loss. Add to total loss with a small weight
        (e.g. 0.01) during training:
            loss = task_loss + 0.01 * compute_aux_loss(routing_weights)
    """
    n_experts = routing_weights.shape[1]

    # Hard assignment: which expert has the highest weight per sample
    top1 = routing_weights.argmax(dim=-1)                    # (B,)
    f = torch.bincount(top1, minlength=n_experts).to(routing_weights.dtype)
    f = f / top1.numel()                                     # fraction of batch → expert e

    # Mean routing weight per expert
    p = routing_weights.mean(dim=0)                         # (n_experts,)

    return n_experts * (f * p).sum()


# ======================================================================
# MoE decoder
# ======================================================================

class MoEDecoder(nn.Module):
    """
    Soft-routing Mixture-of-Experts decoder.

    Replaces MoEDecoderStub in model.py.

    Args:
        embed_dim:  Feature dimension (default 256).
        n_experts:  Number of expert decoders (default 4).
        expansion:  Channel expansion ratio inside each expert (default 2).

    To swap into model.py:
        from moe_decoder import MoEDecoder
        self.decoder = MoEDecoder(
            embed_dim=cfg.embed_dim,
            n_experts=cfg.n_experts,
        )

    During training, collect routing weights for the aux loss:
        decoded, routing_w = model.decoder(state, return_routing=True)
        loss = task_loss + 0.01 * compute_aux_loss(routing_w)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_experts: int = 4,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_experts = n_experts

        self.router  = Router(embed_dim, n_experts)
        self.experts = nn.ModuleList([
            ExpertBlock(embed_dim, expansion) for _ in range(n_experts)
        ])

        # Final norm after blending
        self.out_norm = nn.GroupNorm(8, embed_dim)

    def forward(
        self,
        state: Tensor,
        return_routing: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            state:          (B, D, H, W)
            return_routing: If True, also return routing weights (B, n_experts).
                            Set True during training to compute aux loss.

        Returns:
            decoded:         (B, D, H, W)
            routing_weights: (B, n_experts)  only if return_routing=True
        """
        routing_weights = self.router(state)                # (B, n_experts)

        # Compute all expert outputs and blend
        out = torch.zeros_like(state)
        for e, expert in enumerate(self.experts):
            w = routing_weights[:, e].view(-1, 1, 1, 1)    # (B, 1, 1, 1)
            out = out + w * expert(state)

        out = self.out_norm(out)

        if return_routing:
            return out, routing_weights
        return out


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python moe_decoder.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, D, H, W = 4, 256, 64, 64   # B=4 for meaningful load-balance stats
    n_experts   = 4

    state = torch.randn(B, D, H, W)

    decoder = MoEDecoder(embed_dim=D, n_experts=n_experts)
    decoder.eval()

    with torch.no_grad():
        out, routing_w = decoder(state, return_routing=True)

    expected = (B, D, H, W)
    status = "OK" if tuple(out.shape) == expected else f"MISMATCH — expected {expected}"
    print(f"Output shape:     {tuple(out.shape)}  {status}")

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Parameters:       {n_params:,}")

    nan_count = torch.isnan(out).sum().item()
    print(f"NaNs in output:   {nan_count}")

    print(f"\nRouting weights (B=4 samples × {n_experts} experts):")
    for b in range(B):
        w_str = "  ".join(f"{routing_w[b, e].item():.3f}" for e in range(n_experts))
        print(f"  sample {b}: [{w_str}]")

    aux = compute_aux_loss(routing_w)
    print(f"\nAux loss at init: {aux.item():.4f}  (1.0 = uniform, {n_experts}.0 = collapsed)")

    if tuple(out.shape) == expected and nan_count == 0:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Smoke test failed — see above.")


if __name__ == "__main__":
    run_smoke_test()
