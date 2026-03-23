"""
temporal.py — ConvLSTM temporal module

Processes the fused spatiotemporal feature sequence (B, T, D, H, W)
and produces a single hidden state (B, D, H, W) summarising the
temporal evolution across T steps.

Architecture:
    Two stacked ConvLSTM layers with residual connection.

    Layer 1: ConvLSTM  D → D  (processes raw fused features)
    Layer 2: ConvLSTM  D → D  (refines temporal state)
    Residual: add input mean over T to final hidden state

    Output: final hidden state of Layer 2  (B, D, H, W)

ConvLSTM recap:
    Standard LSTM gates, but weight matrices are replaced by convolutions.
    This preserves the spatial structure of the hidden state — unlike a
    standard LSTM that collapses all spatial info into a 1D vector.

    For each time step t:
        i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)   input gate
        f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)   forget gate
        g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g) cell gate
        o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)   output gate
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)

    All W_* are depthwise-separable convolutions (cheap, spatially local).

Why two layers?
    Layer 1 captures short-range temporal patterns (day-to-day variability,
    bloom onset signals). Layer 2 integrates those patterns into a longer-range
    summary (multi-day trends, persistent gap structures).

Why return only the final hidden state?
    The downstream decoder + heads need a single spatial feature map, not a
    sequence. The final hidden state of the last ConvLSTM layer summarises
    the full temporal context — it's conditioned on all T steps via the
    recurrent state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# ConvLSTM cell
# ======================================================================

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell — processes one time step.

    Uses depthwise-separable convolutions for efficiency:
        pointwise (1×1) mixes channels, depthwise (k×k) handles spatial.

    Args:
        in_channels:   Input feature channels.
        hidden_dim:    Hidden state / cell state channels.
        kernel_size:   Spatial kernel size (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2

        # Input → gates: projects x_t and h_{t-1} jointly to 4 gate channels
        # Separable: depthwise spatial mixing + pointwise channel mixing
        self.conv_x = nn.Conv2d(
            in_channels, hidden_dim * 4,
            kernel_size=kernel_size, padding=pad, bias=True,
        )
        self.conv_h = nn.Conv2d(
            hidden_dim, hidden_dim * 4,
            kernel_size=kernel_size, padding=pad, bias=False,
        )

        # Layer norm on gates stabilises training (replaces batch norm)
        self.norm = nn.GroupNorm(8, hidden_dim * 4)

        self._init_weights()

    def _init_weights(self) -> None:
        # Forget gate bias = 1 → encourages remembering at init
        nn.init.constant_(self.conv_x.bias[self.hidden_dim:2 * self.hidden_dim], 1.0)

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:      (B, in_channels, H, W)   input at time t
            state:  (h, c) each (B, hidden_dim, H, W), or None for zeros

        Returns:
            h_new, c_new  each (B, hidden_dim, H, W)
        """
        B, _, H, W = x.shape
        device = x.device

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=x.dtype)
        else:
            h, c = state

        # cuDNN requires contiguous tensors — fused features from Perceiver
        # may arrive non-contiguous after cross-attention reshaping.
        x = x.contiguous()
        h = h.contiguous()

        gates = self.norm(self.conv_x(x) + self.conv_h(h))  # (B, 4D, H, W)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# ======================================================================
# ConvLSTM layer (unrolls over T)
# ======================================================================

class ConvLSTMLayer(nn.Module):
    """
    Unrolls a ConvLSTMCell over T time steps.

    Input:  (B, T, D, H, W)
    Output:
        final hidden state      (B, D, H, W)
        or full hidden sequence (B, T, D, H, W)
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_dim, kernel_size)

    def forward(self, x: Tensor, return_sequence: bool = False) -> Tensor:
        B, T, D, H, W = x.shape
        if T == 0:
            raise ValueError("ConvLSTMLayer requires a non-empty sequence (T > 0).")

        state = None
        outputs: list[Tensor] = []
        for t in range(T):
            h, c = self.cell(x[:, t], state)
            state = (h, c)
            if return_sequence:
                outputs.append(h)

        if return_sequence:
            return torch.stack(outputs, dim=1)
        return h   # final hidden state: (B, D, H, W)


# ======================================================================
# Temporal module (two stacked ConvLSTM layers)
# ======================================================================

class TemporalModule(nn.Module):
    """
    Two stacked ConvLSTM layers with residual connection.

    Replaces TemporalModuleStub in model.py.

    Input:  fused  (B, T, D, H, W)
    Output: state  (B, D, H, W)

    To swap into model.py:
        from temporal import TemporalModule
        self.temporal = TemporalModule(embed_dim=cfg.embed_dim)
    """

    def __init__(self, embed_dim: int = 256, kernel_size: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.layer1 = ConvLSTMLayer(embed_dim, embed_dim, kernel_size)
        self.layer2 = ConvLSTMLayer(embed_dim, embed_dim, kernel_size)

        # Norm after each layer
        self.norm1 = nn.GroupNorm(8, embed_dim)
        self.norm2 = nn.GroupNorm(8, embed_dim)

    def forward(self, fused: Tensor) -> Tensor:
        """
        Args:
            fused: (B, T, D, H, W)
        Returns:
            state: (B, D, H, W)   temporal summary
        """
        # Ensure contiguous memory layout before ConvLSTM unrolling
        fused = fused.contiguous()
        B, T, D, H, W = fused.shape

        # Layer 1: process the full fused sequence and keep per-step states
        h1_seq = self.layer1(fused, return_sequence=True)                     # (B, T, D, H, W)
        h1_seq = self.norm1(h1_seq.reshape(B * T, D, H, W)).reshape(B, T, D, H, W)

        # Layer 2: refine the full temporal sequence, biased by global context
        seq_mean = fused.mean(dim=1)                 # (B, D, H, W)
        h2_input = h1_seq + seq_mean.unsqueeze(1)    # (B, T, D, H, W)
        h2 = self.norm2(self.layer2(h2_input))       # (B, D, H, W)

        # Residual: add sequence mean to preserve input signal
        return h2 + seq_mean


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python temporal.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, D, H, W = 2, 10, 256, 64, 64

    fused = torch.randn(B, T, D, H, W)

    module = TemporalModule(embed_dim=D)
    module.eval()

    with torch.no_grad():
        out = module(fused)

    expected = (B, D, H, W)
    status = "OK" if tuple(out.shape) == expected else f"MISMATCH — expected {expected}"
    print(f"Output shape:   {tuple(out.shape)}  {status}")

    n_params = sum(p.numel() for p in module.parameters())
    print(f"Parameters:     {n_params:,}")

    nan_count = torch.isnan(out).sum().item()
    print(f"NaNs in output: {nan_count}")

    # Verify temporal ordering matters — output should differ if sequence is reversed
    fused_rev = fused.flip(dims=[1])
    with torch.no_grad():
        out_rev = module(fused_rev)
    max_diff = (out - out_rev).abs().max().item()
    ordering_note = "OK (outputs differ)" if max_diff > 1e-4 else "WARN (outputs identical — check cell)"
    print(f"Temporal ordering sensitivity: max_diff={max_diff:.4f}  {ordering_note}")

    if tuple(out.shape) == expected and nan_count == 0:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Smoke test failed — see above.")

    try:
        module(fused[:, :0])
    except ValueError:
        print("Empty sequence guard: OK")
    else:
        raise RuntimeError("Empty sequence guard failed.")


if __name__ == "__main__":
    run_smoke_test()
