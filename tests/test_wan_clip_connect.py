"""Tests for src/diffusion/wan_clip_connect.py.

All tests run on CPU with tiny tensors — no model weights required.
Covers: latent layout math, clip validation, hard-constraint overwrite logic,
and normalisation round-trip.
"""
from __future__ import annotations

import pytest
import torch

from diffusion.wan_clip_connect import (
    WanLatentLayout,
    derive_latent_layout,
    denormalize_clip_to_zero_to_one,
    normalize_clip_to_neg_one_to_one,
    overwrite_anchor_regions,
    validate_clip_tensor,
)


# ── derive_latent_layout ──────────────────────────────────────────────────────

class TestDeriveLatentLayout:
    def test_canonical_72_frame_spec(self):
        """The main use case: 72 total frames, 24 anchor frames, 480×848."""
        layout = derive_latent_layout(
            num_frames=72, height=480, width=848, anchor_frames=24
        )
        assert layout.latent_frames == 18   # (72-1)//4 + 1
        assert layout.latent_h      == 60   # 480 // 8
        assert layout.latent_w      == 106  # 848 // 8
        assert layout.anchor_len    == 6    # (24-1)//4 + 1
        assert layout.mid_start     == 6
        assert layout.mid_end       == 12

    def test_middle_section_correct(self):
        layout = derive_latent_layout(
            num_frames=72, height=480, width=848, anchor_frames=24
        )
        middle_len = layout.mid_end - layout.mid_start
        assert middle_len == 6
        assert layout.anchor_len + middle_len + layout.anchor_len == layout.latent_frames

    def test_custom_dimensions(self):
        layout = derive_latent_layout(
            num_frames=81, height=720, width=1280, anchor_frames=25
        )
        assert layout.latent_frames == 21   # (81-1)//4 + 1
        assert layout.latent_h      == 90   # 720 // 8
        assert layout.latent_w      == 160  # 1280 // 8
        assert layout.anchor_len    == 7    # (25-1)//4 + 1
        assert layout.mid_start     == 7
        assert layout.mid_end       == 14

    def test_raises_when_no_middle_frames(self):
        """If both anchor clips fill the canvas entirely, raise ValueError."""
        with pytest.raises(ValueError, match="No middle frames"):
            # 9 frames → 3 latent frames. anchor 9 → anchor_len 3.
            # Each anchor takes 3, no room for middle.
            derive_latent_layout(
                num_frames=9, height=480, width=848, anchor_frames=9
            )

    def test_temporal_formula_edge_case(self):
        """Single latent frame per anchor (1 pixel frame)."""
        layout = derive_latent_layout(
            num_frames=13, height=64, width=64, anchor_frames=1
        )
        assert layout.anchor_len == 1       # (1-1)//4 + 1 = 1
        assert layout.latent_frames == 4    # (13-1)//4 + 1 = 4
        assert layout.mid_start == 1
        assert layout.mid_end   == 3


# ── validate_clip_tensor ──────────────────────────────────────────────────────

class TestValidateClipTensor:
    def test_valid_tensor_passes(self):
        t = torch.zeros(1, 3, 24, 480, 848)
        validate_clip_tensor(t, "clip")  # should not raise

    def test_valid_negative_one_to_one(self):
        t = torch.rand(1, 3, 24, 64, 64) * 2.0 - 1.0
        validate_clip_tensor(t)  # should not raise

    def test_raises_on_wrong_ndim(self):
        with pytest.raises(ValueError, match="5-D"):
            validate_clip_tensor(torch.zeros(3, 24, 64, 64))  # 4-D, missing batch

    def test_raises_on_out_of_range_high(self):
        t = torch.ones(1, 3, 4, 8, 8) * 1.5
        with pytest.raises(ValueError, match="outside \\[-1, 1\\]"):
            validate_clip_tensor(t, "test_clip")

    def test_raises_on_out_of_range_low(self):
        t = torch.ones(1, 3, 4, 8, 8) * -2.0
        with pytest.raises(ValueError, match="outside \\[-1, 1\\]"):
            validate_clip_tensor(t)

    def test_boundary_values_accepted(self):
        t = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)
        validate_clip_tensor(t)  # should not raise


# ── overwrite_anchor_regions ──────────────────────────────────────────────────

class TestOverwriteAnchorRegions:
    def _layout(self) -> WanLatentLayout:
        return derive_latent_layout(
            num_frames=72, height=480, width=848, anchor_frames=24
        )

    def _make_canvas(self, fill: float = 0.0) -> torch.Tensor:
        layout = self._layout()
        return torch.full(
            (1, 16, layout.latent_frames, layout.latent_h, layout.latent_w), fill
        )

    def test_start_region_is_overwritten(self):
        layout  = self._layout()
        canvas  = self._make_canvas(0.0)
        z_start = torch.ones(1, 16, layout.anchor_len, layout.latent_h, layout.latent_w)
        z_end   = torch.full_like(z_start, 2.0)

        result = overwrite_anchor_regions(canvas, z_start, z_end, layout)
        assert torch.all(result[:, :, :layout.anchor_len, :, :] == 1.0)

    def test_end_region_is_overwritten(self):
        layout  = self._layout()
        canvas  = self._make_canvas(0.0)
        z_start = torch.ones(1, 16, layout.anchor_len, layout.latent_h, layout.latent_w)
        z_end   = torch.full_like(z_start, 2.0)

        result = overwrite_anchor_regions(canvas, z_start, z_end, layout)
        assert torch.all(result[:, :, layout.mid_end:, :, :] == 2.0)

    def test_middle_region_is_unchanged(self):
        layout  = self._layout()
        fill    = 99.0
        canvas  = self._make_canvas(fill)
        z_start = torch.zeros(1, 16, layout.anchor_len, layout.latent_h, layout.latent_w)
        z_end   = torch.zeros_like(z_start)

        result = overwrite_anchor_regions(canvas, z_start, z_end, layout)
        mid = result[:, :, layout.mid_start:layout.mid_end, :, :]
        assert torch.all(mid == fill)

    def test_does_not_mutate_input(self):
        layout  = self._layout()
        canvas  = self._make_canvas(7.0)
        original = canvas.clone()
        z_start = torch.zeros(1, 16, layout.anchor_len, layout.latent_h, layout.latent_w)
        z_end   = torch.zeros_like(z_start)

        overwrite_anchor_regions(canvas, z_start, z_end, layout)
        assert torch.all(canvas == original)

    def test_output_shape_matches_canvas(self):
        layout  = self._layout()
        canvas  = self._make_canvas()
        z_start = torch.zeros(1, 16, layout.anchor_len, layout.latent_h, layout.latent_w)
        z_end   = torch.zeros_like(z_start)

        result = overwrite_anchor_regions(canvas, z_start, z_end, layout)
        assert result.shape == canvas.shape


# ── Normalisation helpers ─────────────────────────────────────────────────────

class TestNormalisation:
    def test_normalize_maps_zero_to_minus_one(self):
        t = torch.zeros(1, 3, 4, 8, 8)
        out = normalize_clip_to_neg_one_to_one(t)
        assert torch.allclose(out, torch.full_like(out, -1.0))

    def test_normalize_maps_one_to_one(self):
        t = torch.ones(1, 3, 4, 8, 8)
        out = normalize_clip_to_neg_one_to_one(t)
        assert torch.allclose(out, torch.ones_like(out))

    def test_normalize_maps_half_to_zero(self):
        t = torch.full((1, 3, 4, 8, 8), 0.5)
        out = normalize_clip_to_neg_one_to_one(t)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_round_trip(self):
        original = torch.rand(1, 3, 4, 8, 8)  # [0, 1]
        recovered = denormalize_clip_to_zero_to_one(
            normalize_clip_to_neg_one_to_one(original)
        )
        assert torch.allclose(original, recovered, atol=1e-6)
