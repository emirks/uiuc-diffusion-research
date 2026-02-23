from .schedules import BetaSchedule, linear_beta_schedule
from .forward import q_sample

# Optional Wan clip-connect exports. Wrapped so the base package remains usable
# in environments without the full Diffusers / Transformers video stack.
try:
    from .wan_clip_connect import (
        WanLatentLayout,
        WanVideoConnectingPipeline,
        denormalize_clip_to_zero_to_one,
        derive_latent_layout,
        normalize_clip_to_neg_one_to_one,
        overwrite_anchor_regions,
        validate_clip_tensor,
    )
except Exception:  # pragma: no cover
    pass

try:
    from .wan_clip_connect_slerp import (
        WanVideoConnectingSlerpPipeline,
        compute_slerp_guidance,
        slerp,
    )
except Exception:  # pragma: no cover
    pass
