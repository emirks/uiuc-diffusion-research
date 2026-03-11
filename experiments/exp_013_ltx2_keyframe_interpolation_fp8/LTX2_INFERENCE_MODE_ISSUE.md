# Pipeline `__call__` not in inference mode → OOM when used from Python

If you use the pipelines from your own script (not the CLI), `__call__` isn’t under `torch.inference_mode()`. The text encoder keeps ~37 GB of graph/activations, so after you drop it and load the transformer you OOM.

**Fix:** Either put `@torch.inference_mode()` on each pipeline’s `__call__`, or mention in the README that callers should wrap pipeline calls in `torch.inference_mode()`.
