# Data

- **`DATASET.md`**: the CTT v2 training dataset SPEC — the single source of truth for strata,
  counts, caption grammar, holdouts, accepted risks, the pre-launch assert suite and reproduction.
  Authoritative over any script or note that disagrees with it.
- `raw/`: immutable downloads
- `processed/`: derived datasets / cached tensors

Keep large files out of git. Document any preprocessing steps in the relevant experiment folder.
