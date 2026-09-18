# Instrument constant backup — `reference_v4.npz` @ sha256 459fd9a7…

The reference artifact that scored evals/028 (paper arms, grid v3) and evals/030 (externals) — its sha is
recorded in both evals' `meta.yaml` — existed ONLY as an uncommitted modification in the `eval-v4-cert`
worktree (branch `eval/v4-metrics` @ 4e792c0). Every committed `reference_v4.npz` (eval/v4-metrics HEAD,
feature-store HEAD) is sha256 e6ea4011… (the 4.0.0-certified build). This copy (cp -p, mtime preserved) is a
safeguard until P3b commits it with the 4.0.1-draft.1 version bump, where P4's bar-8 rebuild-parity check
verifies it. Do not delete; do not use directly — the scorer reads its own package copy.
