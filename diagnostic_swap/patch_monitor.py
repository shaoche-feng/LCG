"""Monitoring-only patch (diagnostic-gaussian-h10-ema-component-swap): log the distributions of the imagined
lambda-returns and critic values, as the replay-value ablation did. No learning path, RNG use or state changes.
usage: python patch_monitor.py WORKTREE"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "src/models/pmpo_beta.py"
raw = path.read_bytes()
crlf = b"\r\n" in raw
text = raw.decode().replace("\r\n", "\n")
old = """            metrics.update(summary_statistics(advantage, "advantage"))
"""
new = """            metrics.update(summary_statistics(advantage, "advantage"))
            # Diagnostic (component swap): distributions of the imagined lambda-returns and critic values; monitoring only.
            metrics.update(summary_statistics(targets, "lambda_return_dist"))
            metrics.update(summary_statistics(values.detach(), "value_dist"))
"""
if text.count(old) != 1:
    raise SystemExit("edit target not found exactly once")
text = text.replace(old, new)
path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode())
print(f"patched {path} (CRLF={crlf})")
