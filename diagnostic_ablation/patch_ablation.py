"""Diagnostic patch (diagnostic-gaussian-h10-replayvalue-ablation): separate grouped replay sampling from the replay-value
loss, and add monitoring-only statistics. Exact string replacements on src/models/pmpo_beta.py (CRLF preserved).
Defaults (replay_sampling=False) leave every learning path unchanged. usage: python patch_ablation.py WORKTREE"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "src/models/pmpo_beta.py"
raw = path.read_bytes()
crlf = b"\r\n" in raw
text = raw.decode().replace("\r\n", "\n")

EDITS = [
    # 1. config field
    ("""    replay_value_coef: float = 0.0
    replay_starts: int = 4
""",
     """    replay_value_coef: float = 0.0
    replay_starts: int = 4
    # Diagnostic (replay-value ablation): install the grouped replay sampling (segments x replay_starts consecutive
    # starts) even when replay_value_coef == 0. The replay statistics are then computed without gradients for
    # monitoring only and never enter the loss. False (default) keeps the sampler tied to replay_value_coef > 0.
    replay_sampling: bool = False
"""),
    # 2. sampler selection separated from loss enablement
    ("""        if self.cfg.replay_value_coef > 0:
            # Reviewed DreamerV3-controller wiring""",
     """        if self.cfg.replay_value_coef > 0 or self.cfg.replay_sampling:
            # Reviewed DreamerV3-controller wiring"""),
    # 3. exact decomposition of the replay targets (monitoring only)
    ("""                         replay_boot_mean=boot.mean(), replay_reward_mean=rew[:, :K - 1].mean())
""",
     """                         replay_boot_mean=boot.mean(), replay_reward_mean=rew[:, :K - 1].mean())
            # Diagnostic: the replay lambda-return is linear in (rew, boot), so R = R(rew, 0) + R(0, boot) exactly.
            rew_part = replay_lambda_return(rew, term, last, torch.zeros_like(boot), c.gamma, c.lambda_).flatten()
            boot_part = replay_lambda_return(torch.zeros_like(rew), term, last, boot, c.gamma, c.lambda_).flatten()
            stats.update(replay_target_reward_part_mean=(weight * rew_part).sum() / wsum,
                         replay_target_boot_part_mean=(weight * boot_part).sum() / wsum,
                         replay_target_decomposition_error=(weight * (rew_part + boot_part - targets.flatten()).abs()).sum() / wsum)
"""),
    # 4. replay loss only when the coefficient is positive; monitoring otherwise
    ("""            imag_value_loss = value_loss.detach()
            repval_loss, replay_metrics = self.replay_value_loss(self.replay_loader.replay, targets[:, 0], obs[:, 0])
            value_loss = value_loss + c.replay_value_coef * repval_loss
            replay_metrics.update(loss_value_imag=imag_value_loss, loss_repval_weighted=c.replay_value_coef * repval_loss.detach())
""",
     """            imag_value_loss = value_loss.detach()
            if c.replay_value_coef > 0:
                repval_loss, replay_metrics = self.replay_value_loss(self.replay_loader.replay, targets[:, 0], obs[:, 0])
                value_loss = value_loss + c.replay_value_coef * repval_loss
                replay_metrics.update(loss_value_imag=imag_value_loss, loss_repval_weighted=c.replay_value_coef * repval_loss.detach())
            else:
                # Diagnostic (replay_sampling only): the same replay statistics without gradients; not in the loss.
                with torch.no_grad():
                    _, replay_metrics = self.replay_value_loss(self.replay_loader.replay, targets[:, 0], obs[:, 0])
                replay_metrics.update(loss_value_imag=imag_value_loss, loss_repval_weighted=torch.zeros((), device=self.device))
            replay_metrics["replay_loss_in_objective"] = torch.tensor(float(c.replay_value_coef > 0))
"""),
    # 5. distributions of imagined lambda-returns and critic values (monitoring only)
    ("""            metrics.update(summary_statistics(advantage, "advantage"))
""",
     """            metrics.update(summary_statistics(advantage, "advantage"))
            metrics.update(summary_statistics(targets, "lambda_return_dist"))
            metrics.update(summary_statistics(values.detach(), "value_dist"))
"""),
]
for old, new in EDITS:
    if text.count(old) != 1:
        raise SystemExit(f"edit target not found exactly once: {old[:80]!r}")
    text = text.replace(old, new)
out = text.replace("\n", "\r\n") if crlf else text
path.write_bytes(out.encode())
print(f"patched {path} ({len(EDITS)} edits, CRLF={crlf})")
