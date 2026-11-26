"""Predictor-based LCG (intrinsic_reward.method=lcg_predictor; design: docs/implement/LCG_PREDICTOR_DESIGN.md).

One feature predictor f_psi(z_t, a_t) -> z_{t+1} fitted on Plan2Explore's frozen-denoiser features by a fixed
budget of steps on the explicit MAP objective (not guaranteed to reach the MAP), a diagonal GGN-based Laplace
approximation around the fitted weights over all its parameters, and an LCG score of each candidate (z_t, a_t):
the trace of the approximate posterior's output covariance relative to the feature-likelihood variance, an
information-gain surrogate (not a calibrated uncertainty).

- lifecycle.py: configuration, refresh, frozen scoring hook, checkpoint state and milestone files
- training.py: the explicit MAP objective and its optimizer loop
- curvature.py: exact GGN diagonal (closed form and generic autograd) and the exact trace
- scoring.py: parameter layout, JVP probe bank and forward-mode scoring
"""
from .lifecycle import LCGPredictor, LCGPredictorConfig, MILESTONE_FILES

__all__ = ["LCGPredictor", "LCGPredictorConfig", "MILESTONE_FILES"]
