"""Intrinsic-reward methods: method-specific scoring, scorer refreshes and their state.

lcg/              LCG: historical precision, candidate bank, scoring, reward scaling, lifecycle
ensemble.py       full diffusion-world-model ensemble disagreement (members, training, scorer)
plan2explore.py   Plan2Explore-style feature extraction, predictor-head fitting and disagreement
laplace.py        Laplace-sampled diffusion ensemble (uses lcg's historical precision)
intrinsic_eval.py real-episode intrinsic evaluation with each method's frozen scorer

Uniform random has no scorer. exploration_trainer.py orchestrates refreshes and scoring; configurations
are in config/intrinsic_reward/.
"""
