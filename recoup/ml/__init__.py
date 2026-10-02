"""Phase 4: the learned recovery-propensity model.

This package is a strict *addition*.  Nothing here runs unless it is explicitly
wired in via :func:`recoup.agent.control.configure_model`.  The default policy
stays the rules heuristic, so the submission never depends on the model
training, loading, or being present at all.

Layout
------
* ``features``  -- the leakage-safe feature contract: a pure map from an
  observable record dict + a candidate intervention to a fixed-length vector.
* ``dataset``   -- builds the labelled potential-outcomes table from the frozen
  batch and the hidden environment (the only place latent truth is touched).
* ``model``     -- ``PropensityModel``: LightGBM + isotonic calibration, with a
  ``predict_proba(record_dict, intervention)`` that drops into the Phase-4 hook.
"""
