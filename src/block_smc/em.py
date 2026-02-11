"""Full EM loop: alternates boundary optimisation (E-step) and twist training (M-step).

Components:
    run_em    — outer EM loop returning EMHistory with per-iteration diagnostics
    EMHistory — dataclass storing boundaries, twist loss, log_ml, metrics per iteration
"""
