"""Long-hold dividend research system V4."""

from .core import (
    ContractError,
    allocate_core_targets,
    audit_snapshot,
    compute_price_features,
    entry_decision,
    estimate_trade_cost,
    load_config,
    score_universe,
    t_decision,
)

__all__ = [
    "ContractError",
    "allocate_core_targets",
    "audit_snapshot",
    "compute_price_features",
    "entry_decision",
    "estimate_trade_cost",
    "load_config",
    "score_universe",
    "t_decision",
]
