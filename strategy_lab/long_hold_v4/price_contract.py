"""Price-basis boundaries shared by research, execution, and accounting."""

from __future__ import annotations

from typing import Any

from .core import ContractError


EXECUTABLE_PRICE_BASIS = "unadjusted_executable"
EXECUTABLE_RETURN_BASES = {EXECUTABLE_PRICE_BASIS}


def require_executable_price_basis(value: Any, label: str = "price_basis") -> str:
    basis = str(value).strip().lower()
    if basis not in EXECUTABLE_RETURN_BASES:
        raise ContractError(f"{label} must be unadjusted_executable")
    return basis
