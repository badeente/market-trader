"""Validation shared by offline configuration readers."""

from decimal import Decimal


def require_fields(table: object, name: str, *required: str) -> dict:
    if not isinstance(table, dict):
        raise ValueError(f"{name} must be a table")
    fields = set(required)
    if unexpected := table.keys() - fields:
        raise ValueError(f"unexpected {name} fields: {', '.join(sorted(unexpected))}")
    if missing := fields - table.keys():
        raise ValueError(f"missing {name} fields: {', '.join(sorted(missing))}")
    return table


def config_decimal(value: object, name: str) -> Decimal:
    if type(value) is int:
        return Decimal(value)
    if isinstance(value, Decimal) and value.is_finite():
        return value
    raise ValueError(f"{name} must be a finite TOML number")
