"""Decimal operations with explicit contexts for reproducible data processing."""

from collections.abc import Sequence
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import ceil, floor


def sum_decimals(values: Sequence[Decimal]) -> Decimal:
    """Sum validated finite decimals without rounding away small summands."""
    if not values:
        return Decimal(0)
    places = max(value.adjusted() for value in values) - min(
        value.as_tuple().exponent for value in values
    ) + 1
    precision = max(28, places + len(str(len(values))))
    with localcontext(Context(prec=precision, Emin=MIN_EMIN, Emax=MAX_EMAX)):
        return sum(values, Decimal(0))


def mean_decimals(values: Sequence[Decimal]) -> Decimal:
    """Round the exact total's mean to 28 significant digits, half even."""
    total = sum_decimals(values)
    with localcontext(Context(
        prec=28, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX
    )):
        return total / len(values)


def round_to_grid(value: Decimal | Fraction, tick: Decimal, *, upward: bool) -> Decimal:
    """Round to a multiple of a positive tick without losing off-grid differences."""
    units = Fraction(value) / Fraction(tick)
    steps = Decimal(ceil(units) if upward else floor(units))
    with localcontext() as context:
        context.prec = max(context.prec, len(steps.as_tuple().digits) + len(tick.as_tuple().digits))
        return steps * tick
