"""Utilities for PINO-initialized NF2 extrapolation workflows."""

from .bridge import PinoField, load_pino_field, prepare_nf2_init_samples

__all__ = [
    "PinoField",
    "load_pino_field",
    "prepare_nf2_init_samples",
    "prefit_nf2_from_pino",
]


def __getattr__(name: str):
    if name == "prefit_nf2_from_pino":
        from .nf2_init import prefit_nf2_from_pino

        return prefit_nf2_from_pino
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
