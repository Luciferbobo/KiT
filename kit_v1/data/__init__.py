"""Data pipeline: candle features, normalization, and calendar utilities.

- features:  candle features (r_gap/r_body/r_up/r_dn/v) and their inverse
- normalize: per-bucket robust (MAD) scaling with tanh soft clipping
- calendar:  trading calendar, per-bar calendar features, event tagging
"""

from kit_v1.data.features import FEATURE_NAMES, compute_features, invert_features
from kit_v1.data.normalize import (
    fit_mad_scale,
    fit_stats,
    load_stats,
    normalize,
    denormalize,
    save_stats,
)
from kit_v1.data.calendar import (
    MARKET_IDS,
    SESSION_TEMPLATES,
    TIMESCALE_IDS,
    TIMESCALE_MINUTES,
    attach_window_calendar,
    bars_per_day,
    bars_per_session,
    detect_events,
    gen_calendar,
    gen_future_timestamps,
)

__all__ = [
    "FEATURE_NAMES",
    "compute_features",
    "invert_features",
    "fit_mad_scale",
    "fit_stats",
    "save_stats",
    "load_stats",
    "normalize",
    "denormalize",
    "MARKET_IDS",
    "TIMESCALE_IDS",
    "TIMESCALE_MINUTES",
    "SESSION_TEMPLATES",
    "bars_per_day",
    "bars_per_session",
    "gen_calendar",
    "gen_future_timestamps",
    "detect_events",
    "attach_window_calendar",
]
