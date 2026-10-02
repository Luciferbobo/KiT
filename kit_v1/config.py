"""Config dataclasses (model / data / train / infer), presets, and YAML/JSON I/O.

Does not import torch; pyyaml is imported lazily. The full model presets live
in configs/*.yaml (e.g. --config configs/dit_300m.yaml).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Per-scale tables are compact strings, e.g.
#   data.scale_windows = "1m:1200/120,1d:500/20/125"   # scale:Lc/Lh[/trunc_min]
#   data.scale_strides = "1m:60,1d:10"                  # scale:stride (bars)
# Scales not listed fall back to the global Lc/Lh/trunc_min/stride.

_WIN_SPEC_HELP = ("format 'scale:Lc/Lh[/trunc_min]', multiple items separated by commas, "
                  "e.g. '1m:1200/120,1d:500/20/125'")
_STRIDE_SPEC_HELP = ("format 'scale:stride', multiple items separated by commas, "
                     "e.g. '1m:60,1d:10' (stride <=1 means online random start for that scale)")


def _known_timescales() -> Dict[str, int]:
    from kit_v1.data.calendar import TIMESCALE_IDS
    return TIMESCALE_IDS


def _split_scale_items(spec: str, key: str, help_text: str) -> List[Tuple[str, str]]:
    """Split "scale:value,scale:value" into [(scale, value_str)] and validate names."""
    known = _known_timescales()
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid {key} item (missing ':'): {item!r}; {help_text}")
        scale, _, rest = item.partition(":")
        scale, rest = scale.strip(), rest.strip()
        if scale not in known:
            raise ValueError(
                f"unknown timescale {scale!r} ({key}={spec!r}); "
                f"choose from {sorted(known, key=lambda s: known[s])}")
        if scale in seen:
            raise ValueError(f"{key}: timescale {scale!r} appears more than once ({spec!r})")
        if not rest:
            raise ValueError(f"{key}: scale {scale!r} has no value ({spec!r}); {help_text}")
        seen.add(scale)
        out.append((scale, rest))
    return out


def parse_timescales(spec: str) -> Optional[List[str]]:
    """Parse a timescale whitelist like "1m,30m" into a list; empty string gives None (no filter)."""
    known = _known_timescales()
    out: List[str] = []
    seen: set = set()
    for item in spec.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in known:
            raise ValueError(
                f"data.timescales contains unknown timescale {name!r} (spec={spec!r}); "
                f"choose from {sorted(known, key=lambda s: known[s])}")
        if name in seen:
            raise ValueError(
                f"timescale {name!r} appears more than once in data.timescales ({spec!r})")
        seen.add(name)
        out.append(name)
    return out or None


def _known_markets() -> Dict[str, int]:
    from kit_v1.data.calendar import MARKET_IDS
    return MARKET_IDS


def parse_market_weights(spec: str) -> Optional[Dict[str, float]]:
    """Parse 'ashare:0.5,us:0.3,crypto:0.2' into {market: weight}; empty gives None.

    Weights need not sum to 1 and may cover only some markets.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    known = _known_markets()
    out: Dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"invalid data.market_weights item (missing ':'): {item!r}; "
                "format 'market:weight,...', e.g. 'ashare:0.5,us:0.3,crypto:0.2'")
        mk, _, rest = item.partition(":")
        mk, rest = mk.strip(), rest.strip()
        if mk not in known:
            raise ValueError(
                f"data.market_weights contains unknown market {mk!r}; choose from {sorted(known)}")
        if mk in out:
            raise ValueError(f"market {mk!r} appears more than once in data.market_weights ({spec!r})")
        try:
            w = float(rest)
        except ValueError:
            raise ValueError(
                f"weight of {mk!r} in data.market_weights is not a number: {rest!r}") from None
        if w < 0:
            raise ValueError(f"weight of {mk!r} in data.market_weights must not be negative, got {w}")
        out[mk] = w
    if out and sum(out.values()) <= 0:
        raise ValueError(f"data.market_weights weights must sum to > 0 ({spec!r})")
    return out or None


def parse_scale_windows(spec: str) -> Dict[str, Tuple[int, int, int]]:
    """Parse the per-scale window table into {scale: (Lc, Lh, trunc_min)}.

    trunc_min is 0 when omitted (see DataConfig.trunc_min_for).
    """
    out: Dict[str, Tuple[int, int, int]] = {}
    for scale, rest in _split_scale_items(spec, "data.scale_windows", _WIN_SPEC_HELP):
        parts = [p.strip() for p in rest.split("/")]
        if len(parts) not in (2, 3):
            raise ValueError(
                f"data.scale_windows: scale {scale!r} has value {rest!r}, should be "
                f"'Lc/Lh' or 'Lc/Lh/trunc_min'; {_WIN_SPEC_HELP}")
        vals: List[int] = []
        for raw, name in zip(parts, ("Lc", "Lh", "trunc_min")):
            try:
                val = int(raw)
            except ValueError:
                raise ValueError(
                    f"data.scale_windows {scale}.{name} is not an integer: {raw!r}") from None
            if val <= 0:
                raise ValueError(
                    f"data.scale_windows {scale}.{name} must be a positive integer, got {val}")
            vals.append(val)
        Lc, Lh = vals[0], vals[1]
        trunc_min = vals[2] if len(vals) == 3 else 0
        if trunc_min > Lc:
            raise ValueError(
                f"data.scale_windows {scale}.trunc_min({trunc_min}) "
                f"must not exceed Lc({Lc})")
        out[scale] = (Lc, Lh, trunc_min)
    return out


def parse_scale_strides(spec: str) -> Dict[str, int]:
    """Parse the per-scale stride table into {scale: stride}."""
    out: Dict[str, int] = {}
    for scale, rest in _split_scale_items(spec, "data.scale_strides",
                                          _STRIDE_SPEC_HELP):
        try:
            val = int(rest)
        except ValueError:
            raise ValueError(
                f"stride of {scale} in data.scale_strides is not an integer: {rest!r}") from None
        if val < 0:
            raise ValueError(
                f"stride of {scale} in data.scale_strides must not be negative, got {val}")
        out[scale] = val
    return out


# Suggested per-scale windows and strides (Lc/Lh/trunc_min and start stride in bars).
# DataConfig defaults to empty strings, i.e. one global window for all scales.
DEFAULT_SCALE_WINDOWS: str = (
    "1m:1200/120/300,5m:960/96/240,15m:800/64/200,30m:640/40/160,"
    "1h:400/20/100,2h:360/20/90,1d:250/20/64"
)
DEFAULT_SCALE_STRIDES: str = "1m:60,5m:48,15m:32,30m:20,1h:10,2h:10,1d:10"


@dataclass
class ModelConfig:
    """Model architecture settings."""
    d_model: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    ffn_hidden: int = 5632          # SwiGLU hidden size
    in_features: int = 5            # diffusion state dimension
    rope_theta: float = 10000.0
    n_registers: int = 4            # register tokens
    inst_vocab: int = 32768         # instrument embedding slots
    inst_dim: int = 512             # low-rank instrument embedding dim
    market_slots: int = 16
    sector_slots: int = 64
    scale_slots: int = 16           # timescale embedding slots
    t_emb_dim: int = 256            # timestep sinusoidal dim
    clock_dim: int = 16             # clock-time Fourier dim
    sess_hidden: int = 256          # in-session position MLP hidden size
    yday_dim: int = 8               # day-of-year Fourier dim (sin/cos for k=1..4)
    qk_norm: bool = True
    attn_backend: str = "auto"      # auto | flash2 | sdpa | math
    use_activation_checkpointing: bool = False


@dataclass
class DataConfig:
    """Data pipeline settings."""
    data_root: str = ""
    stats_path: str = ""
    Lc: int = 1024                  # default history length
    Lh: int = 128                   # default horizon length
    trunc_prob: float = 0.3         # probability of random history truncation
    trunc_min: int = 256            # default minimum truncated length
    # Timescale whitelist, e.g. "30m" or "1m,30m". Empty = all scales in the data tree.
    timescales: str = ""
    # Timescale whitelist for validation only. Empty = same as timescales.
    val_timescales: str = ""
    scale_windows: str = ""         # "scale:Lc/Lh[/trunc_min],..."
    stride: int = 0                 # global start stride; <=1 = online random start
    scale_strides: str = ""         # "scale:stride,..."
    stride_jitter: float = 0.5      # stride jitter ratio +-round(stride*jitter), only if stride>1
    soft_clip_c: float = 6.0        # soft clip c*tanh(x/c)
    vol_ema_half_life_days: float = 5.0  # volume-ratio EMA half-life
    temperature_tau: float = 0.5    # bucket sampling temperature; 1.0 = epoch traversal mode
    per_inst_cap: int = 0           # max samples per instrument, 0 = unlimited
    # Market sampling weights, e.g. 'ashare:0.5,us:0.3,crypto:0.2'. Empty = by bar count^tau.
    market_weights: str = ""
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    synthetic: bool = False
    cache_max_inst: int = 256      # LRU capacity of the instrument cache, 0 = unlimited
    # If >0, only rows from (start - this * half-life) are processed per window
    # instead of the whole series. <=0 processes the full series.
    ema_warmup_halflives: float = 12.0
    raw_1m_root: str = ""           # raw 1m root; if set, aggregate to multi-scale tree before training
    agg_timescales: str = "1m,5m,15m,30m,1h,2h,1d"  # target scales, comma separated
    agg_workers: int = 8            # aggregation workers

    def timescales_list(self) -> Optional[List[str]]:
        """Timescale whitelist, or None for all scales."""
        return parse_timescales(self.timescales)

    def market_weights_map(self) -> Optional[Dict[str, float]]:
        """Parsed market weights, or None."""
        return parse_market_weights(self.market_weights)

    def val_timescales_list(self) -> Optional[List[str]]:
        """Validation timescale whitelist; falls back to timescales."""
        return parse_timescales(self.val_timescales) or self.timescales_list()

    def windows_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Parsed {scale: (Lc, Lh, trunc_min or 0)}."""
        return parse_scale_windows(self.scale_windows)

    def strides_map(self) -> Dict[str, int]:
        """Parsed {scale: stride}."""
        return parse_scale_strides(self.scale_strides)

    def window_for(self, timescale: str) -> Tuple[int, int]:
        """(Lc, Lh) for a scale, falling back to the global values."""
        entry = self.windows_map().get(timescale)
        return (entry[0], entry[1]) if entry is not None else (self.Lc, self.Lh)

    def trunc_min_for(self, timescale: str) -> int:
        """Minimum truncated length for a scale: the table value, else min(trunc_min, Lc)."""
        entry = self.windows_map().get(timescale)
        if entry is not None and entry[2] > 0:
            return entry[2]
        Lc = entry[0] if entry is not None else self.Lc
        return min(self.trunc_min, Lc)

    def stride_for(self, timescale: str) -> int:
        """Start stride for a scale, falling back to the global stride."""
        val = self.strides_map().get(timescale)
        return int(self.stride if val is None else val)

    def max_seq_len(self) -> int:
        """Longest Lc+Lh over the selected scales."""
        selected = self.timescales_list()
        if selected is not None:
            return max(sum(self.window_for(ts)) for ts in selected)
        cands = [self.Lc + self.Lh]
        cands += [Lc + Lh for Lc, Lh, _ in self.windows_map().values()]
        return max(cands)


@dataclass
class TrainConfig:
    """Training settings."""
    global_batch: int = 1024
    micro_batch: int = 8
    lr: float = 2e-4
    warmup_steps: int = 2000
    total_steps: int = 400000
    decay_frac: float = 0.2         # fraction of steps for the final decay
    min_lr: float = 1e-5
    lr_drop_step: int = 0           # >0: two-stage lr; from this step lr is multiplied by lr_drop_ratio
    lr_drop_ratio: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-15
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    precision: str = "bf16"         # fp32 | fp16 | bf16
    parallel: str = "fsdp"          # none | ddp | fsdp
    fsdp_sharding: str = "grad_op"  # grad_op | full
    inst_dropout: float = 0.1       # instrument embedding dropout
    allcond_dropout: float = 0.05   # drop all conditions
    hist_dropout: float = 0.1       # drop the whole history
    compile: bool = False
    seed: int = 42
    log_every: int = 10
    val_every: int = 1000
    val_batches: int = 8            # validation micro-batches; use >= number of scales
    ckpt_every: int = 1000
    out_dir: str = "runs/default"
    resume: str = ""


@dataclass
class InferConfig:
    """Inference settings."""
    steps: int = 32
    solver: str = "euler"           # euler | heun
    n_paths: int = 64
    cfg_w: float = 1.0              # CFG weight, 1.0 = off
    hist_cfg_w: float = 1.0         # history-CFG weight, 1.0 = off


@dataclass
class Config:
    """Top-level config: model / data / train / infer."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
            infer=InferConfig(**d.get("infer", {})),
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_yaml(self, path: str) -> None:
        """Write all four sections to YAML."""
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False,
                           allow_unicode=True, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str, base: "Config | None" = None) -> "Config":
        """Load a (possibly partial) YAML file on top of base (default Config()).

        Unknown sections or keys raise ValueError; values are converted to the field types.
        """
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML top level must be a mapping (file {path}), got {type(raw).__name__}")
        cfg = cls.from_dict((base if base is not None else cls()).to_dict())
        for sec_name, sec_raw in raw.items():
            if sec_name not in ("model", "data", "train", "infer"):
                raise ValueError(f"unknown config section: {sec_name!r} (file {path})")
            if sec_raw is None:
                continue
            if not isinstance(sec_raw, dict):
                raise ValueError(
                    f"config section {sec_name!r} must be a mapping (file {path}), "
                    f"got {type(sec_raw).__name__}")
            section = getattr(cfg, sec_name)
            names = {f.name for f in dataclasses.fields(section)}
            for field_name, value in sec_raw.items():
                key = f"{sec_name}.{field_name}"
                if field_name not in names:
                    raise ValueError(f"unknown config key: {key!r} (file {path})")
                cur = getattr(section, field_name)
                setattr(section, field_name, _coerce(value, type(cur), key))
        cfg.validate()
        return cfg

    def apply_overrides(self, overrides: List[str]) -> "Config":
        """Apply "section.key=value" overrides, converting to the field type."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"invalid override item (missing '='): {item!r}")
            key, _, raw = item.partition("=")
            key = key.strip()
            if "." not in key:
                raise ValueError(f"invalid override key (expected section.field): {key!r}")
            sec_name, _, field_name = key.partition(".")
            if sec_name not in ("model", "data", "train", "infer"):
                raise ValueError(f"unknown config section: {sec_name!r} (from {item!r})")
            section = getattr(self, sec_name)
            fields = {f.name: f for f in dataclasses.fields(section)}
            if field_name not in fields:
                raise ValueError(f"unknown config key: {key!r}")
            target_type = fields[field_name].type
            cur = getattr(section, field_name)
            setattr(section, field_name, _convert(raw.strip(), type(cur), key))
            del target_type
        self.validate()
        return self

    def validate(self) -> "Config":
        """Check values and cross-field constraints; raises ValueError."""
        parse_scale_windows(self.data.scale_windows)
        parse_scale_strides(self.data.scale_strides)
        parse_timescales(self.data.timescales)
        parse_timescales(self.data.val_timescales)
        parse_market_weights(self.data.market_weights)
        if self.data.stride < 0:
            raise ValueError(f"data.stride must not be negative, got {self.data.stride}")
        if self.data.stride_jitter < 0:
            raise ValueError(
                f"data.stride_jitter must not be negative, got {self.data.stride_jitter}")
        if not (0 < self.data.trunc_min <= self.data.Lc):
            raise ValueError(
                f"requires 0 < data.trunc_min({self.data.trunc_min}) "
                f"<= data.Lc({self.data.Lc})")
        if self.train.val_batches < 1:
            raise ValueError(
                f"train.val_batches must be at least 1, got {self.train.val_batches}")
        return self


def _convert(raw: str, ty: type, key: str) -> Any:
    """Convert a string to the target type."""
    if ty is bool:
        low = raw.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot parse {raw!r} as bool (key {key!r})")
    if ty is int:
        val = float(raw)
        if val != int(val):
            raise ValueError(f"cannot parse {raw!r} as int (key {key!r})")
        return int(val)
    if ty is float:
        return float(raw)
    if ty is str:
        return raw
    raise ValueError(f"unsupported field type {ty} (key {key!r})")


def _coerce(val: Any, ty: type, key: str) -> Any:
    """Validate/convert a YAML scalar to the target type."""
    if isinstance(val, str):
        return _convert(val.strip(), ty, key)
    if ty is bool:
        if isinstance(val, bool):
            return val
        raise ValueError(f"cannot parse {val!r} as bool (key {key!r})")
    if ty is int:
        if isinstance(val, bool):
            raise ValueError(f"cannot use bool {val!r} as int (key {key!r})")
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val == int(val):
            return int(val)
        raise ValueError(f"cannot parse {val!r} as int (key {key!r})")
    if ty is float:
        if isinstance(val, bool):
            raise ValueError(f"cannot use bool {val!r} as float (key {key!r})")
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError(f"cannot parse {val!r} as float (key {key!r})")
    if ty is str:
        import datetime
        # bare YAML dates are parsed as date objects
        if isinstance(val, (datetime.date, datetime.datetime)):
            return str(val)
        raise ValueError(f"cannot parse {val!r} as str (key {key!r})")
    raise ValueError(f"unsupported field type {ty} (key {key!r})")


def get_config(preset: str = "tiny") -> Config:
    """Return a preset config. Only "tiny" (small CPU smoke-test size) exists.

    For the full model sizes load configs/dit_*.yaml with Config.load_yaml.
    """
    if preset == "tiny":
        cfg = Config()
        cfg.model = ModelConfig(
            d_model=128, n_layers=2, n_heads=4, ffn_hidden=384,
            inst_vocab=128, inst_dim=32, t_emb_dim=64, sess_hidden=64,
            attn_backend="sdpa",
        )
        cfg.data.Lc = 64
        cfg.data.Lh = 16
        cfg.data.trunc_min = 16
        cfg.train.global_batch = 8
        cfg.train.micro_batch = 8
        cfg.train.precision = "fp32"
        cfg.train.parallel = "none"
        cfg.train.warmup_steps = 10
        cfg.train.total_steps = 50
        cfg.train.log_every = 1
        cfg.train.val_every = 0
        cfg.train.val_batches = 4
        cfg.train.ckpt_every = 25
        cfg.train.out_dir = "runs/tiny"
        cfg.infer.steps = 8
        cfg.infer.n_paths = 8
        return cfg
    raise ValueError(
        f"unknown preset: {preset!r} (only the 'tiny' smoke preset is supported; for the three family sizes dit_30m / "
        "dit_100m / dit_300m use --config configs/<name>.yaml, see README)")
