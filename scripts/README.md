# scripts/

Data preparation for the KiT training pipeline.

## `prepare_data.py`

Builds the directory tree and metadata the trainer/eval consume:

```
<out>/<market>/<timescale>/<code>.parquet
<out>/instruments.json
<out>/stats.json
<out>/trade_dates.csv
```

1. **aggregate** a raw fine-scale series (default `1m`) into the seven
   timescales with the session-aware grid in `kit_v1.data.aggregate` (use
   `--no-aggregate` if the raw root is already a multi-timescale tree);
2. **instruments.json** — `{code: {name, market, sector, vocab_index, list_date}}`
   with a 1-based `vocab_index` (0 is reserved for `[UNK]`);
3. **trade_dates.csv** — unique `YYYYMMDD` trading days taken from the 1d bars;
4. **stats.json** — per `<market>|<timescale>|<feature>` robust MAD scales fitted
   on rows up to `--train-end`.

```bash
python scripts/prepare_data.py --raw-root raw/ashare_1m --out data --market ashare

# reuse the metadata bundled with a pre-trained checkpoint (keeps the trained
# vocab_index / sector ids and the exact normalisation):
python scripts/prepare_data.py --raw-root raw/ashare_1m --out data --market ashare \
    --instruments ckpt/step_market_284M/instruments.json \
    --stats       ckpt/step_market_284M/stats.json
```

Raw layout accepted by `--raw-root` (with aggregation):

```
raw/            # flat: one parquet per instrument at the source timescale
  000001.SZ.parquet
  ...
```

or, with `--no-aggregate`, an existing tree:

```
raw/ashare/{1m,5m,15m,30m,1h,2h,1d}/<code>.parquet
```

Columns may be lower/upper case and `date`/`datetime`; they are normalised to
`datetime, open, high, low, close, volume`.

## `aggregate_1m.py`

Thin CLI around `aggregate_tree` that only writes the parquet tree (metadata is
built by `prepare_data.py`).

```bash
python scripts/aggregate_1m.py --raw-root raw/ashare_1m --out data --market ashare
```

## Notes

- Aggregation is session aware: a coarse bar's timestamp is its close minute and
  it aggregates every fine bar whose close falls in the interval, so the result
  lines up exactly with `kit_v1.data.calendar` (`bars_per_session`,
  `gen_calendar`) used at training and inference time.
- Markets and session templates come from `kit_v1.data.calendar` (`ashare`,
  `hk`, `us`, `futures`, `crypto`).
