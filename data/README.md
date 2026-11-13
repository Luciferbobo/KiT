# data/

Put the 100-stock A-share demo dataset here (contents are git-ignored).

Expected layout:

```
data/
  instruments.json      market / sector / vocab_index of the 100 stocks
  stats.json            per-bucket normalisation statistics from training
  trade_dates.csv       A-share trading calendar
  ashare/{1m,5m,15m,30m,1h,2h,1d}/<code>.parquet
```

Download the demo data with Hugging Face (see the top-level README), or build
your own tree with `scripts/prepare_data.py`:

```bash
python scripts/prepare_data.py --raw-root raw/ashare_1m --out data --market ashare
```

See [`scripts/README.md`](../scripts/README.md) for details.
