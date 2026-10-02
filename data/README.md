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

TODO: download link (cloud drive) will be added here and in the top-level README.
