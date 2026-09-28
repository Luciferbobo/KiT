
<h2 align="center">KiT: A Foundation Model for Financial Time-Series Forecasting using Diffusion Transformers</h2>

> KiT is a diffusion-based foundation model for candlestick (K-line) forecasting. It is trained on billions of bars spanning U.S. equities, Chinese A-shares, and cryptocurrencies across seven granularities, from one minute to one day, and achieves SOTA performance in both return forecasting and volatility prediction.


## Intro

KiT casts multi-horizon candlestick forecasting as conditional path generation via flow matching. The overall pipeline is illustrated below: raw OHLCV bars are encoded into a five-dimensional log-ratio state $x_t=(r_{\mathrm{gap}}, r_{\mathrm{body}}, r_{\mathrm{up}}, r_{\mathrm{dn}}, v_t)$, which is the state the diffusion model operates on. History and horizon are assembled into a single token sequence and processed by the KiT backbone, the history is returned bit-identical and only the forecast span is filled in with generated bars. KiT block employs QK-Norm and SwiGLU to improve training stability. Signals that are constant over the window modulate every layer through a shared AdaLN trunk, whereas signals that vary per bar are added directly to the token embeddings.

## Prediciton Demo

### K-line forecasting

### Backtest

### Return & volatility forecasting

## Get started

code will be available soon.

