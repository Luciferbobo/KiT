"""Compare warm FP32 MPS flow_sample and AppleSampler on a KiT prediction case."""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASE = "300368.SZ,1d,2026-01-05"


def load_window(root, cfg, case=CASE):
    from pandas import DatetimeIndex

    from eval.infer_for_pred import MARKET, load_case_frame, parse_case
    from kit_v1.data.calendar import bars_per_day, load_trading_days
    from kit_v1.data.normalize import load_stats
    from kit_v1.infer import prepare_inference_window

    code, scale, end, _ = parse_case(case)
    context, horizon = cfg.data.window_for(scale)
    half_life = cfg.data.vol_ema_half_life_days
    keep = context + math.ceil(12 * half_life * bars_per_day(MARKET, scale)) + 8
    history, future = load_case_frame(root, code, scale, end, context, horizon, keep)
    instrument = json.loads((root / "instruments.json").read_text())[code]
    win = prepare_inference_window(
        history,
        MARKET,
        scale,
        context,
        horizon,
        stats=load_stats(root / "stats.json"),
        soft_clip_c=cfg.data.soft_clip_c,
        half_life_days=half_life,
        end_ts=end,
        sector_id=int(instrument.get("sector", 0)),
        inst_id=int(instrument.get("vocab_index", 0)),
        trading_days=load_trading_days(root / "trade_dates.csv"),
    )
    if not DatetimeIndex(future["datetime"]).equals(win["future_timestamps"]):
        raise ValueError(f"The trading calendar does not match the following bars for {case}")
    return {**win, "symbol": code, "scale": scale}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default=CASE,
        metavar="CODE,SCALE,ANCHOR",
        help=f"Prediction case, e.g. 300591.SZ,5m,2026-01-23 (default: {CASE})",
    )
    parser.add_argument("--seed", type=int, default=6, help="Seed for the common initial noise")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "ckpt" / "step_market_284M.pt")
    parser.add_argument(
        "--steps", type=int, default=1, help="Euler steps per window; use 32 for the full sampler"
    )
    parser.add_argument(
        "--pairs", type=int, default=2, help="Timed pairs; order alternates AB then BA"
    )
    parser.add_argument("--out", type=Path, help="Optional JSON report (otherwise stdout only)")
    args = parser.parse_args()
    if min(args.steps, args.pairs) < 1:
        parser.error("--steps and --pairs must both be positive")
    if not torch.backends.mps.is_available():
        parser.error("An available MPS device is required")
    from kit_v1.infer import flow_sample
    from kit_v1.infer.apple import AppleSampler
    from kit_v1.infer.loader import load_model_and_config

    torch.set_num_threads(4)
    model, cfg = load_model_and_config(str(args.ckpt), "tiny", args.seed, mmap=True)
    win = load_window(args.data_root, cfg, args.case)
    inputs = (win["x_ctx"], win["cond"])
    horizon = len(win["future_timestamps"])
    options = dict(
        Lh=horizon, n_paths=16, chunk_size=16, steps=args.steps, solver="euler", precision="fp32"
    )
    rtol, atol = 1e-4, 2e-5

    def errors(value, reference):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("Sampler outputs must have the same shape and dtype")
        delta = value.double() - reference.double()
        return {
            "max_abs": delta.abs().max().item(),
            "relative_l2": delta.norm().item() / max(reference.double().norm().item(), 1e-30),
            "outside_tolerance": (~torch.isclose(value, reference, rtol=rtol, atol=atol))
            .sum()
            .item(),
        }

    pairs = []
    with AppleSampler(model.eval(), cpu_paths=4) as engine:

        def run(name):
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            torch.mps.synchronize()
            start = perf_counter()
            if name == "flow_sample":
                output = flow_sample(
                    engine.model, *inputs, **options, generator=generator, device="mps"
                )
            else:
                output = engine.sample(*inputs, **options, generator=generator)
            output = output.cpu()
            torch.mps.synchronize()
            return perf_counter() - start, output

        names = ("flow_sample", "apple")
        _, reference = run(names[0])
        _, warm = run(names[1])
        if engine.last_run.get("backend") != "apple_fp32":
            raise SystemExit(f"The optimized sampler was unavailable: {engine.last_run}")
        warm_errors = errors(warm, reference)
        for i in range(args.pairs):
            order = names if i % 2 == 0 else names[::-1]
            runs = {name: run(name) for name in order}
            base, candidate = (runs[name][1] for name in names)
            pairs.append(
                {
                    "order": order,
                    "seconds": {k: v[0] for k, v in runs.items()},
                    **errors(candidate, base),
                }
            )
        allocation = dict(engine.last_run)
    medians = {name: median(p["seconds"][name] for p in pairs) for name in names}
    report = {
        "case": args.case,
        "symbol": win["symbol"],
        "scale": win["scale"],
        "context": win["x_ctx"].shape[1],
        "horizon": horizon,
        "anchor": win["ctx_timestamps"][-1].isoformat(),
        "forecast_start": win["future_timestamps"][0].isoformat(),
        "forecast_end": win["future_timestamps"][-1].isoformat(),
        "seed": args.seed,
        "dtype": "float32",
        "paths": 16,
        "steps": args.steps,
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "scope": "warm sampler call through synchronized CPU output",
        "apple_allocation": allocation,
        "rtol": rtol,
        "atol": atol,
        "warmup_errors": warm_errors,
        "passed": all(row["outside_tolerance"] == 0 for row in [warm_errors, *pairs]),
        "pairs": pairs,
        "median_seconds": medians,
        "speedup": medians["flow_sample"] / medians["apple"],
    }
    result = json.dumps(report, indent=2)
    print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Apple and reference outputs differed beyond the reported FP32 tolerances")


if __name__ == "__main__":
    main()
