#!/usr/bin/env python3
"""Forecast Polymarket Next-CEO-of-NVIDIA Yes prices: 5 quantiles x 240 minutes.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes *.out to
$MAKEFUN_OUTPUT_DIR (else /output). Stdlib only, deterministic, offline.
"""
import json, math, os, glob

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
QUANT = [0.05, 0.25, 0.5, 0.75, 0.95]

def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def price6(x):
    r = round(clamp01(x) * 1e6) / 1e6
    return r + 0.0  # avoid -0.0

def median_sorted(a):
    n = len(a)
    if n == 0:
        return 0.0
    m = n // 2
    if n % 2:
        return a[m]
    return 0.5 * (a[m - 1] + a[m])

def quantile_sorted(a, tau):
    # linear interpolation on sorted data
    n = len(a)
    if n == 0:
        return 0.0
    if n == 1:
        return a[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return a[lo] * (1.0 - frac) + a[hi] * frac

def forecast_market(prices, horizon, params):
    """prices: list of minute prices ending at checkpoint. Returns 240x5 rows."""
    n = len(prices)
    x = prices[-1]
    mode = params.get("mode", "empirical")
    if mode == "baseline":
        sigma = 0.0025
        rows = []
        for h in range(1, horizon + 1):
            s = sigma * math.sqrt(h)
            rows.append([price6(x + z * s) for z in Z])
        return rows
    # ---- median anchor ----
    anchor_n = params.get("anchor_n", 1)
    anchor_w = params.get("anchor_w", 0.0)
    if anchor_n > 1 and n >= anchor_n:
        anchor = median_sorted(sorted(prices[-anchor_n:]))
    else:
        anchor = x
    med = x * (1.0 - anchor_w) + anchor * anchor_w
    # ---- spread ----
    spread = params.get("spread", "emp")
    win = params.get("win", 4320)
    seg = prices[-win:] if n > win else prices
    m = len(seg)
    shrink = params.get("shrink", 0.0)
    rows = []
    if spread == "fixed":
        sigma = params.get("sigma", 0.0025)
        for h in range(1, horizon + 1):
            s = sigma * math.sqrt(h)
            row = [price6(med + z * s) for z in Z]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        return rows
    if spread == "sig":
        diffs = [seg[i + 1] - seg[i] for i in range(m - 1)]
        if diffs:
            mu = sum(diffs) / len(diffs)
            var = sum((d - mu) ** 2 for d in diffs) / len(diffs)
            sigma = math.sqrt(var)
        else:
            sigma = 0.0025
        sigma = min(max(sigma, params.get("sigma_floor", 0.0005)), params.get("sigma_cap", 0.05))
        sat_H = params.get("sat_H", 0.0)  # 0 = no saturation
        tail = params.get("tail", 1.0)
        for h in range(1, horizon + 1):
            if sat_H:
                s = sigma * math.sqrt(h / (1.0 + h / sat_H))
            else:
                s = sigma * math.sqrt(h)
            s *= (1.0 - shrink)
            row = []
            for j, z in enumerate(Z):
                zz = z * tail if j in (0, 4) else z
                row.append(price6(med + zz * s))
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        return rows
    # spread == "emp": recentered empirical quantiles of h-step changes
    grid = params.get("grid", [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240])
    grid = [g for g in grid if g <= horizon and g < m]
    if not grid:
        grid = [1]
    qgrid = []
    for g in grid:
        ch = sorted(seg[i + g] - seg[i] for i in range(m - g))
        med_ch = quantile_sorted(ch, 0.5)
        row = [(quantile_sorted(ch, t) - med_ch) * (1.0 - shrink) for t in QUANT]
        qgrid.append(row)
    # optional floor: at least the baseline spread (prevents collapse on dead markets)
    floor_w = params.get("floor_w", 0.0)
    for h in range(1, horizon + 1):
        if h <= grid[0]:
            qs = qgrid[0]
        elif h >= grid[-1]:
            qs = qgrid[-1]
        else:
            j = 0
            while j < len(grid) - 1 and grid[j + 1] < h:
                j += 1
            g0, g1 = grid[j], grid[j + 1]
            w = (math.sqrt(h) - math.sqrt(g0)) / (math.sqrt(g1) - math.sqrt(g0))
            qs = [qgrid[j][k] * (1 - w) + qgrid[j + 1][k] * w for k in range(5)]
        if floor_w:
            s = 0.0025 * math.sqrt(h)
            qs = [qs[k] if abs(qs[k]) >= floor_w * abs(Z[k]) * s else math.copysign(floor_w * abs(Z[k]) * s, Z[k] if Z[k] != 0 else 1) for k in range(5)]
            qs[2] = 0.0
        row = [price6(med + q) for q in qs]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows

DEFAULT_PARAMS = {"mode": "empirical", "spread": "emp", "win": 720, "anchor_n": 1, "anchor_w": 0.0, "shrink": 0.35, "floor_w": 0.0}

def process_case(case, params):
    horizon = case["horizon_minutes"]
    forecasts = []
    for mk in case["markets"]:
        q = forecast_market(mk["prices"], horizon, params)
        forecasts.append({"id": mk["id"], "q": q})
    return {"forecasts": forecasts}

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    params = dict(DEFAULT_PARAMS)
    variant = os.environ.get("MAKEFUN_VARIANT")
    if variant:
        params.update(json.loads(variant))
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        out = process_case(case, params)
        base = os.path.basename(path)
        cid = base[:-3] if base.endswith(".in") else base
        with open(os.path.join(out_dir, cid + ".out"), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), sort_keys=True)

if __name__ == "__main__":
    main()
