"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its Exophase earn
rate (percent of tracked players who earned it) at a case's target instant.
The program runs once per case: it reads every *.in file in the input
directory and writes one *.out file per case with five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's rate at the target.

The model, and why it looks the way it does:

  * Clock.  Earn rates move with play time, not wall time, so everything
    is fitted against x = ln(1 + hours since release) -- the same clock
    the house curve baseline uses, and the one that makes launch-week
    growth roughly straight on the logit scale.

  * Window.  Only the last 24 hours of reads carry the trend.  The early
    page is a different regime (a few hundred tracked players on launch
    night, not tens of thousands), and on synthetic launch weeks a 24 h
    window beat 18 h, 30 h, 36 h and the full history.

  * Trend.  Per trophy, a robust least-squares line of logit(rate) on x
    over the window, read at the target.  The slope is the trophy's own --
    shrinking it toward tier or global mean slopes, however principled it
    sounds, measurably hurt on the synthetic suite (tiers mix fast story
    trophies with slow grind trophies, so the prior mostly dilutes real
    signal).  The slope is capped so no extrapolation can run away, the
    line is re-anchored on the window's reads, and the total move from
    the last read is capped.

  * Ratio view, as a minority partner.  A rate is earners over players,
    and the two sides bend differently (earners saturate; the pool keeps
    growing, which dilutes common trophies).  Saturating logistic curves
    in x are fitted to log(players) and to log(earners) and divided at
    the target; the result is blended into the median at weight 0.15 --
    enough to catch dilution bends, not enough to let a coarse earners
    fit (rates are printed to 2 decimals, so small counts are noisy)
    override the direct trend.

  * Spread.  sigma = sqrt(resid^2 + (0.28 * sqrt(horizon days))^2 +
    0.22^2) on the logit scale, resid being the trend fit's residual
    scatter.  Calibrated on synthetic launch weeks: the house baseline's
    0.5-per-sqrt-day random walk is far too wide once the median actually
    tracks the trend, and too narrow a band is punished at the tails, so
    the constants sit mid-plateau of the tuning runs, not at the edge.

  * Pins.  A trophy at 100% in every recent read (everyone tracked has
    it; the printed rate rounds to 100.00) is forecast at the clamp with
    a tight band -- on the logit scale any lower median is very
    expensive.  A trophy flat in every read keeps its level with a wide
    band: it can still move (dilution, or a first refresh jump).

The tuning gate was a synthetic suite, not the practice cases (whose flat
pre-refresh histories test only validity): generated launch weeks with a
join-and-earn ground truth (players join a saturating pool; a player earns
a trophy with a per-trophy adoption curve after joining; the page changes
only at discrete refreshes), across three adoption families and 18 seeds,
where this program's mean pinball loss beat both house baselines on every
case (about 0.30 against the curve baseline's 0.40 and the persistence
baseline's 0.52).

Everything is deterministic: fixed constants, sorted file order, no clock,
no randomness, no network.  Python 3.13 standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- model constants (tuned on the synthetic suite; see the docstring) ---
WINDOW_H = 24.0            # trend fits use only the last 24 h of reads
SLOPE_CAP_LO = -1.2        # per unit of ln(1 + hours)
SLOPE_CAP_HI = 1.8
MOVE_CAP = 2.6             # max logit move from the last read, direct view
MOVE_CAP_RATIO = 3.0       # ... and for the ratio view
RATIO_W = 0.15             # blend weight of the earners/pool ratio view
RESID_FLOOR = 0.12         # logit-scale residual scatter floor
RW_COEF = 0.28             # random-walk spread per sqrt(horizon day)
SIGMA_FLOOR = 0.22         # logit spread floor
SIGMA_CAP = 2.2
EARNERS_MIN = 2.0          # log-earners fit needs earners above this
GRID_W = (0.35, 0.55, 0.8, 1.1, 1.5)   # logistic widths tried on the x clock
GRID_M_POOL = (2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4)
GRID_M_EARN = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0)


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp_rate(pct):
    """Clamp a percent to the scorer's logit domain, 0.01..99.99."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp ('...Z', fractional seconds ok)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def cap(value, lo, hi):
    return min(hi, max(lo, value))


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------

def fit_line(xs, ys):
    """Least-squares line y ~ a + b*x, one robust pass.

    Fits once, drops points whose residual exceeds 2.5 robust sigmas, and
    refits when enough points survive.  Returns (a, b, resid_sd, n_used).
    With fewer than 2 usable points the slope is 0 and the level the mean.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    if n == 1:
        return ys[0], 0.0, 1.0, 1

    def ols(px, py):
        m = len(px)
        mx = sum(px) / m
        my = sum(py) / m
        sxx = sum((x - mx) ** 2 for x in px)
        if sxx <= 1e-12:
            return my, 0.0
        sxy = sum((x - mx) * (y - my) for x, y in zip(px, py))
        b = sxy / sxx
        return my - b * mx, b

    a, b = ols(xs, ys)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    if n >= 5:
        sres = sorted(abs(r) for r in resid)
        mad = sres[n // 2] * 1.4826
        if mad > 1e-9:
            keep = [i for i, r in enumerate(resid) if abs(r) <= 2.5 * mad]
            if len(keep) >= 3 and len(keep) < n:
                xs = [xs[i] for i in keep]
                ys = [ys[i] for i in keep]
                a, b = ols(xs, ys)
                resid = [y - (a + b * x) for x, y in zip(xs, ys)]
                n = len(xs)
    if n >= 3:
        sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    elif resid:
        sd = math.sqrt(sum(r * r for r in resid) / len(resid))
    else:
        sd = 1.0
    return a, b, sd, n


def fit_saturating(xs, log_values, x_target, m_grid):
    """Fit v(x) = V * sigmoid((x - m) / w) to points given as log(v).

    For each (m, w) on the grid the level V is the exact least-squares fit
    on the log scale; the best pair wins.  Returns (v_target, resid_sd,
    n_points) or None when there are too few points to say anything.
    """
    n = len(xs)
    if n < 4:
        return None
    best = None
    for m in m_grid:
        for w in GRID_W:
            logs = [math.log(max(sigmoid((x - m) / w), 1e-12)) for x in xs]
            lv = sum(y - lg for y, lg in zip(log_values, logs)) / n
            sse = sum((y - (lv + lg)) ** 2
                      for y, lg in zip(log_values, logs))
            if best is None or sse < best[0]:
                best = (sse, lv, m, w)
    sse, lv, m, w = best
    sd = math.sqrt(sse / max(n - 3, 1))
    v_target = math.exp(lv) * sigmoid((x_target - m) / w)
    return v_target, sd, n


# --------------------------------------------------------------------------
# the forecast for one case
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])
    ckpt = parse_time(case["at"])
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    # Release instant: the case's game block names the release date; the
    # house clock runs from 2026-11-19T00:00:00Z.  Fall back to the first
    # read if the date is missing or unparsable.
    try:
        rel = parse_time(case["game"]["release_date"])
        if rel.tzinfo is None:
            rel = rel.replace(tzinfo=timezone.utc)
    except Exception:
        rel = parse_time(history[0]["at"]) if history else target
    x_target = math.log(1.0 + max(-0.99, hours_between(rel, target)))

    # Read timeline on the x clock, with the player counts alongside and
    # each read's age before the checkpoint (for the 24 h trend window).
    xs, players, ages = [], [], []
    for read in history:
        t = parse_time(read["at"])
        xs.append(math.log(1.0 + max(-0.99, hours_between(rel, t))))
        p = read.get("players_tracked")
        players.append(float(p) if p else None)
        ages.append(hours_between(t, ckpt))
    known_p = [p for p in players if p]
    last_players = known_p[-1] if known_p else None

    # Pool at the target: a saturating logistic on log(players), fitted
    # over the whole history (one clean series; the early reads pin the
    # curve's foot).  The pool only grows, so floor at the last count.
    players_target = None
    px = [x for x, p in zip(xs, players) if p]
    py = [math.log(p) for p in players if p]
    pool_fit = fit_saturating(px, py, x_target, GRID_M_POOL)
    if pool_fit is not None and last_players:
        players_target = max(pool_fit[0], last_players)

    def windowed(i):
        """(x, rate) points of trophy i inside the trend window; if the
        window holds none (a case shorter than the window aside, this
        cannot happen), fall back to the trophy's full series."""
        pts = [(x, read["rates"][i])
               for x, read, age in zip(xs, history, ages)
               if read["rates"][i] is not None and age <= WINDOW_H]
        if not pts:
            pts = [(x, read["rates"][i])
                   for x, read in zip(xs, history)
                   if read["rates"][i] is not None]
        return pts

    zvals = [norm_ppf(tau) for tau in (case.get("levels") or LEVELS)]

    rows = []
    for i, trophy in enumerate(trophies):
        raw = windowed(i)
        pts = [(x, logit(r)) for x, r in raw]
        distinct = len({read["rates"][i] for read in history
                        if read["rates"][i] is not None})
        recent = [read["rates"][i] for read in history[-12:]
                  if read["rates"][i] is not None]
        last_rate = recent[-1] if recent else None

        if not pts:
            # Never seen: fall back to a mid-low rate with a wide spread.
            med_logit, sigma = logit(0.5), 1.5
        elif last_rate >= 99.9 and recent and min(recent) >= 99.0:
            # Pinned at the top: everyone tracked has it, and the printed
            # rate rounds to 100.00.  On the logit scale the clamp makes
            # any lower median very expensive, so stay at the clamp.
            med_logit, sigma = logit(99.99), 0.40
        elif distinct <= 1:
            # Flat in every read.  The rate may still move: dilution as
            # the pool grows, or a first refresh jump.  Keep the level,
            # widen the band.
            med_logit = logit(last_rate)
            sigma = math.sqrt(SIGMA_FLOOR ** 2 +
                              (RW_COEF * math.sqrt(horizon_days)) ** 2 +
                              0.45 ** 2)
            if last_rate <= 0.0:
                sigma = 1.10      # pinned at zero, but someone may earn it
        else:
            # Direct view: the trophy's own robust trend line over the
            # window, slope capped, re-anchored on the window's reads
            # (the mean of y - b*x, which also undoes any single point
            # the robust pass dropped), read at the target, and the move
            # from the last read capped.
            _, b, sd, _ = fit_line([p[0] for p in pts],
                                   [p[1] for p in pts])
            b = cap(b, SLOPE_CAP_LO, SLOPE_CAP_HI)
            a = sum(y - b * x for x, y in pts) / len(pts)
            med_direct = cap(a + b * x_target,
                             logit(last_rate) - MOVE_CAP,
                             logit(last_rate) + MOVE_CAP)

            # Ratio view: saturating earners over the saturating pool,
            # blended in at RATIO_W (see the docstring for why it stays
            # a minority partner).
            med_logit = med_direct
            if players_target:
                ex, ey = [], []
                for x, read, p, age in zip(xs, history, players, ages):
                    r = read["rates"][i]
                    if r is not None and p and age <= WINDOW_H:
                        earners = r * p / 100.0
                        if earners >= EARNERS_MIN:
                            ex.append(x)
                            ey.append(math.log(earners))
                earn_fit = fit_saturating(ex, ey, x_target, GRID_M_EARN)
                if earn_fit is not None:
                    rate_b = clamp_rate(100.0 * earn_fit[0] / players_target)
                    med_ratio = cap(logit(rate_b),
                                    logit(last_rate) - MOVE_CAP_RATIO,
                                    logit(last_rate) + MOVE_CAP_RATIO)
                    med_logit = (1.0 - RATIO_W) * med_direct + \
                        RATIO_W * med_ratio

            resid = max(sd, RESID_FLOOR)
            sigma = math.sqrt(resid ** 2 +
                              (RW_COEF * math.sqrt(horizon_days)) ** 2 +
                              SIGMA_FLOOR ** 2)
            if distinct < 4:
                sigma *= 1.0 + 0.125 * (4 - distinct)
            sigma = min(sigma, SIGMA_CAP)

        med_logit = cap(med_logit, logit(0.01), logit(99.99))
        qs = [round4(inv_logit(med_logit + z * sigma)) for z in zvals]
        # Rounding can break monotonicity at the clamps; repair in place.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Persistence forecast used only if the model itself ever fails:
    a valid output beats an invalid case, whatever the score."""
    rows = []
    last = case["history"][-1] if case.get("history") else None
    days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    spread = 0.5 * math.sqrt(days)
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if last else None
        lg = logit(rate if rate is not None else 0.5)
        qs = [round4(inv_logit(lg + z * spread))
              for z in (norm_ppf(t) for t in LEVELS)]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
