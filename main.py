"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its Exophase earn
rate (percent of tracked players who earned it) at a case's target instant.
The program runs once per case: it reads every *.in file in the input
directory and writes one *.out file per case with five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's rate at the target.

Model, in one paragraph: an earn rate is a saturating curve of play time, so
we work on the logit scale against x = ln(1 + hours since release), the
clock the house curve baseline also uses.  For each trophy we fit a robust
straight line of logit(rate) on x, shrink its slope toward its tier's mean
slope and the global mean slope (small-sample trophies borrow the trend of
their peers), and re-anchor the line on the recent reads.  In parallel we
fit the same kind of line to log(players_tracked) and to log(earners) --
earners being rate * players / 100 -- and divide at the target: that second
view sees dilution (a trophy whose earners stall while the player pool
grows is *falling* in rate even if its recent rates look flat).  The two
medians are blended on the logit scale.  The spread grows from the fit's
residual scatter plus a random-walk term in sqrt(horizon days), with a
floor wide enough that one Exophase refresh step between checkpoint and
target stays inside the 90% band.

Everything is deterministic: fixed constants, sorted file order, no clock,
no randomness, no network.  Python 3.13 standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- model constants (tuned on synthetic launch-week histories, see notes) ---
SLOPE_CAP_LO = -1.2        # per unit of ln(1 + hours): dilution can be steep
SLOPE_CAP_HI = 1.8         # but a rising trophy must not explode either
SHRINK_K = 5.0             # distinct reads at which a fit is half believed
RESID_FLOOR = 0.12         # logit-scale residual scatter floor
RW_COEF = 0.30             # random-walk spread per sqrt(horizon day)
SIGMA_FLOOR = 0.40         # logit spread floor: covers a refresh step
SIGMA_CAP = 2.2
ANCHOR_DECAY = 0.6         # recency weight per unit of x when re-anchoring
EARNERS_MIN = 3.0          # log-earners fit needs earners above this


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


# --------------------------------------------------------------------------
# robust weighted line fit
# --------------------------------------------------------------------------

def fit_line(xs, ys):
    """Least-squares line y ~ a + b*x, one robust pass.

    Fits once, drops points whose residual exceeds 2.5 robust sigmas, and
    refits when enough points survive.  Returns (a, b, resid_sd, n_used).
    With fewer than 2 usable points the slope is 0 and the level is the mean.
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
                xs2 = [xs[i] for i in keep]
                ys2 = [ys[i] for i in keep]
                a, b = ols(xs2, ys2)
                resid = [y - (a + b * x) for x, y in zip(xs2, ys2)]
                n = len(xs2)
    if n >= 3:
        sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    elif resid:
        sd = math.sqrt(sum(r * r for r in resid) / len(resid))
    else:
        sd = 1.0
    return a, b, sd, n


def cap(value, lo, hi):
    return min(hi, max(lo, value))


# --------------------------------------------------------------------------
# the forecast for one case
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])
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

    # Read timeline on the x clock, with the player counts alongside.
    xs, players = [], []
    for read in history:
        t = parse_time(read["at"])
        xs.append(math.log(1.0 + max(-0.99, hours_between(rel, t))))
        p = read.get("players_tracked")
        players.append(float(p) if p else None)

    # Players line: log(players) on x, fitted once for the whole case; the
    # pool only grows, so the slope is floored at 0 and the target pool is
    # capped at 40x the last read (log-time growth cannot run away).
    px = [x for x, p in zip(xs, players) if p]
    py = [math.log(p) for p in players if p]
    if len(px) >= 2:
        pa, pb, _, _ = fit_line(px, py)
        pb = cap(pb, 0.0, 1.5)
        pa = py[-1] - pb * px[-1]
        players_target = math.exp(pa + pb * x_target)
        if players:
            last_p = [p for p in players if p][-1]
            players_target = cap(players_target, last_p * 0.95, last_p * 40.0)
    else:
        players_target = None

    # First pass: a robust line per trophy on the logit scale.
    fits = []
    for i in range(len(trophies)):
        pts = [(x, logit(read["rates"][i]))
               for x, read in zip(xs, history)
               if read["rates"][i] is not None]
        distinct = len({read["rates"][i] for read in history
                        if read["rates"][i] is not None})
        if pts:
            a, b, sd, _ = fit_line([p[0] for p in pts], [p[1] for p in pts])
        else:
            a, b, sd = 0.0, 0.0, 1.0
        fits.append({"pts": pts, "a": a, "b": b, "sd": sd,
                     "distinct": distinct})

    # Tier and global mean slopes over the trophies whose fits carry signal
    # (at least 3 distinct rates seen); these are the shrinkage priors.
    def mean_slope(rows):
        vals = [r["b"] for r in rows if r["distinct"] >= 3]
        return sum(vals) / len(vals) if vals else 0.0

    global_slope = mean_slope(fits)
    tier_slope = {}
    for tier in {t["tier"] for t in trophies}:
        rows = [f for f, t in zip(fits, trophies) if t["tier"] == tier]
        tier_slope[tier] = mean_slope(rows) if any(
            r["distinct"] >= 3 for r in rows) else global_slope

    zvals = [norm_ppf(tau) for tau in
             (case.get("levels") or LEVELS)]

    rows = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]
        pts = fit["pts"]
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break

        if not pts:
            # Never seen: fall back to a mid-low rate with a wide spread.
            med_logit, sigma = logit(0.5), 1.5
        elif fit["distinct"] <= 1:
            # Flat in every read.  The rate may still move: dilution as the
            # pool grows, or a first refresh jump.  Keep the level, widen.
            med_logit = logit(last_rate)
            sigma = math.sqrt(SIGMA_FLOOR ** 2 +
                              (RW_COEF * math.sqrt(horizon_days)) ** 2 + 0.45 ** 2)
            if last_rate >= 100.0:
                sigma = 0.30          # pinned at the top: it can only stay
            elif last_rate <= 0.0:
                sigma = 1.10          # pinned at zero, but someone may earn it
        else:
            # Shrink the slope: believed in proportion to the distinct
            # reads behind it, discounted by how scattered the fit is.
            prior = 0.65 * tier_slope[trophy["tier"]] + 0.35 * global_slope
            w = fit["distinct"] / (fit["distinct"] + SHRINK_K)
            w /= (1.0 + fit["sd"])
            b = cap(w * fit["b"] + (1.0 - w) * prior,
                    SLOPE_CAP_LO, SLOPE_CAP_HI)
            # Re-anchor on recent reads: weight decays with distance in x
            # from the last read, so the line passes through where the
            # trophy is now, not where the average says it was.
            x_last = pts[-1][0]
            num = den = 0.0
            for x, y in pts:
                wgt = ANCHOR_DECAY ** max(0.0, x_last - x)
                num += wgt * (y - b * x)
                den += wgt
            a = num / den
            med_logit = a + b * x_target

            # Second view: earners and players separately.  Earners are
            # rate * players / 100; fit log(earners) on x where the count
            # is big enough for the log to mean something, then divide by
            # the target pool.  Blend the two medians evenly.
            ex, ey = [], []
            for x, read, p in zip(xs, history, players):
                r = read["rates"][i]
                if r is not None and p:
                    earners = r * p / 100.0
                    if earners >= EARNERS_MIN:
                        ex.append(x)
                        ey.append(math.log(earners))
            if len(ex) >= 3 and players_target:
                ea, eb, _, _ = fit_line(ex, ey)
                eb = cap(eb, -0.5, 2.5)
                ea = ey[-1] - eb * ex[-1]
                earners_target = math.exp(ea + eb * x_target)
                rate_b = clamp_rate(100.0 * earners_target / players_target)
                med_logit = 0.5 * med_logit + 0.5 * logit(rate_b)

            resid = max(fit["sd"], RESID_FLOOR)
            sigma = math.sqrt(resid ** 2 +
                              (RW_COEF * math.sqrt(horizon_days)) ** 2 +
                              SIGMA_FLOOR ** 2)
            if fit["distinct"] < 4:
                sigma *= 1.0 + 0.125 * (4 - fit["distinct"])
            sigma = min(sigma, SIGMA_CAP)

        med_logit = cap(med_logit, logit(0.01), logit(99.99))
        qs = []
        for z in zvals:
            qs.append(round4(inv_logit(med_logit + z * sigma)))
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
