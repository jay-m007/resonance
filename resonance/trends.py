"""Is a topic growing unusually fast?

Tickets are counted per period (days for up to 8 weeks of data, else weeks).
The most recent window (the last 7 days, or the last 2 weeks for weekly data)
is compared with everything before it. The expected recent count for a topic
is its earlier count times the *median* growth across all topics, so a busy
week for everyone isn't mistaken for a spike, and one topic spiking doesn't
make all the others look like they're falling (as scaling by the total would).

The test is a conditional binomial test. If nothing changed, each of a topic's
n = recent + earlier tickets falls in the recent window with probability
scale / (1 + scale), so the recent count follows Binomial(n, that probability).
This accounts for the earlier count being random too (a first version treated
it as an exact rate, a Poisson test, and raised far too many false alarms on
small topics). Because dozens of topics
are tested at once, the p-values are corrected with the Benjamini-Hochberg
procedure (false discovery rate FDR = 5%): of the topics flagged, at most
about 5% are expected to be false alarms. A topic is flagged as rising only if
it passes that test, the count is at least 1.5x the expectation, and at least
MIN_COUNT tickets. (A Bonferroni correction was tried first; it kept false
alarms at zero but missed most real spikes on the benchmark.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import binom

MIN_COUNT = 5
#: A topic needs this many earlier tickets before it can be called rising or falling;
#: with fewer, its "normal" level is too uncertain to compare against.
MIN_EARLIER = 10
MIN_RATIO = 1.5
FDR = 0.05


@dataclass
class Trend:
    status: str  # "rising" | "falling" | "steady" | "new" | "n/a"
    series: list[int] = field(default_factory=list)
    latest: int = 0
    expected: float = 0.0
    ratio: float | None = None
    p_value: float | None = None


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Which tests pass at false discovery rate q: sort p-values, find the largest i with p(i) <= q*i/m."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    ok = np.zeros(m, dtype=bool)
    if passed.any():
        last = np.max(np.where(passed)[0])
        ok[order[: last + 1]] = True
    return ok


def periods(times: list[datetime | None]) -> tuple[list[datetime], str] | None:
    known = [t for t in times if t is not None]
    if len(known) < 20:
        return None
    lo, hi = min(known), max(known)
    span = hi - lo
    if span < timedelta(days=2):
        return None
    step = timedelta(days=1) if span <= timedelta(days=56) else timedelta(days=7)
    unit = "day" if step.days == 1 else "week"
    # The last period ends at the newest ticket, so it's a full period, not a partial one.
    edges = [hi + timedelta(seconds=1)]
    while edges[-1] > lo:
        edges.append(edges[-1] - step)
    return list(reversed(edges)), unit


def bucket(times: list[datetime | None], edges: list[datetime]) -> np.ndarray:
    ts = np.array([t.timestamp() if t is not None else np.nan for t in times])
    e = np.array([x.timestamp() for x in edges])
    b = np.searchsorted(e, ts, side="right") - 1
    b[np.isnan(ts)] = -1
    b[(b < 0) | (b >= len(edges) - 1)] = -1
    return b


def trends(labels: np.ndarray, times: list[datetime | None], n_topics: int) -> tuple[list[Trend], dict | None]:
    p = periods(times)
    if p is None:
        return [Trend("n/a") for _ in range(n_topics)], None
    edges, unit = p
    b = bucket(times, edges)
    n_periods = len(edges) - 1
    if n_periods < 3:
        return [Trend("n/a") for _ in range(n_topics)], None

    totals = np.bincount(b[b >= 0], minlength=n_periods)
    # Recent window: the last 7 days (daily data) or 2 weeks (weekly), at most a third of the history.
    window = max(1, min(7 if unit == "day" else 2, n_periods // 3))
    last_total = totals[-window:].sum()
    earlier_total = totals[:-window].sum()
    all_series = [np.bincount(b[(labels == j) & (b >= 0)], minlength=n_periods) for j in range(n_topics)]
    growth = [s[-window:].sum() / s[:-window].sum() for s in all_series if s[:-window].sum() >= 20]
    scale = float(np.median(growth)) if growth else (last_total / earlier_total if earlier_total else 0.0)
    out: list[Trend] = []
    p_up = np.ones(n_topics)
    p_down = np.ones(n_topics)
    for j in range(n_topics):
        series = all_series[j]
        latest = int(series[-window:].sum())
        earlier = int(series[:-window].sum())
        expected = float(earlier * scale)
        t = Trend("steady", series.tolist(), latest, round(expected, 2))
        if earlier == 0 and latest >= MIN_COUNT:
            t.status = "new"
        elif expected > 0:
            t.ratio = round(latest / expected, 2)
            if earlier >= MIN_EARLIER:
                n = latest + earlier
                share = scale / (1 + scale)
                p_up[j] = float(binom.sf(latest - 1, n, share))  # P(recent >= latest)
                p_down[j] = float(binom.cdf(latest, n, share))  # P(recent <= latest)
                t.p_value = float(min(p_up[j], p_down[j]))
        out.append(t)

    up_ok = benjamini_hochberg(p_up, FDR)
    down_ok = benjamini_hochberg(p_down, FDR)
    for j, t in enumerate(out):
        if t.status != "steady" or t.ratio is None:
            continue
        if up_ok[j] and t.latest >= MIN_COUNT and t.ratio >= MIN_RATIO:
            t.status, t.p_value = "rising", float(p_up[j])
        elif down_ok[j] and t.expected >= MIN_COUNT and t.ratio <= 1 / MIN_RATIO:
            t.status, t.p_value = "falling", float(p_down[j])
    period_info = {
        "unit": unit,
        "starts": [e.astimezone(timezone.utc).isoformat() for e in edges[:-1]],
        "totals": totals.tolist(),
        "recent_window": window,
    }
    return out, period_info
