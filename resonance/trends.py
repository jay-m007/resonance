"""Is a topic growing unusually fast?

Tickets are counted per period (days for up to 8 weeks of data, else weeks).
The most recent window (the last 7 days, or the last 2 weeks for weekly data)
is compared with everything before it. The expected recent count for a topic
is its earlier count times the *median* growth across all topics, so a busy
week for everyone isn't mistaken for a spike, and one topic spiking doesn't
make all the others look like they're falling (as scaling by the total would).

The observed count is compared with that expectation using a Poisson test:
how likely is a count at least this large by chance? A topic is flagged as
rising only if the count is at least 1.5x the expectation, at least
MIN_COUNT tickets, and the probability is below 0.01 divided by the number of
topics (a Bonferroni correction: with 40 topics, testing each at 1% would
raise a false alarm somewhere almost every time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import poisson

MIN_COUNT = 5
MIN_RATIO = 1.5
ALPHA = 0.01


@dataclass
class Trend:
    status: str  # "rising" | "falling" | "steady" | "new" | "n/a"
    series: list[int] = field(default_factory=list)
    latest: int = 0
    expected: float = 0.0
    ratio: float | None = None
    p_value: float | None = None


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
    alpha = ALPHA / max(1, n_topics)
    all_series = [np.bincount(b[(labels == j) & (b >= 0)], minlength=n_periods) for j in range(n_topics)]
    growth = [s[-window:].sum() / s[:-window].sum() for s in all_series if s[:-window].sum() >= MIN_COUNT]
    scale = float(np.median(growth)) if growth else (last_total / earlier_total if earlier_total else 0.0)
    out: list[Trend] = []
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
            p_up = float(poisson.sf(latest - 1, expected))  # P(X >= latest)
            p_down = float(poisson.cdf(latest, expected))  # P(X <= latest)
            if latest >= MIN_COUNT and t.ratio >= MIN_RATIO and p_up < alpha:
                t.status, t.p_value = "rising", p_up
            elif expected >= MIN_COUNT and t.ratio <= 1 / MIN_RATIO and p_down < alpha:
                t.status, t.p_value = "falling", p_down
            else:
                t.p_value = min(p_up, p_down)
        out.append(t)
    period_info = {
        "unit": unit,
        "starts": [e.astimezone(timezone.utc).isoformat() for e in edges[:-1]],
        "totals": totals.tolist(),
        "recent_window": window,
    }
    return out, period_info
