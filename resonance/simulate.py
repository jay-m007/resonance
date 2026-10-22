"""A simulated month of support tickets with a known spike.

Public support datasets don't come with realistic timestamps, so to test (and
demo) trend detection we spread labelled tickets over a period and make one
intent jump in the final week. Because we know which intent spiked, we can
measure whether Resonance finds it, and how often it raises false alarms.

Everything produced here is clearly synthetic; never present it as real data.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .ingest import Ticket


def simulate(
    pool: list[Ticket],
    days: int = 28,
    per_day: int = 120,
    spike: str | None = None,
    spike_factor: float = 3.0,
    spike_days: int = 7,
    seed: int = 0,
    start: datetime | None = None,
) -> tuple[list[Ticket], str]:
    """Returns tickets with timestamps (and their true label), plus the intent that spiked."""
    rng = random.Random(seed)
    by_label: dict[str, list[Ticket]] = defaultdict(list)
    for t in pool:
        if t.label:
            by_label[t.label].append(t)
    labels = sorted(by_label)
    if not labels:
        raise ValueError("simulation needs labelled tickets (an intent/label column)")
    spike = spike or rng.choice(labels)
    if spike not in by_label:
        raise ValueError(f"unknown label {spike!r}")
    weights = {l: rng.uniform(0.5, 1.5) for l in labels}  # topics differ in normal volume
    start = start or datetime(2026, 9, 1, tzinfo=timezone.utc)

    out: list[Ticket] = []
    used: set[int] = set()
    for d in range(days):
        day = start + timedelta(days=d)
        w = dict(weights)
        if d >= days - spike_days:
            w[spike] *= spike_factor
        total_w = sum(w.values())
        volume = int(per_day * rng.uniform(0.85, 1.15) * (1 + 0.15 * (day.weekday() < 5)))  # busier on weekdays
        for _ in range(volume):
            r = rng.random() * total_w
            for label in labels:
                r -= w[label]
                if r <= 0:
                    break
            candidates = by_label[label]
            src = candidates[rng.randrange(len(candidates))]
            ts = day + timedelta(seconds=rng.randrange(86400))
            out.append(Ticket(id=f"sim-{len(out) + 1}", text=src.text, created=ts, customer=f"cust-{rng.randrange(1, 900)}", label=label))
            used.add(id(src))
    out.sort(key=lambda t: t.created)
    return out, spike


def write_csv(tickets: list[Ticket], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "created_at", "customer", "text", "intent"])
        for t in tickets:
            w.writerow([t.id, t.created.isoformat() if t.created else "", t.customer or "", t.text, t.label or ""])
