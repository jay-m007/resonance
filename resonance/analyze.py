"""The pipeline: tickets in, a ranked list of product issues out (as a JSON-ready dict)."""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .cluster import UNSORTED, cluster
from .describe import keywords, name_topic, representatives, severity, type_counts
from .embed import embed
from .ingest import Ticket, clean, redact
from .trends import trends

EXAMPLES_PER_ISSUE = 6


def impact(size: int, sev: float, status: str, ratio: float | None) -> float:
    """How much an issue deserves attention.

    impact = tickets x (1 + mean severity) x trend factor
    trend factor = 1 + min(ratio, 4) / 2 when rising or new (so up to 3x), else 1.
    Volume counts most; angry and fast-growing issues move up.
    """
    factor = 1.0
    if status in ("rising", "new"):
        factor = 1 + min(ratio or 4.0, 4.0) / 2
    return round(size * (1 + sev) * factor, 1)


def analyze(tickets: list[Ticket], backend: str = "tfidf", k: int | None = None, seed: int = 0) -> dict:
    started = time.time()
    if not tickets:
        raise ValueError("no tickets to analyze")
    texts = [clean(t.text) for t in tickets]
    X = embed(texts, backend=backend, seed=seed)
    cl = cluster(X, k=k, seed=seed)
    labels = cl.labels
    n_topics = len(cl.centroids)
    sev = np.array([severity(t.text) for t in tickets])

    groups = [np.where(labels == j)[0] for j in range(n_topics)]
    kw = keywords([[texts[i] for i in g] for g in groups])
    tr, period_info = trends(labels, [t.created for t in tickets], n_topics)

    issues = []
    for j, g in enumerate(groups):
        if len(g) == 0:
            continue
        reps = representatives(X, cl.centroids[j], g, EXAMPLES_PER_ISSUE)
        examples = [{"id": tickets[i].id, "text": redact(tickets[i].text)[:400]} for i in reps]
        mean_sev = float(sev[g].mean())
        t = tr[j]
        customers = {tickets[i].customer for i in g if tickets[i].customer}
        issue = {
            "id": j,
            "name": name_topic(kw[j], [e["text"] for e in examples]),
            "keywords": kw[j],
            "tickets": int(len(g)),
            "share": round(len(g) / len(tickets), 4),
            "customers": len(customers) if customers else None,
            "types": type_counts([tickets[i].text for i in g]),
            "severity": round(mean_sev, 3),
            "trend": {"status": t.status, "series": t.series, "latest": t.latest, "expected": t.expected, "ratio": t.ratio, "p_value": t.p_value},
            "cohesion": round(float(cl.similarity[g].mean()), 3),
            "examples": examples,
            "ticket_ids": [tickets[i].id for i in g],
        }
        issue["impact"] = impact(issue["tickets"], mean_sev, t.status, t.ratio)
        issue["type"] = max((x for x in issue["types"] if x != "other"), key=lambda x: issue["types"][x], default="other")
        if issue["types"][issue["type"]] == 0:
            issue["type"] = "other"
        if any(tk.label for tk in tickets):
            lab = Counter(tickets[i].label for i in g if tickets[i].label)
            top, cnt = lab.most_common(1)[0] if lab else (None, 0)
            issue["label"] = {"majority": top, "purity": round(cnt / len(g), 3)}
        issues.append(issue)

    issues.sort(key=lambda x: (-x["impact"], -x["tickets"]))
    for rank, iss in enumerate(issues, 1):
        iss["rank"] = rank

    unsorted = np.where(labels == UNSORTED)[0]
    dated = [t.created for t in tickets if t.created]
    return {
        "meta": {
            "tool": "resonance",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.time() - started, 2),
            "embedding": backend,
            "clustering": cl.method,
        },
        "summary": {
            "tickets": len(tickets),
            "issues": len(issues),
            "unsorted": int(len(unsorted)),
            "rising": sum(1 for i in issues if i["trend"]["status"] in ("rising", "new")),
            "first": min(dated).isoformat() if dated else None,
            "last": max(dated).isoformat() if dated else None,
            "customers": len({t.customer for t in tickets if t.customer}) or None,
        },
        "periods": period_info,
        "issues": issues,
        "unsorted_examples": [{"id": tickets[i].id, "text": redact(tickets[i].text)[:400]} for i in unsorted[:8]],
    }
