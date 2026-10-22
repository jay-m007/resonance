"""Measuring how well Resonance works, on labelled data.

Two questions:

1. Topics: do the discovered topics match the true intents?
   * purity: share of assigned tickets whose topic's most common intent is their own
   * NMI (normalised mutual information, 0..1): how much the topics tell you about the intent
   * ARI (adjusted Rand index, about 0 for random, 1 for perfect): pairwise agreement
   * coverage: share of tickets assigned to a topic (the rest are "unsorted")
   * intents found: share of true intents that are the majority of at least one topic
   Unsorted tickets are left out of purity/NMI/ARI and reported as coverage instead.
   nmi_all and ari_all count "unsorted" as one more topic, the like-for-like comparison
   with methods that sort every ticket.

2. Trends: in a simulated month where one intent spikes in the final week,
   is it flagged rising, and how many other topics are wrongly flagged?

Baselines, so the numbers mean something:
   * kmeans-oracle-k: k-means given the TRUE number of intents (an advantage no real user has)
   * kmeans-silhouette: k-means choosing k by the usual silhouette score
   * keyword: each ticket grouped by its single most distinctive word (what a quick script would do)
"""
from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .analyze import analyze
from .cluster import UNSORTED, cluster
from .embed import embed
from .ingest import Ticket, clean
from .simulate import simulate


def topic_metrics(true: list[str], pred: np.ndarray) -> dict:
    pred = np.asarray(pred)
    m = pred != UNSORTED
    t = np.asarray(true)[m]
    p = pred[m]
    if len(p) == 0:
        return {"nmi_all": 0.0, "ari_all": 0.0, "purity": 0.0, "nmi": 0.0, "ari": 0.0, "coverage": 0.0, "topics": 0, "intents_found": 0.0}
    majority = {j: Counter(t[p == j]).most_common(1)[0] for j in set(p)}
    purity = sum(c for _, c in majority.values()) / len(p)
    found = {lab for lab, _ in majority.values()}
    # Strict versions that count "unsorted" as one more (bad) topic, for a like-for-like comparison.
    whole = np.where(pred == UNSORTED, pred.max() + 1, pred)
    return {
        "nmi_all": round(float(normalized_mutual_info_score(true, whole)), 3),
        "ari_all": round(float(adjusted_rand_score(true, whole)), 3),
        "purity": round(purity, 3),
        "nmi": round(float(normalized_mutual_info_score(t, p)), 3),
        "ari": round(float(adjusted_rand_score(t, p)), 3),
        "coverage": round(float(m.mean()), 3),
        "topics": len(set(p)),
        "intents_found": round(len(found) / len(set(true)), 3),
    }


def keyword_baseline(texts: list[str]) -> np.ndarray:
    tv = TfidfVectorizer(stop_words="english", min_df=5)
    X = tv.fit_transform(texts)
    best = np.asarray(X.argmax(axis=1)).ravel()
    empty = np.asarray(X.sum(axis=1)).ravel() == 0
    best[empty] = UNSORTED
    return best


def kmeans_silhouette(X: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    best_k, best_s = 2, -1.0
    for k in sorted(set(int(round(x)) for x in np.geomspace(4, 60, 10))):
        lab = KMeans(k, random_state=seed, n_init=3).fit_predict(X[idx])
        s = silhouette_score(X[idx], lab, metric="cosine", sample_size=2000, random_state=seed)
        if s > best_s:
            best_k, best_s = k, s
    return KMeans(best_k, random_state=seed, n_init=3).fit_predict(X)


def evaluate_topics(tickets: list[Ticket], seed: int = 0, methods: list[str] | None = None) -> dict:
    labelled = [t for t in tickets if t.label]
    if len(labelled) < 50:
        raise ValueError("topic evaluation needs at least 50 labelled tickets")
    texts = [clean(t.text) for t in labelled]
    true = [t.label for t in labelled]
    n_true = len(set(true))
    started = time.time()
    X = embed(texts, seed=seed)
    embed_s = time.time() - started
    methods = methods or ["resonance", "kmeans-oracle-k", "kmeans-silhouette", "keyword"]
    results = {}
    for m in methods:
        s = time.time()
        if m == "resonance":
            pred = cluster(X, seed=seed).labels
        elif m == "kmeans-oracle-k":
            pred = KMeans(n_true, random_state=seed, n_init=4).fit_predict(X)
        elif m == "kmeans-silhouette":
            pred = kmeans_silhouette(X, seed)
        elif m == "keyword":
            pred = keyword_baseline(texts)
        else:
            raise ValueError(m)
        results[m] = topic_metrics(true, pred) | {"seconds": round(time.time() - s + (embed_s if m != "keyword" else 0), 1)}
    return {"tickets": len(labelled), "true_intents": n_true, "methods": results}


HIT_SHARE = 0.25


def evaluate_trends(pool: list[Ticket], runs: int = 10, factor: float = 3.0, seed: int = 0, per_day: int = 120) -> dict:
    """Simulates `runs` months, each with a different random intent spiking, and checks detection."""
    labels = sorted({t.label for t in pool if t.label})
    rng = random.Random(seed)
    spikes = rng.sample(labels, k=min(runs, len(labels)))
    rows = []
    for i, spike in enumerate(spikes):
        tickets, _ = simulate(pool, spike=spike, spike_factor=factor, seed=seed + i, per_day=per_day)
        r = analyze(tickets, seed=seed)
        label_of = {t.id: t.label for t in tickets}
        def spike_share(issue: dict) -> float:
            ids = issue["ticket_ids"]
            return sum(label_of[i] == spike for i in ids) / max(1, len(ids))
        rising = [x for x in r["issues"] if x["trend"]["status"] in ("rising", "new")]
        # A rising topic counts as catching the spike if at least a quarter of it is the spiking intent:
        # the spike's tickets are often split across a few topics, each of which then rises.
        hits = [x for x in rising if spike_share(x) >= HIT_SHARE]
        false_alarms = [x for x in rising if spike_share(x) < HIT_SHARE]
        spike_rank = next((x["rank"] for x in r["issues"] if spike_share(x) >= HIT_SHARE), None)
        rows.append({"spike": spike, "detected": bool(hits), "false_alarms": len(false_alarms), "top_rank_of_spike": spike_rank, "rising": len(rising)})
    n = len(rows)
    return {
        "runs": n,
        "factor": factor,
        "detection_rate": round(sum(r["detected"] for r in rows) / n, 3),
        "false_alarms_per_run": round(sum(r["false_alarms"] for r in rows) / n, 2),
        "spike_ranked_first": round(sum(r["top_rank_of_spike"] == 1 for r in rows) / n, 3),
        "detail": rows,
    }
