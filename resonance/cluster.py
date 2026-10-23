"""Grouping tickets into topics.

Two steps:

1. Discover topics with HDBSCAN on a sample of tickets. HDBSCAN finds dense
   groups of similar tickets and decides the number of topics itself; tickets
   that don't clearly belong anywhere are left out instead of being forced
   into a topic.
2. Assign every ticket to the most similar topic centre (cosine similarity),
   but only if it is at least ASSIGN_THRESHOLD similar. The rest go to an
   "unsorted" bucket, which the report shows honestly instead of hiding.

Why not plain k-means? It needs the number of topics up front, and the usual
way to choose it (the silhouette score) kept preferring the largest k on our
benchmark. The threshold 0.2 was chosen on the Bitext benchmark: it sorts about
90% of tickets and, counting the unsorted ones as a topic, scores as well as
k-means that is *told* the true number of intents (see evaluate.py and the README).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans

UNSORTED = -1
ASSIGN_THRESHOLD = 0.2
DISCOVERY_SAMPLE = 8000


@dataclass
class Clustering:
    labels: np.ndarray  # topic id per ticket, UNSORTED (-1) for unassigned
    centroids: np.ndarray  # unit vectors, one per topic
    similarity: np.ndarray  # similarity of each ticket to its topic centre
    method: str


def _centroids(X: np.ndarray, labels: np.ndarray, ids: list[int]) -> np.ndarray:
    C = np.array([X[labels == j].mean(axis=0) for j in ids])
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)


def assign(X: np.ndarray, C: np.ndarray, threshold: float = ASSIGN_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    sim = X @ C.T
    best = sim.argmax(axis=1)
    score = sim[np.arange(len(X)), best]
    return np.where(score >= threshold, best, UNSORTED), score


def cluster(
    X: np.ndarray,
    k: int | None = None,
    seed: int = 0,
    min_topic_share: float = 1 / 300,
    threshold: float = ASSIGN_THRESHOLD,
) -> Clustering:
    n = len(X)
    if n < 3 or not np.any(X):
        # Too few tickets to group, or nothing left to compare after cleaning: one topic.
        C = X.mean(axis=0, keepdims=True) if n else X
        return Clustering(np.zeros(n, dtype=int), C, np.ones(n), "single")

    if k is not None:
        # Explicit number of topics: k-means, every ticket assigned.
        km = KMeans(n_clusters=max(1, min(k, n)), random_state=seed, n_init=4).fit(X)
        ids = sorted(set(km.labels_))
        C = _centroids(X, km.labels_, ids)
        sim = (X * C[km.labels_]).sum(axis=1)
        return Clustering(km.labels_.astype(int), C, sim, f"kmeans(k={k})")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(DISCOVERY_SAMPLE, n), replace=False)
    S = X[idx]
    min_size = max(5, int(len(S) * min_topic_share))
    if n < 2 * min_size:
        # Too few tickets for density-based discovery: a couple of k-means topics.
        return cluster(X, k=max(1, min(int(np.sqrt(n / 2)), n - 1)), seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        found = HDBSCAN(min_cluster_size=min_size, copy=True).fit_predict(S)
    ids = sorted(set(found) - {UNSORTED})

    if len(ids) < 2:
        # Too little structure for HDBSCAN (small or very uniform data): fall back to k-means.
        k = max(2, min(int(np.sqrt(n / 2)), n - 1))
        return cluster(X, k=k, seed=seed)

    C = _centroids(S, found, ids)
    labels, sim = assign(X, C, threshold)
    # Drop topics that ended up with too few tickets after assignment; renumber densely.
    counts = np.bincount(labels[labels >= 0], minlength=len(C))
    keep = [j for j in range(len(C)) if counts[j] >= max(3, min_size // 2)]
    if len(keep) < len(C):
        C = C[keep]
        labels, sim = assign(X, C, threshold)
    return Clustering(labels.astype(int), C, sim, f"hdbscan(min_size={min_size})+assign({threshold})")
