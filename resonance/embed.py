"""Turning ticket text into vectors, so that tickets with similar meaning end up close together.

The default needs no download and no API: TF-IDF over words and character
n-grams, reduced with LSA (truncated SVD). Character n-grams make it robust to
the typos real customers make ("oorder", "puchase"); LSA merges words that
appear in similar contexts ("refund" and "money back").

Optional, higher-quality backends:
  * "st": a sentence-transformers model, if that package is installed
  * "openai": an OpenAI-compatible embeddings API
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def tfidf_lsa(texts: list[str], dims: int = 128, seed: int = 0) -> np.ndarray:
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english")
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.5, sublinear_tf=True)
    try:
        X = hstack([words.fit_transform(texts), 0.5 * chars.fit_transform(texts)]).tocsr()
    except ValueError:
        # Tiny inputs can leave no term above min_df: fall back to no thresholds.
        words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X = words.fit_transform(texts)
    k = max(2, min(dims, X.shape[1] - 1, X.shape[0] - 1))
    if k < 2:
        return normalize(X.toarray())
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    return normalize(Z)


def sentence_transformers(texts: list[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError("the 'st' backend needs: pip install sentence-transformers") from e
    m = SentenceTransformer(model)
    return np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True))


def openai_embeddings(texts: list[str], model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> np.ndarray:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("the 'openai' backend needs OPENAI_API_KEY")
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = model or os.environ.get("RESONANCE_EMBED_MODEL") or "text-embedding-3-small"
    out: list[list[float]] = []
    for i in range(0, len(texts), 256):
        req = urllib.request.Request(
            f"{base}/embeddings",
            data=json.dumps({"model": model, "input": texts[i : i + 256]}).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            data = json.loads(res.read())
        out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
    return normalize(np.asarray(out, dtype=np.float32))


def embed(texts: list[str], backend: str = "tfidf", seed: int = 0) -> np.ndarray:
    if backend == "tfidf":
        return tfidf_lsa(texts, seed=seed)
    if backend == "st":
        return sentence_transformers(texts)
    if backend == "openai":
        return openai_embeddings(texts)
    raise ValueError(f"unknown embedding backend {backend!r} (use tfidf, st or openai)")
