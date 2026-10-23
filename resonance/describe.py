"""Describing each topic so a product team can act on it: a name, keywords,
what kind of issue it is, how upset customers are, and representative tickets.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Words that appear in nearly every support message and say nothing about the topic.
SUPPORT_STOP = set(
    "help need want like know trying try tried please thanks thank hi hello hey just able way get got make really ".split()
    + "assist assistance support customer service question questions problem issue could would can will".split()
    # Leftovers of contractions ("don't" -> "don", "isn't" -> "isn") are not topic words.
    + "don doesn didn isn wasn aren weren won wouldn couldn shouldn haven hasn ll ve re im".split()
)
STOP = list(ENGLISH_STOP_WORDS | SUPPORT_STOP)


def keywords(texts_by_topic: list[list[str]], top: int = 6) -> list[list[str]]:
    """Words and two-word phrases that are common in one topic but not elsewhere.

    For each term: p_t = share of the topic's tickets containing it, p_all = share of
    all tickets containing it. score = p_t x log(p_t / p_all), a "lift" weighted by how
    common the term is in the topic. Terms in fewer than 8% of the topic's tickets are
    ignored, so one odd word repeated in a few tickets can't name a topic.
    """
    all_texts = [t for ts in texts_by_topic for t in ts]
    if not all_texts:
        return [[] for _ in texts_by_topic]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=STOP, binary=True, min_df=2, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    try:
        M = cv.fit_transform(all_texts)
    except ValueError:
        return [[] for _ in texts_by_topic]
    vocab = np.array(cv.get_feature_names_out())
    p_all = np.asarray(M.mean(axis=0)).ravel()
    out, start = [], 0
    for ts in texts_by_topic:
        n = len(ts)
        if n == 0:
            out.append([])
            continue
        p_t = np.asarray(M[start : start + n].mean(axis=0)).ravel()
        start += n
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where((p_t >= 0.08) & (p_all > 0), p_t * np.log(p_t / p_all), 0.0)
        words: list[str] = []
        for i in np.argsort(-score):
            if score[i] <= 0 or len(words) >= top:
                break
            w = vocab[i]
            if any(w in x.split() or x in w.split() for x in words):
                continue
            words.append(w)
        out.append(words)
    return out


def name_topic(words: list[str], examples: list[str]) -> str:
    """A short readable name: the top keywords, e.g. "Track refund · status"."""
    if not words:
        return (examples[0][:48] + "…") if examples and len(examples[0]) > 48 else (examples[0] if examples else "Misc")
    parts = words[:2]
    if len(parts) == 2 and (set(parts[0].split()) & set(parts[1].split())):
        parts = parts[:1]
    name = " · ".join(parts)
    return name[0].upper() + name[1:]


# Issue types. Patterns are deliberately simple and readable, so a reviewer can
# see why a ticket got its type; each ticket gets the first type that matches.
QUESTION_START = re.compile(r"^\s*(how|where|what|which|when|why|is it possible|can i|could i|do you|does)\b", re.I)

TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("bug", re.compile(r"\b(not working|doesn'?t work|isn'?t working|won'?t (load|open|work)|broken|error|crash\w*|bug|fail\w*|stuck|can'?t|cannot|unable|wrong|missing|declined|problem with|issue with|glitch)\b")),
    ("complaint", re.compile(r"\b(complain\w*|terrible|awful|horrible|worst|disappointed|unacceptable|ridiculous|angry|furious|rude|scam|frustrat\w*|fed up|poor service)\b")),
    ("request", re.compile(r"\b(would like|i'?d like|wish|feature|add (a|an|the)?|option to|could you (add|make)|suggest\w*|improve\w*|cancel\w*|change|update|edit|delete|remove|subscribe|unsubscribe|switch)\b")),
    ("how-to", re.compile(r"\b(how (do|can|to|would)|where (is|can|do)|what (is|are|do)|which|when (will|does|can)|is it possible|can i|do you|help me|information|info|check|see|know|find)\b")),
]
TYPES = ["bug", "complaint", "request", "how-to", "other"]


def issue_type(text: str) -> str:
    t = text.lower()
    # Bugs and complaints win even when phrased as a question; otherwise a question is a how-to.
    for name, pat in TYPE_PATTERNS[:2]:
        if pat.search(t):
            return name
    if QUESTION_START.search(t):
        return "how-to"
    for name, pat in TYPE_PATTERNS[2:]:
        if pat.search(t):
            return name
    return "other"


URGENT = re.compile(r"\b(urgent\w*|asap|immediately|right now|emergency|critical|still (not|haven'?t|waiting)|again|days?|weeks?|third time|second time)\b", re.I)
ANGRY = re.compile(r"\b(terrible|awful|horrible|worst|disappointed|unacceptable|ridiculous|angry|furious|rude|scam|frustrat\w*|fed up|useless|hate|lawyer|refund now)\b", re.I)


def severity(text: str) -> float:
    """0..1: how urgent or upset a message reads. Counts urgency and anger words, '!!' and SHOUTING."""
    score = 0.0
    score += 0.35 * min(2, len(URGENT.findall(text)))
    score += 0.45 * min(2, len(ANGRY.findall(text)))
    if "!!" in text or text.count("!") >= 3:
        score += 0.2
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        score += 0.3
    return min(1.0, score / 1.2)


def type_counts(texts: list[str]) -> dict[str, int]:
    c = Counter(issue_type(t) for t in texts)
    return {t: c.get(t, 0) for t in TYPES}


def representatives(X: np.ndarray, centroid: np.ndarray, idx: np.ndarray, n: int = 5) -> list[int]:
    """The tickets closest to the topic centre, skipping near-duplicates of each other."""
    if len(idx) == 0:
        return []
    sims = X[idx] @ centroid
    chosen: list[int] = []
    for j in idx[np.argsort(-sims)]:
        if all(float(X[j] @ X[c]) < 0.95 for c in chosen):
            chosen.append(int(j))
        if len(chosen) >= n:
            break
    return chosen
