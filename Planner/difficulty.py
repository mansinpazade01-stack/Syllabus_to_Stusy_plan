"""Step 4 - estimate topic difficulty (Easy / Medium / Hard).

Three methods:
  heuristic  - topic length + technical keywords + term rarity (TF-IDF IDF), no training data
  classifier - TF-IDF + logistic regression trained on data/labeled_topics.csv
  hybrid     - average of both (default)

You can edit HARD_KEYS / EASY_KEYS or add rows to data/labeled_topics.csv to tune it.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .models import Topic

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_topics.csv"

EASY_MAX = 0.40      # score < 0.40  -> Easy
MEDIUM_MAX = 0.62    # score < 0.62  -> Medium, otherwise Hard

HARD_KEYS = (
    "derivation", "derive", "proof", "prove", "theorem", "optimi", "gradient", "backprop", "bayes",
    "probabilis", "stochastic", "convergence", "eigen", "singular value", "tensor", "transformer",
    "attention", "neural", "deep", "reinforcement", "markov", "regulariz", "kernel", "support vector",
    "dimensionality", "complexity", "np-", "np complete", "concurrency", "recovery", "serializab",
    "b+ tree", "indexing", "lagrang", "entropy", "inference", "generative", "adversarial",
    "convolution", "recurrent", "lstm", "dynamic programming", "approximation", "distributed",
    "cryptograph", "compiler", "parsing", "expectation maximi", "transaction",
)
EASY_KEYS = (
    "introduction", "intro ", "overview", "basics", "basic ", "fundamental", "history", "definition",
    "concept", "terminology", "application", "types of", "need of", "scope", "advantages",
    "disadvantages", "motivation", "example", "case study", "trends", "challenges", "ethic",
    "life cycle", "lifecycle", "features", "characteristics", "preliminar",
)
CLASS_SCORE = {"Easy": 0.1, "Medium": 0.5, "Hard": 0.9}


def score_to_label(score: float) -> str:
    return "Easy" if score < EASY_MAX else "Medium" if score < MEDIUM_MAX else "Hard"


# --------------------------------------------------------------------------- heuristic
def _heuristic_scores(names: List[str]) -> np.ndarray:
    low = [n.lower() for n in names]

    # rarity: topics made of rare terms are more specialised; terms shared by many topics are core/general
    rarity = np.zeros(len(names))
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        X = vec.fit_transform(low)
        idf = vec.idf_
        for i in range(len(names)):
            cols = X[i].indices
            rarity[i] = idf[cols].mean() if len(cols) else 0.0
        if rarity.max() > rarity.min():
            rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min())
        else:
            rarity[:] = 0.5
    except ValueError:
        rarity[:] = 0.5

    scores = np.zeros(len(names))
    for i, text in enumerate(low):
        words = len(re.findall(r"[a-z0-9+\-]+", text))
        length = min(1.0, max(0.0, (words - 1) / 8))
        tech = min(1.0, 0.5 * sum(k in text for k in HARD_KEYS))
        easy = min(1.0, 0.5 * sum(k in text for k in EASY_KEYS))
        scores[i] = 0.30 * length + 0.40 * tech + 0.30 * rarity[i] - 0.35 * easy + 0.15
    return np.clip(scores, 0.0, 1.0)


# --------------------------------------------------------------------------- classifier
@lru_cache(maxsize=1)
def _train_classifier():
    if not DATA_PATH.exists():
        return None
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = [(r["topic"], r["label"]) for r in csv.DictReader(f) if r.get("topic") and r.get("label")]
    if len({lab for _, lab in rows}) < 2:
        return None
    X, y = zip(*rows)
    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    model.fit(list(X), list(y))
    return model


def _classifier_scores(names: List[str]):
    model = _train_classifier()
    if model is None:
        return None
    proba = model.predict_proba(names)
    weights = np.array([CLASS_SCORE[c] for c in model.classes_])
    return proba @ weights


# --------------------------------------------------------------------------- public API
def estimate_difficulty(topics: List[Topic], method: str = "hybrid") -> None:
    """Fill `topic.score` and `topic.difficulty` in place."""
    if not topics:
        return
    names = [t.name for t in topics]
    heur = _heuristic_scores(names)
    clf = _classifier_scores(names) if method in ("classifier", "hybrid") else None

    if method == "heuristic" or clf is None:
        final = heur
    elif method == "classifier":
        final = clf
    else:
        final = 0.5 * heur + 0.5 * clf

    for t, s in zip(topics, final):
        t.score = round(float(s), 3)
        t.difficulty = score_to_label(float(s))
