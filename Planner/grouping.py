"""Step 3 - group related topics: sentence embeddings + clustering + TF-IDF keywords.

Topics are clustered *inside each unit*, so the study order of units is preserved
while related topics of a unit are studied together.
If `sentence-transformers` is not installed (or its model cannot be downloaded),
the code silently falls back to TF-IDF character n-grams.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import StudyGroup, Unit

SBERT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SBERT_MODEL)


def embed_topics(texts: Sequence[str], method: str = "auto") -> Tuple[np.ndarray, str]:
    """Return (embedding matrix, backend name). method: auto | sbert | tfidf."""
    texts = list(texts)
    if method in ("auto", "sbert"):
        try:
            model = _load_sbert()
            emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(emb), f"sentence-transformers ({SBERT_MODEL})"
        except Exception as exc:
            if method == "sbert":
                raise RuntimeError(f"sentence-transformers unavailable: {exc}") from exc
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    return vec.fit_transform(texts).toarray(), "TF-IDF character n-grams (fallback)"


def _cluster(emb: np.ndarray, group_size: int, seed: int) -> np.ndarray:
    n = len(emb)
    if n <= group_size:
        return np.zeros(n, dtype=int)
    k = max(1, int(round(n / group_size)))
    k = min(k, len(np.unique(np.round(emb, 6), axis=0)))      # can't have more clusters than distinct points
    if k <= 1:
        return np.zeros(n, dtype=int)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)


def _original_case(term: str, texts: Sequence[str]) -> str:
    for t in texts:
        m = re.search(rf"\b{re.escape(term)}\b", t, flags=re.IGNORECASE)
        if m:
            s = m.group(0)
            return s if s.isupper() else s.capitalize()
    return term.capitalize()


class _KeywordLabeler:
    """TF-IDF keyword extraction used to name each study group."""

    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
        try:
            self.vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
            self.X = self.vec.fit_transform(self.texts)
            self.terms = np.array(self.vec.get_feature_names_out())
        except ValueError:                                     # empty vocabulary
            self.vec = None

    def label(self, idx: List[int], top_n: int = 3) -> Optional[str]:
        if self.vec is None:
            return None
        scores = np.asarray(self.X[idx].sum(axis=0)).ravel()
        best = [i for i in scores.argsort()[::-1][:top_n] if scores[i] > 0]
        if not best:
            return None
        return " · ".join(_original_case(self.terms[i], self.texts) for i in best)


def build_groups(units: List[Unit], embeddings: np.ndarray, group_size: int = 4, seed: int = 42) -> List[StudyGroup]:
    flat = [t for u in units for t in u.topics]
    labeler = _KeywordLabeler([t.name for t in flat])
    groups: List[StudyGroup] = []
    offset = 0
    for unit in units:
        n = len(unit.topics)
        labels = _cluster(embeddings[offset:offset + n], group_size, seed)
        buckets: dict = {}
        for local_i, lab in enumerate(labels):
            buckets.setdefault(int(lab), []).append(local_i)
        for idx in sorted(buckets.values(), key=lambda ix: ix[0]):      # keep syllabus order
            topics = [unit.topics[i] for i in idx]
            name = labeler.label([offset + i for i in idx]) or topics[0].name
            gid = len(groups)
            for t in topics:
                t.group_id = gid
            groups.append(StudyGroup(unit.number, unit.title, name, topics))
        offset += n
    return groups
