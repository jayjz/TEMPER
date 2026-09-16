"""Small deterministic classifiers used before calibration experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MajorityBaseline:
    def fit(self, labels: NDArray[np.int64], *, n_classes: int) -> MajorityBaseline:
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1D array")
        if n_classes <= int(labels.max()):
            raise ValueError("n_classes must include every observed label")
        self.n_classes = n_classes
        self.majority_class = int(np.bincount(labels, minlength=n_classes).argmax())
        return self

    def predict_proba(self, texts: list[str]) -> NDArray[np.float64]:
        probabilities = np.zeros((len(texts), self.n_classes), dtype=np.float64)
        probabilities[:, self.majority_class] = 1.0
        return probabilities

    def predict(self, texts: list[str]) -> NDArray[np.int64]:
        return self.predict_proba(texts).argmax(axis=1).astype(np.int64)


class TfidfLogisticRegressionBaseline:
    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="word", lowercase=True, ngram_range=(1, 1), norm="l2"
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            random_state=random_state,
            solver="lbfgs",
        )

    def fit(self, texts: list[str], labels: NDArray[np.int64]) -> TfidfLogisticRegressionBaseline:
        if len(texts) != labels.size:
            raise ValueError("texts and labels must have matching rows")
        self.classifier.fit(self.vectorizer.fit_transform(texts), labels)
        return self

    def predict_proba(self, texts: list[str]) -> NDArray[np.float64]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform(texts))
        return np.asarray(probabilities, dtype=np.float64)

    def predict(self, texts: list[str]) -> NDArray[np.int64]:
        predictions = self.classifier.predict(self.vectorizer.transform(texts))
        return np.asarray(predictions, dtype=np.int64)
