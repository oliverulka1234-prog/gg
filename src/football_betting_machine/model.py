from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


class OutcomeModel:
    def __init__(self, estimator: str = "rf", random_state: int = 42):
        self.estimator_name = estimator
        self.random_state = random_state
        self.model = self._build()

    def _build(self):
        if self.estimator_name == "xgb":
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=self.random_state,
                )
            except Exception:
                pass

        return RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=self.random_state,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return list(self.model.classes_)

    def evaluate(self, X, y) -> dict[str, float]:
        pred = self.model.predict(X)
        p = self.predict_proba(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "log_loss": float(log_loss(y, p, labels=self.classes_)),
            "avg_confidence": float(np.mean(np.max(p, axis=1))),
        }
