from abc import ABC, abstractmethod
import json
from numpy import ndarray
import numpy as np


class FusionModel(ABC):
    @abstractmethod
    def load(self, config_path: str):
        pass

    @abstractmethod
    def predict(self, features: list) -> ndarray:
        pass


class LogisticRegression(FusionModel):
    def __init__(self, config_path: str):
        self._weights = None
        self._intercept = None
        self._feature_names = None
        self.load(config_path)

    def load(self, config_path: str):
        """Load model weights from JSON file."""
        with open(config_path, "r") as f:
            model_data = json.load(f)
            self._weights = np.array(model_data["weights"])
            self._intercept = model_data["intercept"]
            self._feature_names = model_data["features"]

    def predict(self, features: list[str]) -> ndarray:
        """Predict probabilities using logistic regression.

        Args:
            features: List of feature strings in format ["1:0.5", "2:0.7", ...]

        Returns:
            Array of prediction probabilities
        """
        # Convert feature strings to dense array
        assert self._weights is not None

        feature_values = np.zeros(len(self._weights))

        for feature in features:
            if ":" in feature:
                idx, value = feature.split(":")
                feature_values[int(idx) - 1] = float(value)

        # Apply logistic regression
        score = np.dot(feature_values, self._weights) + self._intercept
        probability = 1 / (1 + np.exp(-score))

        return probability

    def get_feature_importance(self) -> dict:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to their importance (absolute weight values)
        """
        assert (
            self._feature_names is not None and self._weights is not None
        ), "Model not initialized"
        return dict(zip(self._feature_names, np.abs(self._weights)))
