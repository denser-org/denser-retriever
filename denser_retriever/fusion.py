from abc import ABC, abstractmethod
import json
from numpy import ndarray
from importlib import resources
import numpy as np


class FusionModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, features: list) -> ndarray:
        pass


class LogisticRegression(FusionModel):
    def __init__(self, model_name: str):
        self.weights = None
        self.intercept = None
        self.feature_names = None
        self.load(model_name)

    def load(self, model_name: str):
        """Load model weights from JSON file."""
        with resources.open_text("denser_retriever.models", model_name) as f:
            model_data = json.load(f)
            self.weights = np.array(model_data["weights"])
            self.intercept = model_data["intercept"]
            self.feature_names = model_data["features"]

    def predict(self, features: list[str]) -> ndarray:
        """Predict probabilities using logistic regression.

        Args:
            features: List of feature strings in format ["1:0.5", "2:0.7", ...]

        Returns:
            Array of prediction probabilities
        """
        # Convert feature strings to dense array
        assert self.weights is not None

        feature_values = np.zeros(len(self.weights))

        for feature in features:
            if ":" in feature:
                idx, value = feature.split(":")
                feature_values[int(idx) - 1] = float(value)

        # Apply logistic regression
        score = np.dot(feature_values, self.weights) + self.intercept
        probability = 1 / (1 + np.exp(-score))

        return probability

    def get_feature_importance(self) -> dict:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to their importance (absolute weight values)
        """
        assert (
            self.feature_names is not None and self.weights is not None
        ), "Model not initialized"
        return dict(zip(self.feature_names, np.abs(self.weights)))
