from abc import ABC, abstractmethod
import json
import numpy as np
from numpy import ndarray
from typing import Dict


class DenserFusionModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, features: list) -> ndarray:
        pass


class LogisticRegression(DenserFusionModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weights = None
        self.intercept = None
        self.feature_names = None
        self.load()

    def load(self):
        """Load model weights from JSON file."""
        with open(self.model_path, "r") as f:
            model_data = json.load(f)
            self.weights = np.array(model_data["weights"])
            self.intercept = model_data["intercept"]
            self.feature_names = model_data["features"]

    def predict(self, features: list) -> ndarray:
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


if __name__ == "__main__":
    # Test LogisticRegression class singleton pattern
    model1 = LogisticRegression(
        "/home/ubuntu/denser-retriever/exps/exp_scifact/models/weights_es+vs+rr.json"
    )
    model2 = LogisticRegression(
        "/home/ubuntu/denser-retriever/exps/exp_scifact/models/weights_es+vs+rr.json"
    )
    print(f"Same instances: {model1 is model2}")  # Should print True

    features = ["1:0.5", "2:0.7", "3:0.2", "4:0.9"]
    prediction = model1.predict(features)
    print(f"Prediction probability: {prediction[0]:.4f}")
    print("Feature importance:")
    for feature, importance in model1.get_feature_importance().items():
        print(f"{feature}: {importance:.4f}")
