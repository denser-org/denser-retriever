from abc import ABC, abstractmethod
from numpy import ndarray
import xgboost as xgb
from scipy.sparse import csr_matrix


class DenserGradientBoost(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, csr_data: csr_matrix) -> ndarray:
        pass


class XGradientBoost(DenserGradientBoost):
    def __init__(
        self,
        xgb_model_path: str,
    ):
        self.xgb_model_path = xgb_model_path
        self.load()

    def load(self):
        self.model = xgb.Booster()
        self.model.load_model(self.xgb_model_path)

    def predict(self, csr_data: csr_matrix):
        return self.model.predict(xgb.DMatrix(csr_data))
