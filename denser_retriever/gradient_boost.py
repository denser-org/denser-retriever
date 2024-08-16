import xgboost as xgb
from scipy.sparse import csr_matrix


class DenserGradientBoost:
    def __init__(
        self,
        xgb_model_path: str,
    ):
        self.xgb_model_path = xgb_model_path
        self.load()

    def load(self):
        xgb_model = xgb.Booster()
        xgb_model.load_model(self.xgb_model_path)
        self.xgb_model = xgb_model

    def predict(self, csr_data: csr_matrix):
        test_data = xgb.DMatrix(csr_data)
        return self.xgb_model.predict(test_data)
