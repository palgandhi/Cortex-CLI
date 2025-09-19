import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from ..base import BaseModel

class LightGBMRegressorModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "LightGBM Regressor"
        self.model = lgb.LGBMRegressor(**self.hyperparameters)
        self.param_grid = {
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'learning_rate': [0.1, 0.05]
        }

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {"MSE": mse, "R-squared": r2}