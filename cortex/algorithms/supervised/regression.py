from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ..base import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Linear Regression"
        self.model = LinearRegression(**self.hyperparameters)
        self.param_grid = {} 
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {"MSE": mse, "R-squared": r2}