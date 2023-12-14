import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: look at training perfo vs intensity (sport) dataset
# Generating fake dataset
np.random.seed(0)
x = np.random.rand(100, 1) * 10
noise = np.random.randn(100, 1)
y = 1 + 2*x + 3*x**2 - 4*x**3 + noise

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.title('Synthetic Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Logarithmic transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(np.where(X <= 0, np.min(X[X > 0]), X))


# Exponential transformer
class ExpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.exp(X / np.max(X))


# Creating pipelines
polynomial_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

log_pipeline = Pipeline([
    ('log_transform', LogTransformer()),
    ('linear_regression', LinearRegression())
])

exp_pipeline = Pipeline([
    ('exp_transform', ExpTransformer()),
    ('linear_regression', LinearRegression())
])

# Applying pipelines to the dataset
polynomial_pipeline.fit(x, y)
log_pipeline.fit(x, y)
exp_pipeline.fit(x, y)

# Predictions
y_pred_poly = polynomial_pipeline.predict(x)
y_pred_log = log_pipeline.predict(x)
y_pred_exp = exp_pipeline.predict(x)

# Performance Evaluation
# Polynomial
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# Logarithmic
mse_log = mean_squared_error(y, y_pred_log)
r2_log = r2_score(y, y_pred_log)

# Exponential
mse_exp = mean_squared_error(y, y_pred_exp)
r2_exp = r2_score(y, y_pred_exp)


# Performance metrics
print("Model Performance Metrics:\n")

models = ["Polynomial", "Logarithmic", "Exponential"]
mses = [mse_poly, mse_log, mse_exp]
r2s = [r2_poly, r2_log, r2_exp]

for model, mse, r2 in zip(models, mses, r2s):
    print(f"{model} Model:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R-squared: {r2:.4f}\n")


