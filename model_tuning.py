import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from model_complexity import complexity_model


# TODO: look at training perfo vs intensity (sport) dataset
# If not: generate outliers
# Generating fake dataset
np.random.seed(0)
x = np.random.rand(1000, 1) * 10 - 5
noise = np.random.randn(1000, 1) * 100
y = 1 + 2*x + 3*x**2 - 4*x**3 + noise
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8)


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


# xxx
class RegressionClassifier:
    def __init__(self, pipeline, name, X_train, Y_train, X_test, Y_test, k):
        self.pipeline = pipeline
        self.name = name
        pipeline.fit(X_train, Y_train)
        self.prediction = pipeline.predict(X_test)
        self.mse = mean_squared_error(Y_test, self.prediction)
        self.r2 = r2_score(Y_test, self.prediction)
        self.complexity = complexity_model(pipeline, X_test, Y_test, k)


# Creating pipelines
polynomes_pipelines = []
log_pipelines = []
exp_pipelines = []

for i in range(20):
    polynomial_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly_features', PolynomialFeatures(degree=i)),
        ('linear_regression', LinearRegression())
    ])
    polynomes_pipelines.append(polynomial_pipeline)

    log_pipeline = Pipeline([
        ('log_transform', LogTransformer()),
        ('poly_features', PolynomialFeatures(degree=i)),
        ('linear_regression', LinearRegression())
    ])
    log_pipelines.append(log_pipeline)

    exp_pipeline = Pipeline([
    ('exp_transform', ExpTransformer()),
    ('poly_features', PolynomialFeatures(degree=i)),
    ('linear_regression', LinearRegression())
])
    exp_pipelines.append(exp_pipeline)


# Creating pipeline objects
polynomes_classifiers = []
log_classifiers = []
exp_classifiers = []

for idx, (pipeline_polynome, pipeline_log, pipeline_exp) in enumerate(zip(polynomes_pipelines, log_pipelines, exp_pipelines)):
    polynomes_classifiers.append([RegressionClassifier(pipeline_polynome, f"{idx}", X_train, Y_train, X_test, Y_test, k) for k in range(5)])
    log_classifiers.append([RegressionClassifier(pipeline_log, f"{idx}", X_train, Y_train, X_test, Y_test, k) for k in range(1, 5)])
    exp_classifiers.append([RegressionClassifier(pipeline_exp, f"{idx}", X_train, Y_train, X_test, Y_test, k) for k in range(1, 5)])


# Plotting Complexity vs. k
complexities = [p.complexity for p in log_classifiers[4]]
indices = list(range(len(log_classifiers[4])))

plt.figure(figsize=(10, 6))
plt.plot(indices, complexities, marker='o')
plt.title('Polynomial Complexity vs Index')
plt.xlabel('Polynomial Object Index')
plt.ylabel('Complexity')
plt.xticks(indices)
plt.grid(True)
plt.show()


# Creating 2D subplots for each list of polynomial objects
# Complexity on the y-axis, names on the x-axis

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 3 * 3))

for polynomials, ax in zip([polynomes_classifiers, log_classifiers, exp_classifiers], axes.flatten()):
    # Extracting names and complexities
    names = [p[0].name for p in polynomials]
    complexities = [p[0].complexity for p in polynomials]

    # Plotting
    ax.plot(names, complexities, marker='o', linestyle='-', color=np.random.rand(3,))
    ax.set_xlabel('Regression Classifier degree')
    ax.set_ylabel('Complexity')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 3 * 3))

for polynomials, ax in zip([polynomes_classifiers, log_classifiers, exp_classifiers], axes.flatten()):
    # Extracting names and complexities
    names = [p[0].name for p in polynomials]
    complexities = [p[0].mse for p in polynomials]

    # Plotting
    ax.plot(names, complexities, marker='o', linestyle='-', color=np.random.rand(3,))
    ax.set_xlabel('Regression Classifier degree')
    ax.set_ylabel('MSE')

plt.tight_layout()
plt.show()

