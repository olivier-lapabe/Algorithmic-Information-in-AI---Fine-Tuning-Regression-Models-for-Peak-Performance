import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from src.Complexity.model_complexity import calculate_model_complexity


# -----------------------------------------------------------------------------
# Custom transformers
# -----------------------------------------------------------------------------
class ExpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.exp(X)
    

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(np.where(X <= 0, np.min(X[X > 0]), X))


# -----------------------------------------------------------------------------
# Regression Classifier
# -----------------------------------------------------------------------------
class RegressionClassifier:
    def __init__(self, pipeline, name, degree):
        self.pipeline = pipeline
        self.name = name
        self.degree = degree
        self.prediction = None
        self.mse = None
        self.r2 = None
        self.complexity = None

    def fit_predict(self, X_train, Y_train, X_test, Y_test):
        self.pipeline.fit(X_train, Y_train)
        self.prediction = self.pipeline.predict(X_test)
        self.mse = mean_squared_error(Y_test, self.prediction)
        self.r2 = r2_score(Y_test, self.prediction)
        self.complexity = calculate_model_complexity(self.pipeline, X_test, Y_test, self.degree)


# -----------------------------------------------------------------------------
# create_pipeline
# -----------------------------------------------------------------------------
def create_pipeline(degree, transformer=None):
    steps = [('scaler', StandardScaler())]

    if transformer:
        steps.append(('transformer', transformer()))

    steps.extend([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])
    return Pipeline(steps)


# -----------------------------------------------------------------------------
# plot_data
# -----------------------------------------------------------------------------
def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.title('Synthetic Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# -----------------------------------------------------------------------------
# plot_complexities
# -----------------------------------------------------------------------------
def plot_complexities(classifiers_groups, title, ylabel):
    fig, axes = plt.subplots(nrows=1, ncols=len(classifiers_groups), figsize=(5*len(classifiers_groups), 5))

    for classifiers, ax in zip(classifiers_groups, axes.flatten()):
        # For each first element in each sublist, extract names & complexity
        # We use the hypothesis that for each classifier the element with the lowest degree is the best
        names = [p[0].name for p in classifiers]
        complexities = [getattr(p[0], ylabel) for p in classifiers]

        # Plotting
        ax.plot(names[1:], complexities[1:], marker='o', linestyle='-', color='blue')
        ax.set_xlabel(f'Regression Classifier Degree')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(names[1:], rotation=45)

    plt.tight_layout()
    plt.show()


class ModelRunner:
    def __init__(self, X, y, train_size=0.1, max_degree=10):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, train_size=train_size)
        self.max_degree = max_degree
        self.polynomes_classifiers = []
        self.log_classifiers = []
        self.exp_classifiers = []

    def run(self):
        for i in range(self.max_degree):
            self.polynomes_classifiers.append([RegressionClassifier(create_pipeline(i), f"Poly-{i}", deg)
                                               for deg in range(5)])
            self.exp_classifiers.append([RegressionClassifier(create_pipeline(i, ExpTransformer), f"Exp-{i}", deg)
                                         for deg in range(5)])
            self.log_classifiers.append([RegressionClassifier(create_pipeline(i, LogTransformer), f"Log-{i}", deg)
                                         for deg in range(5)])

        for classifier_list in [self.polynomes_classifiers, self.log_classifiers, self.exp_classifiers]:
            for classifiers in classifier_list:
                for classifier in classifiers:
                    classifier.fit_predict(self.X_train, self.Y_train, self.X_test, self.Y_test)

        plot_complexities([self.polynomes_classifiers, self.exp_classifiers, self.log_classifiers], 'Model Complexity vs Degree',
                          'complexity')
        plot_complexities([self.polynomes_classifiers, self.exp_classifiers, self.log_classifiers], 'Model MSE vs Degree', 'mse')
