import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from model_complexity import calculate_model_complexity


# -----------------------------------------------------------------------------
# Custom transformers
# -----------------------------------------------------------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(np.where(X <= 0, np.min(X[X > 0]), X))


class ExpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.exp(X / np.max(X))


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
    steps = []
    if transformer:
        steps.append(('transformer', transformer()))
    steps.extend([
        ('scaler', StandardScaler()),
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])
    return Pipeline(steps)


# -----------------------------------------------------------------------------
# generate_synthetic_data
# -----------------------------------------------------------------------------
def generate_synthetic_data():
    np.random.seed(0)
    x = np.random.rand(1000, 1) * 10 - 5
    noise = np.random.randn(1000, 1) * 100
    y = 1 + 2*x + 3*x**2 - 4*x**3 + noise
    return x, y


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
    fig, axes = plt.subplots(nrows=len(classifiers_groups), ncols=1, figsize=(15, 5 * len(classifiers_groups)))

    for classifiers, ax in zip(classifiers_groups, axes.flatten()):
        # For each first element in each sublist, extract names & complexity
        # We use the hypothesis that for each classifier the element with the lowest degree is the best
        names = [p[0].name for p in classifiers]
        complexities = [getattr(p[0], ylabel) for p in classifiers]

        # Plotting
        ax.plot(names[1:], complexities[1:], marker='o', linestyle='-', color='blue')
        ax.set_xlabel('Regression Classifier Degree')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def main():
    x, y = generate_synthetic_data()
    plot_data(x, y)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.1)
    polynomes_classifiers, log_classifiers, exp_classifiers = [], [], []

    for i in range(20):
        polynomes_classifiers.append([RegressionClassifier(create_pipeline(i), f"Poly-{i}", deg)
                                      for deg in range(5)])
        log_classifiers.append([RegressionClassifier(create_pipeline(i, LogTransformer), f"Log-{i}", deg)
                                for deg in range(1, 5)])
        exp_classifiers.append([RegressionClassifier(create_pipeline(i, ExpTransformer), f"Exp-{i}", deg)
                                for deg in range(1, 5)])

    for classifier_list in [polynomes_classifiers, log_classifiers, exp_classifiers]:
        for classifiers in classifier_list:
            for classifier in classifiers:
                classifier.fit_predict(X_train, Y_train, X_test, Y_test)

    plot_complexities([polynomes_classifiers, log_classifiers, exp_classifiers], 'Model Complexity vs Degree',
                      'complexity')
    plot_complexities([polynomes_classifiers, log_classifiers, exp_classifiers], 'Model MSE vs Degree', 'mse')


    ### Ajouté par Goatstat
    parameters = {}
    for i in range(20):
        parameters[i] = np.insert(polynomes_classifiers[i][0].pipeline.named_steps['linear_regression'].coef_[0][1:], 0, polynomes_classifiers[i][0].pipeline.named_steps['linear_regression'].intercept_[0])

    x = StandardScaler().fit_transform(x)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points', s=10)
    plt.scatter(x, parameters[0][0] * np.ones(x.shape[0]), color=(0.56, 0.93, 0.56), s=5)
    plt.scatter(x, parameters[1][0] + parameters[1][1] * x, color=(0.56, 0.93, 0.56), s=5)
    plt.scatter(x, parameters[2][0] + parameters[2][1] * x + parameters[2][2] * x**2, color=(0.56, 0.93, 0.56), s=5)
    plt.scatter(x, parameters[3][0] + parameters[3][1] * x + parameters[3][2] * x**2 + parameters[3][3] * x**3, color=(0.56, 0.93, 0.56), s=5)
    plt.scatter(x, parameters[4][0] + parameters[4][1] * x + parameters[4][2] * x**2 + parameters[4][3] * x**3 + parameters[4][4] * x**4, color=(0.56, 0.93, 0.56), s=5)
    plt.show()


if __name__ == "__main__":
    main()
