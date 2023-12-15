import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def complexity_integer(n: int) -> int:
    """Calculates the complexity of an integer n"""
    complexity_integer = 1 + np.ceil(np.log2(abs(n) + 1))
    return int(complexity_integer)


def complexity_intarray(n: np.array) -> int:
    """Calculates the complexity of an integer array n"""
    complexity_intarray = np.sum(np.vectorize(complexity_integer)(n))
    return int(complexity_intarray)


def extract_digits(n: float, k: int) -> np.array:
    """Extracts each digit of a float n up to k places after the decimal point"""
    if k == 0:
        n = round(abs(n))
    else:
        n = round(abs(n), k)
    str_n = str(n).replace('.', '')
    return np.array([int(char) for char in str_n])


def complexity_float(n: float, k: int) -> int:
    """Calculates the complexity of a float n up to k places after the decimal point"""
    digits = extract_digits(n, k)
    return complexity_intarray(digits)


def complexity_floatarray(n: np.array, k: int) -> int:
    """Calculates the complexity of a float array n up to k places after the decimal point"""
    complexity_floatarray = np.sum(np.vectorize(complexity_float)(n, k))
    return complexity_floatarray


def complexity_model(F: Pipeline, X: np.array, Y: np.array, k: int) -> int:
    """Calculates the complexity of a trained regression model F"""
    # Calculate the complexity of the model parameters
    parameters = np.insert(F.named_steps['linear_regression'].coef_[0][1:], 0, F.named_steps['linear_regression'].intercept_[0])
    complexity_parameters = complexity_floatarray(parameters, k)

    # Calculate the complexity of the data given the model
    Euclidean_distance = abs(Y - F.predict(X))
    complexity_data_given_model = complexity_floatarray(Euclidean_distance, k)

    # Calculate the complexity of the model
    complexity_model = complexity_parameters + complexity_data_given_model
    return complexity_model


# #Testing

# X_2 = np.array([0,1])
# Y_2 = np.array([0,2])
# X_3 = np.array([0,1])
# Y_3 = np.array([0,3])
# X_test = np.array([1, 2, 3, 4, 5, 6])
# Y_test = np.array([2, 5, 9, 10, 11, 17])

# F2 = LinearRegression()
# F2.fit(X_2.reshape(-1, 1), Y_2)
# # print(F2.coef_)
# # print(F2.intercept_)
# # print(complexity_model(F2, X_test.reshape(-1, 1), Y_test, 0))

# F3 = LinearRegression()
# F3.fit(X_3.reshape(-1, 1), Y_3)
# # print(F3.coef_)
# # print(F3.intercept_)
# print(complexity_model(F3, X_test.reshape(-1, 1), Y_test, 0))

# X_4 = np.array([0.5437328,1.528932])
# Y_4 = np.array([0.5273329,-3.65729])
# X_test_f = np.array([1.1543, 2.54322, 3.53, 4.654235, 5.431, 6.86])
# Y_test_f = np.array([2.5, 5.2532, 9.9, 10.15, 11.632, 17.45325323])

# F4 = LinearRegression()
# F4.fit(X_4.reshape(-1, 1), Y_4)
# # print(F4.coef_)
# # print(F4.intercept_)
# # print(complexity_model(F4, X_test_f.reshape(-1, 1), Y_test_f, 7))