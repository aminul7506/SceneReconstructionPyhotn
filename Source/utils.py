import numpy as np


def coefficients_of_polynomial(x_values_array, y_values_array, degree):
    return np.polyfit(x_values_array, y_values_array, degree)


def get_value_from_four_degree_polynomial(polynomial_coefficients, x):
    a = polynomial_coefficients[0]
    b = polynomial_coefficients[1]
    c = polynomial_coefficients[2]
    d = polynomial_coefficients[3]
    e = polynomial_coefficients[4]

    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
