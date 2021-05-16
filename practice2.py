import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import sklearn
import pandas as pd
from pydataset import data
from sklearn.model_selection import train_test_split
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list(decoded_list))

input = 'D:/ProgramData/practika2/linear.txt'
input_data = np.loadtxt(input, delimiter=',')
X, y = input_data[:, :-1], input_data[:, -1]
training_samples = int(0.6 * len(X))
testing_samples = len(X) - 10
X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]
reg_linear = sklearn.linear_model.LinearRegression()
reg_linear.fit(X_train, y_train)
y_test_pred = reg_linear.predict(X_test)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_pred, color='black', linewidth=2)
plt.xticks()
plt.yticks()
plt.show()
print("Performance of Linear regressor:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median abs error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain var scr =", round(sm.explained_variance_score(y_test, y_test_pred),2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

input = 'D:/ProgramData/praktika2/Mul_linear.txt'
input_data = np.loadtxt(input, delimiter=',')
X, y = input_data[:, :-1], input_data[:, -1]
training_samples = int(0.6 * len(X))
testing_samples = len(X) - 10
X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]
reg_linear_mul = linear_model.LinearRegression()
reg_linear_mul.fit(X_train, y_train)
y_test_pred = reg_linear_mul.predict(X_test)
print("Performance of Linear regressor:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median abs error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain var scr=", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[2.23, 1.35, 1.12]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression:\n", reg_linear_mul.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))