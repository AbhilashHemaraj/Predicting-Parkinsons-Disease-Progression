import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('fivethirtyeight')
class LinearRegression:
    def __init__(self,
                 learning_rate,
                 lambda_value,
                 regularization,
                 max_iter,
                 tolerance,
                 method='ols, gd or sgd',
                 set_seed=7):
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.regularization = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.method = method
        self.set_seed = set_seed

    def add_intercept(self, X):
        return np.column_stack([np.ones([X.shape[0], 1]), X])

    def check_matrix_assumptions(self, X):
        x_rank = np.linalg.matrix_rank(X)

        if x_rank == min(X.shape[0], X.shape[1]):
            print('Matrix is full rank')
            self.full_rank = True
        else:
            print('Matrix is not full rank')
            self.full_rank = False

        if X.shape[0] < X.shape[1]:
            self.low_rank = True
            print('Data is low rank')
        else:
            self.low_rank = False
            print('Data is not low rank')

    def ols(self, X, y):
        if not self.regularization:
            w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            return w
        else:
            n = X.shape[1]
            I = np.eye((n))
            w = np.linalg.inv(X.T.dot(X) + self.lambda_value * I).dot(
                X.T).dot(y)
            return w

    def gradient_descent(self, X, y):
        self.error_history = []
        last_error = np.inf
        for i in tqdm(range(self.max_iter)):
            self.w = self.w - self.learning_rate * self.cost_derivative(X, y)
            current_error = self.rmse(X, y)
            diff_error = last_error - current_error
            last_error = current_error

            self.error_history.append(current_error)
            if diff_error < self.tolerance:
                print("The model has converged")
                break

    def stochastic_gradient_descent(self, X, y):
        self.error_history = []
        last_error = np.inf
        for i in tqdm(range(self.max_iter)):
            index = np.random.choice(X.shape[0],
                                     int(0.2 * X.shape[0]),
                                     replace=False)
            self.w = self.w - self.learning_rate * self.cost_derivative(
                X[index], y[index])
            current_error = self.rmse(X[index], y[index])
            diff_error = last_error - current_error
            last_error = current_error

            self.error_history.append(current_error)
            if abs(diff_error) < self.tolerance:
                print("The model has converged")
                break

    def predict(self, X):
        return X.dot(self.w)

    def sse(self, X, y):
        y_predicted = self.predict(X)
        #print(y_predicted)
        return ((y_predicted - y)**2).sum()

    def rmse(self, X, y):
        return np.sqrt((self.sse(X, y)) / X.shape[0])

    def score(self, X, y):
        print('Metrics with l2 Regularization:' if self.
              regularization else 'Metrics without Regularization')
        return print('RMSE test data:', self.rmse(X, y), '\n',
                     'SSE test data:', self.sse(X, y))

    def cost_derivative(self, X, y):
        y_predicted = self.predict(X)
        if not self.regularization:
            return (y_predicted - y).dot(X)
        else:
            return (y_predicted - y).dot(X) + (self.lambda_value * self.w)

    def plot_error_history(self, X, y):
        plt.figure(figsize=(12, 7))
        plt.plot(self.error_history)
        plt.title(r"Cost vs Epochs - {0} {1}".format(*[
            'Gradient Descent' if self.method ==
            'gd' else 'Stochastic Gradient Descent', 'with l2 Regularization'
            if self.regularization else 'without Regularization'
        ]))
        plt.xlabel("Epochs")
        plt.ylabel("Sum of Squared Errors")
        plt.text(x=int(len(self.error_history)*0.6),
                 y=np.max(self.error_history) * 0.7,
                 s=r'RMSE={0:.3f}'.format(self.rmse(X, y)),
                 fontsize=20,
                 color=[0, 158 / 255, 115 / 255],
                 weight='bold',
                 rotation=0,
                 backgroundcolor='#f0f0f0')
        #plt.show()

    def fit(self, X, y):
        # Setting the random seed for reproducability
        np.random.seed(self.set_seed)

        # check for assumptions for closed form solution
        self.check_matrix_assumptions(X)
        
        if self.full_rank and not self.low_rank and X.shape[
                0] < 10000 and self.method == 'ols':
            print(
                "Solving using closed form solution/ Ordinary Least Squares method",
                'with l2 Regularization'
                if self.regularization else 'without Regularization')
            self.w = self.ols(X, y)
            print('RMSE:', self.rmse(X, y))
            print('SSE:', self.sse(X, y))

        elif self.method == 'gd':
            print(
                'Solving using Batch Gradient Descent',
                'with l2 Regularization'
                if self.regularization else 'without Regularization')
            self.w = np.zeros(X.shape[1], dtype=np.float64)
            self.gradient_descent(X, y)
            print('RMSE:', self.rmse(X, y))
            print('SSE:', self.sse(X, y))
            self.plot_error_history(X, y)

        elif self.method == 'sgd':
            print(
                'Solving using Stochastic Gradient Descent',
                'with l2 Regularization'
                if self.regularization else 'without Regularization')
            #self.w = np.random.rand(X.shape[1], 1)
            self.w = np.zeros(X.shape[1], dtype=np.float64)
            self.stochastic_gradient_descent(X, y)
            print('RMSE:', self.rmse(X, y))
            print('SSE:', self.sse(X, y))
            self.plot_error_history(X, y)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        covariance_matrix = np.cov(X, rowvar = 0)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        sorted_indexes = np.argsort(-eigen_values)
        eigen_values = eigen_values[sorted_indexes]
        eigen_vectors = eigen_vectors.T[sorted_indexes]

        # Choose principal components
        self.principal_components = eigen_vectors[:self.n_components,:]

        # Explained variance
        self.explained_variance = [(i/np.sum(eigen_values))*100 for i in eigen_values[:self.n_components]]

        # Cumulative explained variance
        self.total_explained_variance = np.cumsum(self.explained_variance)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.principal_components.T)
