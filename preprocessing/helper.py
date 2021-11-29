import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, filename = 'path', drop_columns = 'list of columns or empty list', target = 'y_variable', test_size = 0.3, add_intercept = True, random_state = 7):
        self.filename = filename
        self.target = target
        self.drop_columns = drop_columns
        self.test_size = test_size
        self.add_intercept = add_intercept
        self.random_state = random_state

    def read_dataset(self, filename):
        return pd.read_csv(filename).drop(self.drop_columns, axis = 1)

    def split_data_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def add_intercept_to_data(self, X):
        return np.column_stack([np.ones([X.shape[0], 1]), X])

    def normalize_train(self, X):
        mean = np.mean(X, 0)
        std = np.std(X, 0)

        X_norm = (X - mean) / std
        if self.add_intercept:
            X_norm = self.add_intercept_to_data(X_norm)

        return X_norm, mean, std

    def normalize_test(self, X, train_mean, train_std):
        X_norm = (X - train_mean) / train_std
        
        if self.add_intercept:
            X_norm = self.add_intercept_to_data(X_norm)

        return X_norm

    def transform(self):
        df = self.read_dataset(self.filename)
        y = df[self.target]
        X = df.drop(self.target, axis = 1)
        self.y = np.array(y)
        self.X = np.array(X)
        
        #self.X = df[:, 0:-1]
        #self.y = df[:, -1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data_train_test(
        )
        self.X_train, self.mean, self.std = self.normalize_train(self.X_train)
        self.X_test = self.normalize_test(self.X_test, self.mean, self.std)
        return self.X_train, self.y_train, self.X_test, self.y_test
