import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        # initialise weights and biases
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias
            
            derivate_w = (1/n_samples) * np.dot(X.T, (y_pred-y))*2
            derivate_b = (1/n_samples) * np.sum(y_pred-y)*2
            self.weight = self.weight - self.lr *derivate_w
            self.bias = self.bias - self.lr*derivate_b
    
    def mse(self, y_test, predictions):
        return np.mean((y_test-predictions)**2)
    
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias
    
    def learning_rate_recommender(self, X_train, X_test, y_train, y_test, rate):
        best_mse = 1000000
        best_lr = -10
        temp = self.lr
        for _ in range(10):
            self.fit(X_train,y_train)
            predictions = self.predict(X_test)
            mse = self.mse(y_test, predictions)
            if mse<best_mse:
                best_mse = mse
                best_lr = self.lr
            self.lr+=rate
        self.lr = temp
        for _ in range(10):
            self.fit(X_train,y_train)
            predictions = self.predict(X_test)
            mse = self.mse(y_test, predictions)
            if mse<best_mse:
                best_mse = mse
                best_lr = self.lr
            self.lr-=rate
        return best_lr
    

        