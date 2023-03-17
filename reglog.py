import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogicalRegression():

    def __init__(self) -> None:
        self.thetas = np.zeros((0,1))
        pass

    def fit(self, X, Y, learning_rate= 0.0000001, epochs=1000, bias=True):
        n = int(len(X))
        y = np.resize(Y, (n,1))

        if bias:
            m = X.shape[1] + 1
            aux = np.ones((n, 1))
            X = np.concatenate((X, aux), axis=1)
        else:
            m = X.shape[1]
        thetas = np.zeros((m,1))

        #sigmoide truncado
        #np.clip

        errores = []
        iter_ = []

    
        for i in range(epochs):
            z = np.dot(X, thetas)
            z  = np.clip(z, -500, 500)
            Y_pred = 1 / (1 + np.exp(-(z)))
            Y_pred = np.clip(Y_pred, -0.9999999999999, 0.9999999999999)
            error = np.dot(X.T, (y - Y_pred))
            thetas = thetas - learning_rate * (-2/m) * error
            iter_.append(i)
            errores.append(self.cost_function(y, Y_pred))
        print(thetas)
        return (iter_, errores)

    def mean_error(self, actual, predicted):
        n = (len(actual))
        mse = 0
        for i in range(n):
            mse += (predicted[i] - actual[i])**2
        mse /= n
        return mse
    
    def cost_function(self, actual, predicted):
        n = (len(actual))
        cost = 0
        for i in range(n):
            cost =+ actual[i] * np.log(predicted[i]) - (( 1 - actual[i]) * np.log(1 - predicted[i]))
        cost /= n
        return cost
        

    