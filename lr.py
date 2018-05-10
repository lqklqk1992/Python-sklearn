import scipy.optimize as opt  
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class logisticRegression:
    def __init__(self):
        """ your code here """
        self.theta=0

    def logistic_func(self,theta, x):
        return float(1) / (1 + np.exp(-x.dot(theta)))

    def cost_func(self,theta, x, y):
    	log_func_v = self.logistic_func(theta,x)
    	y = np.squeeze(y)
    	step1 = y * np.log(log_func_v)
    	step2 = (1-y) * np.log(1 - log_func_v)
    	final = -step1 - step2
    	return np.mean(final)

    def log_gradient(self,theta, x, y):
    	first_calc = self.logistic_func(theta, x) - np.squeeze(y)
    	final_calc = first_calc.T.dot(x)
    	return final_calc

    def grad_desc(self,theta_values, X, y, lr=.0001, converge_change=.0001):
    	#normalize
    	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    	#setup cost iter
    	cost_iter = []
    	cost = self.cost_func(theta_values, X, y)
    	cost_iter.append([0, cost])
    	change_cost = 1
    	i = 1
    	while(change_cost > converge_change):
        	old_cost = cost
        	theta_values = theta_values - (lr * self.log_gradient(theta_values, X, y))
        	cost = self.cost_func(theta_values, X, y)
        	cost_iter.append([i, cost])
        	change_cost = old_cost - cost
        	i+=1
    	return theta_values, np.array(cost_iter)

    def fit(self, X, y):
        """ your code here """
        X=X.astype(np.float64)
        theta=np.zeros(X.shape[1])
        result = self.grad_desc(theta, X, y)
        self.theta=result[0]
        return self


    def predict(self, X):
        """ your code here """
        X=X.astype(np.float64)
        #normalize
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        pred_prob = self.logistic_func(self.theta, X)
        y = np.where(pred_prob >= 0.5, 1, 0)
        return y


