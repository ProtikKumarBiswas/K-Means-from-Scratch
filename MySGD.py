from scipy.stats import norm
import numpy as np
from tqdm import tqdm

class newSGDRegressor:
    
    def __init__(self):
        W = None
        b = None
        
        
    def gradient(self, x, y, W, b):
        
        #The value of the gradients are calculated in this function. The equation of gradients are as follows:
        
        # d_L(W) / d_W = 1/n * sum(-2 * X_i * (y_i - (W . X_i + b)))
        # d_L(W) / d_b = 1/n * sum(-2 * (y_i - (W . X_i + b)))
        
        
        # The whole dataset is taken to find the gradient (d_L / d_w) as the given dataset is too small(about 500 data points).
        
        import random
        n_feat = len(x[0])
        n_point = len(x)
        sample_size = n_point
        sample = random.sample(range(n_point), sample_size)
        grad_W = np.zeros(13)
        grad_b = np.zeros(13)
        
        for i in sample:
            
            yhat = np.dot(W, x[i])
            yhat = yhat + b
            err = y[i] - yhat
            
            grad_W = grad_W + (-2 * x[i])* err
            grad_b = grad_b + (-2 * err)
            
        grad_W /= sample_size
        grad_b /= sample_size
        
        return grad_W , grad_b
    
    
    def fit(self, x, y, max_iter = 1000, eta = .1, tol = 0.001):     #n : No. of Iterations, eta : Learning Rate, tol : Tolarance
        
        n_feat = len(x[0])
        
        #Here the weights are initialized to random normal distribution with mean and std_dev of the given dataset
        mu, std = norm.fit(x)
        self.W = np.random.normal(mu, std, size = n_feat)
        print(mu , std)
        self.b = np.ones(n_feat)
        i = 0
        
        np.set_printoptions(precision=2)
        
        print('Iter {} :\nW = {}\nb = {}'.format(i, self.W, self.b))
        
        for i in tqdm(range(max_iter)):
            
            grad_W, grad_b = self.gradient(x, y, self.W, self.b)
            new_W = self.W - eta * grad_W
            new_b = self.b - eta * grad_b
            
            #The tolarance for checking the equality of New_W and Old_W is set to 1e-05
            if not np.allclose(new_W, self.W, atol = tol):
                self.W = new_W
                self.b = new_b
            else:
                break
            
            #if (i + 1) % 100 == 0:
            #   print(new_W, self.W)
            #    print('Iter {} :\nW = {}\nb = {}\neta : {}\n{}\n{}\n\n'.format(i + 1, self.W, self.b, eta, grad_W, grad_b))
    
    def predict(self, x):
        y = []
        for i in range(len(x)):
            y.append(np.average(np.dot(self.W, x[i]) + self.b))
        
        return y
            
    def get_coef(self):
        return self.W, self.b