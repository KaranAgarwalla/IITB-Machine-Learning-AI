import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt


class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)

        # TODO
        count = 0
        while count < self.iterations:
        	grad = kernel@y[:, np.newaxis] - self.lamda*kernel@self.alpha - kernel@(1/(1+np.exp(-kernel@self.alpha)))
        	self.alpha = self.alpha + self.eta*grad
        	count += 1
        # END TODO
    

    def predict(self, X):
        # TODO 
        kernel = self.kernel(X, self.train_X)
        return np.squeeze((1/(1+np.exp(-kernel@self.alpha))))
        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO
    error = 0 
    len = int(X.shape[0]/k)
    for i in range(k):
    	train_X = np.delete(X, np.s_[i*len:(i+1)*len], 0)
    	train_Y = np.delete(y, np.s_[i*len:(i+1)*len], 0)
    	test_X  = X[i*len:(i+1)*len]
    	test_Y  = y[i*len:(i+1)*len]
    	clf = KernelLogistic(sigma = sigma)
    	clf.fit(train_X, train_Y)
    	y_predict = clf.predict(test_X) > 0.5
    	error += np.sum(y_predict != test_Y)
    return error/k
    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel)
    clf.fit(X1, Y1)
    y_predict = clf.predict(data[900:,:2]) > 0.5
    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")

    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    plt.show()
