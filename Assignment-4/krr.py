import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from kernel import *

def plot_3D(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs=x, ys=y, zs=z, zdir='z')
    plt.show()

def plot_2D(x, y):
    plt.plot(x, y, 'o')
    plt.show()

def plot_alongside_data(X, Y, f, ax, title=''):
    if X.shape[1] == 1:
        # 2D plot
        min_ranges = [min(X[:, i]) for i in range(X.shape[1])]
        max_ranges = [max(X[:, i]) for i in range(X.shape[1])]
        n_points = X.shape[0]
        X_test = [np.linspace(mi, ma, n_points) for mi, ma in zip(min_ranges, max_ranges)]
        X_test = np.stack(X_test).T
        Y_test = f(X_test)
        ax.plot(X, Y, 'o')
        ax.plot(X_test, Y_test)
        ax.set_title(title)
    elif X.shape[1] == 2:
        # 3D plot
        error = np.linalg.norm(f(X)-Y)**2
        print("Error of the fit: ",error)
        min_ranges = [min(X[:, i]) for i in range(X.shape[1])]
        max_ranges = [max(X[:, i]) for i in range(X.shape[1])]
        P = [np.linspace(mi, ma, 20) for mi, ma in zip(min_ranges, max_ranges)]
        X_test = np.transpose([np.tile(P[0], len(P[1])), np.repeat(P[1], len(P[0]))])
        Y_test = f(X_test)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(xs=X[:,0], ys=X[:,1], zs=Y, zdir='z')
        X, Y = np.meshgrid(P[0], P[1])
        Z = Y_test.reshape(len(P[0]), len(P[1]))
        surf = ax.plot_surface(X, Y, Z,color='g')
        plt.show()
        return error

def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [list(map(lambda s: float(s.strip()), x.split(','))) for x in lines]
        lines = np.asarray(lines)
        Y = lines[:, -1:]
        X = lines[:, :-1]
    return X, Y

class KernelRidgeRegression(object):
    def __init__(self, kernel=gaussian_kernel,lamda=0.01,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.alpha = None
        self.lamda = lamda  # Regularization term

    def fit(self, X, y):
        '''Finds the values of self.alpha given X and y
           Use the closed form expression for this!
        Arguments:
            X - N x d matrix
            y - N x 1 matrix
        '''
        self.train_X = X
        self.train_y = y
        # TODO - compute value of alpha and save it to self.alpha
        self.alpha = np.linalg.inv(self.kernel(X, X)+self.lamda*np.identity(X.shape[0]))@y
        # END TODO
    

    def predict(self, X):
        '''Returns the predictions for the given X
        
        Arguments:
            X - M x d matrix 
        Return:
            y - M x 1 matrix of predictions 
        '''
        # TODO 
        y = self.kernel(X, self.train_X)@self.alpha
        return y
        # END TODO

if __name__ == '__main__':
    fig,ax = plt.subplots(nrows=1,ncols=3, figsize=(9,3))
    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,0.01,10)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[1],title='sigma=10')

    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,0.01,100)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[2],title='sigma=100')

    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,0.01,1)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[0],title='sigma=1')

    plt.tight_layout()
    plt.show()

    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(9, 3))
    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,10,10)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[1],title='Lambda=10')

    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,100,10)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[2],title='Lambda=100')

    X, Y = read_data('./data/krr.csv')
    clf = KernelRidgeRegression(gaussian_kernel,0.1,10)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[0],title='Lambda=0.1')

    plt.tight_layout()
    plt.show()

    X, Y = read_data('./data/kernel_design.csv')
    clf = KernelRidgeRegression(my_kernel,0.01,1)
    clf.fit(X, Y)
    plot_alongside_data(X, Y, clf.predict,ax[0],title='Custom Kernel')

    plt.show()
    
    err = plot_alongside_data(X, Y, clf.predict,ax[0],title='Custom Kernel')

    plt.show()

    if err < 7000:
        marks = 1.5
    elif err < 7500:
        marks = 1.0
    elif err < 8000:
        marks = 0.5
    else:
        marks = 0

    print(f"You have provisionally obtained {marks} marks for Task 3")