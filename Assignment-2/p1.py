import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO
    mse = np.sum((X@W-Y)**2)/(2*X.shape[0])
    # END TODO

    return mse


def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.01, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    W = np.random.randn(X_train.shape[1], 1)
    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        W_prev = W.copy()
        W = W - lr*X_train.T @ (X_train @ W - Y_train)/X_train.shape[0]
        W[abs(W)<=lr*_lambda] = 0	#All places where it is less than lambda*lr
        W = W - np.sign(W)*lr*_lambda
        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if np.linalg.norm(W-W_prev) < 1e-4:
        	break
        # End TODO
    return W, train_mses, test_mses

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.0003, max_iter=5000):
	'''
	reg - regularization parameter (lambda in Q2.1 c)
	'''
	train_mses = []
	test_mses = []

	## TODO: Initialize W using using random normal 
	W = np.random.randn(X_train.shape[1], 1)
	## END TODO

	for i in range(max_iter):

		## TODO: Compute train and test MSE
		train_mse = mse(X_train, Y_train, W)
		test_mse = mse(X_test, Y_test, W)
		## END TODO

		train_mses.append(train_mse)
		test_mses.append(test_mse)

		## TODO: Update w and b using a single step of gradient descent
		W = (1 - 2*lr*reg)*W - lr*np.matmul(np.transpose(X_train), np.matmul(X_train, W)-Y_train)/X_train.shape[0]
		## END TODO

	return W, train_mses, test_mses

if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    # Problem 1.2(b)
    lambda_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    train_mses_arr = np.empty(16)
    test_mses_arr  = np.empty(16)
    for i in range(16):
    	_, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, lambda_arr[i], 0.001, 10000)
    	train_mses_arr[i] = train_mses_ista[-1]
    	test_mses_arr[i]  = test_mses_ista[-1]
    plt.plot(lambda_arr, train_mses_arr, label = 'Train MSE')
    plt.plot(lambda_arr, test_mses_arr, label = 'Test MSE')
    plt.title('ISTA Plot')
    plt.legend()
    plt.show()

    #Problem 1.2(c)
    reg = 10
    W_ridge, _, _ = ridge_regression(X_train, Y_train, X_test, Y_test, reg)
    W_ista, _, _ = ista(X_train, Y_train, X_test, Y_test, 0.2, 0.001, 10000)
    plt.scatter(np.arange(W_ista.shape[0]), W_ista, color = 'blue')
    plt.title("ISTA Scatter Plot")
    plt.show()
    plt.scatter(np.arange(W_ridge.shape[0]), W_ridge, color = 'orange')
    plt.title("Ridge Scatter Plot")
    plt.show()
    # End TODO
