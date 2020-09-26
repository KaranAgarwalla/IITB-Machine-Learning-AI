import numpy as np 
from matplotlib import pyplot as plt
import argparse

from utils import *
from p1 import mse

## ONLY CHANGE CODE BETWEEN TODO and END TODO
def prepare_data(X,degree):
    '''
    X is a numpy matrix of size (n x 1)
    return a numpy matrix of size (n x (degree+1)), which contains higher order terms
    '''
    # TODO
    X = X**np.arange(degree+1)
    # End TODO
    return X

# def plotter():
# 	degree_choices = np.array([1, 2, 3, 4, 5, 6])
# 	train_mse = np.ones((4, 6))
# 	test_mse = np.ones((4, 6))
# 	np.random.seed(42)
# 	X_train, Y_train = load_data1('data3_train.csv')
# 	Y_train = Y_train/20
# 	X_test, Y_test   = load_data1('data3_test.csv')
# 	Y_test = Y_test/20

# 	indices_0 = np.random.choice(np.arange(200),40,replace=False)
# 	indices_1 = np.random.choice(np.arange(200),40,replace=False)
# 	indices_2 = np.random.choice(np.arange(200),40,replace=False)
# 	indices_3 = np.random.choice(np.arange(200),40,replace=False)

# 	for i in range(6):
# 		degree = degree_choices[i]
# 		X_train_i = X_train**np.arange(degree+1)

# 		X_0 = X_train_i[indices_0]
# 		Y_0 = Y_train[indices_0]
# 		X_1 = X_train_i[indices_1] 
# 		Y_1 = Y_train[indices_1]
# 		X_2 = X_train_i[indices_2]
# 		Y_2 = Y_train[indices_2]
# 		X_3 = X_train_i[indices_3]
# 		Y_3 = Y_train[indices_3]
# 		W_0 = np.linalg.inv(X_0.T @ X_0) @ X_0.T @ Y_0
# 		W_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ Y_1
# 		W_2 = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ Y_2
# 		W_3 = np.linalg.inv(X_3.T @ X_3) @ X_3.T @ Y_3

# 		X_test_i = X_test**np.arange(degree+1)

# 		train_mse[0][i] = mse(X_0,Y_0,W_0)
# 		train_mse[1][i] = mse(X_1,Y_1,W_1)
# 		train_mse[2][i] = mse(X_2,Y_2,W_2)
# 		train_mse[3][i] = mse(X_3,Y_3,W_3)
# 		test_mse[0][i]  = mse(X_test_i, Y_test, W_0)
# 		test_mse[1][i]  = mse(X_test_i, Y_test, W_1)
# 		test_mse[2][i]  = mse(X_test_i, Y_test, W_2)
# 		test_mse[3][i]  = mse(X_test_i, Y_test, W_3)

# 	plt.plot(degree_choices, train_mse[0,:], label = 'Train MSE')
# 	plt.plot(degree_choices, test_mse[0, :], label = 'Test MSE')
# 	plt.title('Sample 1')
# 	plt.legend()
# 	plt.show()

# 	plt.plot(degree_choices, train_mse[1,:], label = 'Train MSE')
# 	plt.plot(degree_choices, test_mse[1, :], label = 'Test MSE')
# 	plt.title('Sample 2')
# 	plt.legend()
# 	plt.show()

# 	plt.plot(degree_choices, train_mse[2,:], label = 'Train MSE')
# 	plt.plot(degree_choices, test_mse[2, :], label = 'Test MSE')
# 	plt.title('Sample 3')
# 	plt.legend()
# 	plt.show()

# 	plt.plot(degree_choices, train_mse[3,:], label = 'Train MSE')
# 	plt.plot(degree_choices, test_mse[3, :], label = 'Test MSE')
# 	plt.title('Sample 4')
# 	plt.legend()
# 	plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--degree', type=int, default=3,
                    help='Degree of polynomial to use')
    args = parser.parse_args()
    np.random.seed(42)
    degree = args.degree

    X_train, Y_train = load_data1('data3_train.csv')
    Y_train = Y_train/20
    X_test, Y_test   = load_data1('data3_test.csv')
    Y_test = Y_test/20

    X_train = prepare_data(X_train,degree)
    indices_0 = np.random.choice(np.arange(200),40,replace=False)
    indices_1 = np.random.choice(np.arange(200),40,replace=False)
    indices_2 = np.random.choice(np.arange(200),40,replace=False)
    indices_3 = np.random.choice(np.arange(200),40,replace=False)

    ## TODO - compute each fold using indices above, compute weights using OLS
    X_0 = X_train[indices_0]
    Y_0 = Y_train[indices_0]
    X_1 = X_train[indices_1] 
    Y_1 = Y_train[indices_1]
    X_2 = X_train[indices_2]
    Y_2 = Y_train[indices_2]
    X_3 = X_train[indices_3]
    Y_3 = Y_train[indices_3]
    W_0 = np.linalg.inv(X_0.T @ X_0) @ X_0.T @ Y_0
    W_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ Y_1
    W_2 = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ Y_2
    W_3 = np.linalg.inv(X_3.T @ X_3) @ X_3.T @ Y_3

    ## END TODO


    X_test = prepare_data(X_test,degree)

    train_mse_0 = mse(X_0,Y_0,W_0)
    train_mse_1 = mse(X_1,Y_1,W_1)
    train_mse_2 = mse(X_2,Y_2,W_2)
    train_mse_3 = mse(X_3,Y_3,W_3)
    test_mse_0  = mse(X_test, Y_test, W_0)
    test_mse_1  = mse(X_test, Y_test, W_1)
    test_mse_2  = mse(X_test, Y_test, W_2)
    test_mse_3  = mse(X_test, Y_test, W_3)

    X_lin = np.linspace(X_train[:,1].min(),X_train[:,1].max()).reshape((50,1))
    X_lin = prepare_data(X_lin,degree)
    print(f'Test Error 1: %.4f Test Error 2: %.4f Test Error 3: %.4f test E 4: %.4f'%(test_mse_0,test_mse_1,test_mse_2,test_mse_3))
    plt.scatter(X_train[:,1],Y_train,color='orange')
    plt.plot(X_lin[:,1],X_lin @ W_0, c='g')
    plt.plot(X_lin[:,1],X_lin @ W_1, c='r')
    plt.plot(X_lin[:,1],X_lin @ W_2, c='b')
    plt.plot(X_lin[:,1],X_lin @ W_3, color='purple')
    plt.plot(X_lin[:,1],X_lin @(W_1+W_2+W_3+W_0)/4, color='black')
    plt.show()