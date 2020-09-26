import numpy as np
import argparse

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )
	
	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	# The features 1 to 4 are k neighbour classifiers and are normalised
	# The 0th feature is bias 1
	features = np.zeros(5,)
	x = x.reshape(50, 50)
	for i in range(1, 49):
		for j in range(1, 49):
			features[int(x[i-1, j]+x[i+1, j]+x[i, j-1]+x[i, j+1])] += 1
	stdarr  = np.array([150, 16, 19, 15, 146])
	meanarr = np.array([1740, 55, 66, 47, 394])
	features = (features - meanarr)/stdarr
	features[0] = 1
	return features
	### END TODO

class Perceptron():
	def __init__(self, C, D):
		'''
		C - number of classes
		D - number of features
		'''
		self.C = C
		self.weights = np.zeros((C, D))
		
	def pred(self, x):
		'''
		x - numpy array of shape (D,)
		'''
		### TODO: Return predicted class for x
		return np.argmax(self.weights@x)
		### END TODO

	def train(self, X, Y, max_iter=20):
		for iter in range(max_iter):
			for i in range(X.shape[0]):
				### TODO: Update weights
				y_pred = self.pred(X[i])
				if y_pred != Y[i]:
					self.weights[Y[i]] += X[i]
					self.weights[y_pred] -= X[i]
				### END TODO
			# print(f'Train Accuracy at iter {iter} = {self.eval(X, Y)}')

	def eval(self, X, Y):
		n_samples = X.shape[0]
		correct = 0
		for i in range(X.shape[0]):
			if self.pred(X[i]) == Y[i]:
				correct += 1
		return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
