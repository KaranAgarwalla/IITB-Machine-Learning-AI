import numpy as np

def load_data1(file):
	'''
	Given a file, this function returns X, the regression features
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,1.33, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''

	data = np.loadtxt(file, delimiter=',', skiprows=1)
	X = data[:, :-1]
	Y = data[:, -1:]

	return X, Y

def load_data2(file):
	'''
	Given a file, this function returns X, the features 
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,Yes, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''
	data = np.loadtxt(file, delimiter=',', skiprows=1, dtype='str')
	X = data[:, :-1]
	Y = data[:, -1:].astype(float)

	return X, Y


def split_data(X, Y, train_ratio=0.8):
	'''
	Split data into train and test sets
	The first floor(train_ratio*n_sample) samples form the train set
	and the remaining the test set

	Args:
	X - numpy array of shape (n_samples, n_features)
	Y - numpy array of shape (n_samples, 1)
	train_ratio - fraction of samples to be used as training data

	Returns:
	X_train, Y_train, X_test, Y_test
	'''

	## TODO
	train_len = int(np.floor(X.shape[0]*train_ratio))
	X_train = X[:train_len, :]
	Y_train = Y[:train_len, :]
	X_test = X[train_len:, :]
	Y_test = Y[train_len:, :]
	## END TODO

	return X_train, Y_train, X_test, Y_test

def one_hot_encode(X, labels):
	'''
	Args:
	X - numpy array of shape (n_samples, 1) 
	labels - list of all possible labels for current category
	
	Returns:
	X in one hot encoded format (numpy array of shape (n_samples, n_labels))
	'''
	X.shape = (X.shape[0], 1)
	newX = np.zeros((X.shape[0], len(labels)))
	label_encoding = {}
	for i, l in enumerate(labels):
		label_encoding[l] = i
	for i in range(X.shape[0]):
		newX[i, label_encoding[X[i,0]]] = 1
	return newX

def normalize(X):
	'''
	Returns normalized X

	Args:
	X of shape (n_samples, 1)

	Returns:
	(X - mean(X))/std(X)
	'''
	## TODO
	return (X - np.mean(X))/(np.std(X))
	## END TODO

def preprocess(X, Y):
	'''
	X - feature matrix; numpy array of shape (n_samples, n_features) 
	Y - outputs; numpy array of shape (n_samples, 1)

	Convert data X obtained from load_data2 to a usable format by gradient descent function
	Use one_hot_encode() to convert 

	NOTE 1: X has first column denote index of data point. Ignore that column
			and add constant 1 instead (for bias) 
	NOTE 2: For categorical string data, encode using one_hot_encode() and
			normalize the other features and Y using normalize()
	'''

	## TODO
	X_processes = np.ones((X.shape[0], 1))

	for i in range(1, X.shape[1]):
		try:
			Z = X[:, i].astype('float64')
			X_processes = np.concatenate((X_processes, normalize(Z)[:, np.newaxis]), axis = 1)
		except:
			X_processes = np.concatenate((X_processes, one_hot_encode(X[:, i][:, np.newaxis], np.unique(X[:, i]).tolist())), axis = 1)

	Y = normalize(Y)
	return X_processes, Y
	## END TODO

