import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def gen_non_lin_separable_data(num):
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, num)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, num)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, num)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, num)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def split_train(X1, y1, X2, y2, t):
    X1_train = X1[:t]
    y1_train = y1[:t]
    X2_train = X2[:t]
    y2_train = y2[:t]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2, t):
    X1_test = X1[t:]
    y1_test = y1[t:]
    X2_test = X2[t:]
    y2_test = y2[t:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

def read_data(filename):
    '''
    Reads the input training data from filename and 
    Returns the matrices X : [N X D] and Y : [N X 1] where D is number of features and N is the number of data points
    # '''
    # dataframe = pd.read_csv(filename, keep_default_na=False, na_values='')
    # # data = [np.array(dataframe[col]) for col in dataframe]
    # # for i, d in enumerate(data):
    # #     data[i].shape = (data[i].shape[0], 1)
    # # data = np.concatenate(data, axis = 1)
    data = np.loadtxt(filename,delimiter=',')
    X = data[:,:-1]
    Y = data[:,-1]
    Y.shape = (Y.shape[0],1)
    return X, Y

def one_hot_encode(X, labels):
    '''
    X = input [N X 1] matrix data 
    labels = list of all possible labels for current category
    Returns the matrix X : [N X len(labels)] in one hot encoded format
    '''
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i,0]]] = 1
    return newX

def separate_data(X, Y):
    '''
    X = input feature matrix [N X D] 
    Y = output values [N X 1]
    Segregate some part as train and some part as test
    Return the trainX, trainY, testX, testY
    '''
    trainX = X[0:1200, :]
    trainY = Y[0:1200, :]
    
    testX = X[1200:, :]
    testY = Y[1200:, :]

    return trainX, trainY, testX, testY

# def separate_data_randomize(X, Y):
#   '''
#   X = input feature matrix [N X D] 
#   Y = output values [N X 1]
#   Segregate some part as train and some part as test
#   Return the trainX, trainY, testX, testY
#   '''
#   X,Y = shuffle(X,Y)
#   trainX = X[0:30000, :]
#   trainY = Y[0:30000, :]

#   testX = X[30000:, :]
#   testY = Y[30000:, :]

#   return trainX, trainY, testX, testY
