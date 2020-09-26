import numpy as np
import utils, single_var_reg, multi_var_reg

def grade1():
	marks = 0

	try:
		X = np.random.rand(110, 5)
		Y = np.random.rand(110, 1)
		X_train, Y_train, X_test, Y_test = utils.split_data(X, Y, 0.75)
		assert np.allclose(np.vstack([X_train, X_test]), X)
		assert np.allclose(np.vstack([Y_train, Y_test]), Y)
		assert len(X_train) == 82 and len(Y_train) == 82
		marks += 0.5
	except:
		print('Q1 split_data() incorrect')
		return marks

	try:
		x = np.array([9.71711545, 5.27658861, 0.74957658, 7.25267862, 1.57512235, 4.95493874, 4.6645458 , 8.81014817, 5.6875507 , 8.9270358 ]).reshape(10, 1)
		y = np.array([7.4395211 , 1.29711056, 4.99824035, 1.87706798, 0.93306619, 6.65645683, 8.6573449 , 2.54946024, 1.3023241 , 6.52289899]).reshape(10, 1)
		w = 0.513244
		b = 1.839345
		assert np.isclose(single_var_reg.mse(x, y, w, b), 4.319008411331635)
		marks += 0.5
	except:
		print('Q1 mse() incorrect')
		return marks

	try:
		X, Y = utils.load_data1('data1.csv')
		X_train, Y_train, X_test, Y_test = utils.split_data(X, Y)
		w, b, train_mses, test_mses = single_var_reg.ordinary_least_squares(X_train, Y_train, X_test, Y_test)
		assert train_mses[-1] < 52
		assert test_mses[-1] < 68
		for i in range(len(train_mses)-1):
			assert train_mses[i] >= train_mses[i+1]
		marks += 3
	except:
		print('Q1 ordinary_least_squares() incorrect')
		return marks

	return marks

def grade2():
	marks = 0

	try:
		X = np.arange(10).reshape(10, 1)
		assert np.allclose(utils.normalize(X).T, [[-1.5666989,-1.21854359,-0.87038828,-0.52223297,-0.17407766,0.17407766,0.52223297,0.87038828,1.21854359,1.5666989]])
		marks += 0.5

	except:
		print('Q2 normalize() incorrect')
		return marks

	try:
		X = np.arange(6).reshape(3,2).astype(float)
		Y = np.arange(3).reshape(3,1).astype(float)
		X_stud, Y_stud = utils.preprocess(X, Y)
		X_act, Y_act =  [[ 1., -1.22474487], [ 1., 0.], [ 1.,1.22474487]], [[-1.22474487], [0.], [1.22474487]]

		assert np.allclose(X_act, X_stud) and np.allclose(Y_act, Y_stud)
		marks += 1
	except:
		print('Q2 preprocess() incorrect')
		return marks

	try:
		X, Y = utils.load_data2('data2.csv')
		X, Y = utils.preprocess(X, Y)
		X_train, Y_train, X_test, Y_test = utils.split_data(X, Y)
		W, train_mses, test_mses = multi_var_reg.ordinary_least_squares(X_train, Y_train, X_test, Y_test)
		assert train_mses[-1] < 0.23
		assert test_mses[-1] < 0.48
		for i in range(len(train_mses)-1):
			assert train_mses[i] >= train_mses[i+1]
		marks += 1.5
	except:
		print('Q2 ordinary_least_squares() incorrect')

	try:
		reg = 10
		W_act = np.linalg.inv(X_train.T @ X_train + 2 * reg * X_train.shape[0] * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
		W, train_mses, test_mses = multi_var_reg.ridge_regression(X_train, Y_train, X_test, Y_test, reg)
		assert train_mses[-1] < 0.3
		assert test_mses[-1] < 0.35
		assert (W@W.T)[0][0] < 1e-7
		assert np.linalg.norm(W - W_act) < 0.5
		# for i in range(len(train_mses)-1):
		# 	assert train_mses[i] >= train_mses[i+1]
		marks += 1.5
	except:
		print('Q2 ridge_regression() incorrect')
	
	return marks

print(f'Marks = {grade1() + grade2()}')