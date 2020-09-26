import numpy as np
import perceptron
from binary_logistic_regression import *
from utils import *
from multiclass_logistic_regression import *

np.random.seed(335)


def grade1():
	print("="*20 + "Grading Problem 1" + "="*20)
	marks = 0
	accs = [0.90, 0.85, 0.70, 0.50]
	try:
		X_train, Y_train, X_test, Y_test = perceptron.get_data('D2')

		assert perceptron.get_features(X_train[0]).size <=5, 'Atmost 5 features are allowed'
		
		X_train = np.array([perceptron.get_features(x) for x in X_train])
		X_test = np.array([perceptron.get_features(x) for x in X_test])

		C = max(np.max(Y_train), np.max(Y_test))+1
		D = X_train.shape[1]

		p = perceptron.Perceptron(C, D)

		p.train(X_train, Y_train)
		acc = p.eval(X_test, Y_test)

		if acc>=accs[0]:
			marks += 2.0
		elif acc>=accs[1]:
			marks += 1.5
		elif acc>=accs[2]:
			marks += 1.0
		elif acc>=accs[3]:
			marks += 0.5
	except:
		print('Error')
	print("Marks obtained in Problem 1: ", marks)
	return marks


def grade2():
	marks = 0.0
	try:
		X, Y = load_data('data/songs.csv')
		X, Y = preprocess(X, Y)
		X_train, Y_train, X_test, Y_test = split_data(X, Y)
		print("=" * 20 + "Grading Problem 2.2(a)(1)" + "=" * 20)
		try:
			assert X_test.shape[0] == 373 and X_train.shape[0] == 7191
			marks += 0.5
		except:
			print("Split is wrong.")

		print("=" * 20 + "Grading Problem 2.2(a)(3)" + "=" * 20)

		D = X_train.shape[1]
		lr = BinaryLogisticRegression(D)
		lr.train(X_train, Y_train)
		preds = lr.predict(X_test)
		acc = lr.accuracy(preds, Y_test)
		f1 = lr.f1_score(preds, Y_test)

		if acc >= 0.8:
			marks += 2.0
		else:
			print("Test accuracy is too small.")

		print("=" * 20 + "Grading Problem 2.2(c)" + "=" * 20)
		a = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
		p = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
		f_test = lr.f1_score(a, p)

		try:
			assert abs(f_test - 2/3) < 1e-3
			assert f1 >= 0.25
			marks += 1.5
			print("0.5 marks will be awarded based on your justification for this task.")
		except:
			print("Error in problem 2.2(c)")
	except:
		print('Error')
	print("Marks obtained in Problem 2.2(a) and 2.2(c): ", marks)
	return marks


def grade3():

	print("=" * 20 + "Grading Problem 2.2(d)" + "=" * 20)
	marks = 0.0
	X_train, Y_train, X_test, Y_test = get_data("D1")

	C = max(np.max(Y_train), np.max(Y_test)) + 1
	D = X_train.shape[1]

	lr = LogisticRegression(C, D)
	lr.train(X_train, Y_train)
	acc = lr.eval(X_test, Y_test)
	try:
		assert acc > 0.78
		marks += 3
	except:
		print("Test accuracy too small")
	print("Marks obtained in Problem 2.2(d): ", marks)
	return marks


print(f'Total Autograded Marks = {grade1() + grade2() + grade3()} / 9.0')