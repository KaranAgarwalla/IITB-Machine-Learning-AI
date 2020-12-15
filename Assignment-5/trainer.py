'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *
from nn import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		self.printTrainStats = False
		self.printValStats = False
		self.loadModel = False
		self.model_name = ""
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 20
			self.lr = 0.1
			self.nn = NeuralNetwork(self.YTrain.shape[1], self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1], self.YTrain.shape[1], 'softmax'))

		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]
			self.save_model = True
			self.model_name = "model.p"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 25
			self.lr = 0.05
			self.nn = NeuralNetwork(self.YTrain.shape[1], self.lr)
			self.nn.addLayer(ConvolutionLayer(self.XTrain[0].shape, (4, 4), 8, 2, 'relu'))
			self.nn.addLayer(MaxPoolingLayer( (8, 15, 15), (3, 3), 3) )
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(200, self.YTrain.shape[1], 'softmax'))

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 20
			self.lr = 0.5
			self.nn = NeuralNetwork(self.YTrain.shape[1], self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1], 4, 'softmax'))
			self.nn.addLayer(FullyConnectedLayer(4, self.YTrain.shape[1], 'softmax'))

		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 40
			self.lr = 0.2
			self.nn = NeuralNetwork(self.YTrain.shape[1], self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1], 2, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(2, self.YTrain.shape[1], 'softmax'))
	         
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		if self.loadModel:
			model = np.asarray(pickle.load(open(self.model_name, "rb")))
			model_idx = 0
			for l in self.nn.layers:
				if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer" and type(l).__name__!= "MaxPoolingLayer": 
					l.weights 	= model[model_idx]
					l.biases	= model[model_idx+1]
					model_idx 	+= 2					
			print("Model Loaded... ")

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			# TODO

			# Shuffle the training data for the current epoch
			X = np.asarray(self.XTrain)
			Y = np.asarray(self.YTrain)
			# Cannot do shuffle directly; both X and Y have to match
			idx = np.arange(X.shape[0])
			np.random.shuffle(idx)
			X = X[idx]
			Y = Y[idx]

			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches
			numBatches = int(np.ceil(float(X.shape[0])/self.batch_size))
			for batch_idx in np.arange(numBatches):
				batchX = X[batch_idx*self.batch_size: min((batch_idx+1)*self.batch_size, X.shape[0])]
				batchY = Y[batch_idx*self.batch_size: min((batch_idx+1)*self.batch_size, Y.shape[0])]
	
				# Calculate the activations after the feedforward pass
				activations = self.nn.feedforward(batchX)

				# Compute the loss
				batchLoss = self.nn.computeLoss(batchY, activations)  
				trainLoss += batchLoss

				# Calculate the training accuracy for the current batch

				# Check nn.computeAccuracy: compares arrays
				predLabels = oneHotEncodeY(np.argmax(activations[-1], 1), self.nn.out_nodes)
				batchAcc = self.nn.computeAccuracy(batchY, predLabels)
				trainAcc += batchAcc

				# Backpropagation Pass to adjust weights and biases of the neural network
				self.nn.backpropagate(activations, batchY)
			# END TODO

			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			if self.printTrainStats:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer" and type(l).__name__!= "MaxPoolingLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and self.printValStats:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)
