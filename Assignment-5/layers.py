'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes 	= in_nodes
        self.out_nodes 	= out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data 		= None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0, 0.1, (in_nodes, out_nodes))    
        self.biases = np.random.normal(0, 0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        self.data = X@self.weights + self.biases
        if self.activation == 'relu':
            return relu_of_X(self.data)
        elif self.activation == 'softmax':
            return softmax_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO      

    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        batch_size = activation_prev.shape[0]
        if self.activation == 'relu':
            grad = gradient_relu_of_X(relu_of_X(self.data), delta)
            new_delta = grad @ self.weights.T
            self.weights -= lr * activation_prev.T @ grad / batch_size
            self.biases  -= lr * grad.sum(axis = 0) / batch_size
            return new_delta

        elif self.activation == 'softmax':
            grad = gradient_softmax_of_X(softmax_of_X(self.data), delta)
            new_delta = grad @ self.weights.T
            self.weights -= lr * activation_prev.T @ grad / batch_size
            self.biases  -= lr * grad.sum(axis = 0, keepdims = True) / batch_size
            return new_delta

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO
        
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        out = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col))

        # Convolve
        for i in range(self.out_row):
            h_st = i * self.stride
            h_en = h_st + self.filter_row

            for j in range(self.out_col):
                v_st = j * self.stride
                v_en = v_st + self.filter_col

                out[:, :, i, j] = np.sum(X[:, np.newaxis, :, h_st:h_en, v_st:v_en] *
                    self.weights[np.newaxis, :, :, :, :], axis = (2, 3, 4))
        
        self.data = out + self.biases[np.newaxis, :, np.newaxis, np.newaxis]

        if self.activation == 'relu':
            return relu_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO
        ###############################################
        batch_size = activation_prev.shape[0]

        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(relu_of_X(self.data), delta)

            dw = np.zeros_like(self.weights)
            new_delta = np.zeros_like(activation_prev)

            for i in range(self.out_row):
                h_st = i * self.stride
                h_en = h_st + self.filter_row

                for j in range(self.out_col):
                    v_st = j * self.stride
                    v_en = v_st + self.filter_col

                    new_delta[:, :, h_st:h_en, v_st:v_en] += \
                        np.sum(self.weights[np.newaxis, :,  :, :, :] * inp_delta[:, :, np.newaxis, i:i+1, j:j+1], axis = 1)
                        
                    dw += np.sum(activation_prev[:, np.newaxis, :, h_st:h_en, v_st:v_en] * 
                        inp_delta[:, :, np.newaxis, i:i+1, j:j+1], axis = 0)

            self.biases  -= lr * inp_delta.sum(axis = (0, 2, 3)) / batch_size
            self.weights -= lr * dw / batch_size
            return new_delta
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO

class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        # Can also be done like MaxPool
        return X.reshape(X.shape[0], self.in_depth, self.out_row, self.filter_row, self.out_col, self.filter_col).mean(axis = (3, 5))
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        # Can also be done like MaxPool
        # Note difference between np.tile and np.repeat
        new_delta = delta.repeat(self.filter_col).reshape(delta.shape[0], self.in_depth, self.out_row, self.in_col)/self.filter_col # Across col
        new_delta = np.tile(new_delta, self.filter_row).reshape(delta.shape[0], self.in_depth, self.in_row, self.in_col)/self.filter_row # Across row
        return new_delta
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        out = np.zeros((X.shape[0], self.in_depth, self.out_row, self.out_col))
        for i in range(self.out_row):
            h_st = i * self.stride
            h_en = h_st + self.filter_row
            
            for j in range(self.out_col):
                v_st = j * self.stride
                v_en = v_st + self.filter_col
                
                out[:, :, i, j] = np.amax(X[:, :, h_st:h_en, v_st:v_en], axis=(2, 3))
        return out
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        out = np.zeros_like(activation_prev)
        batch_size = activation_prev.shape[0]
        for i in range(self.out_row):
            h_st = i * self.stride
            h_en = h_st + self.filter_row

            for j in range(self.out_col):
                v_st = j * self.stride
                v_en = v_st + self.filter_col

                # Design mask
                a_slice = np.array(activation_prev[:, :, h_st:h_en, v_st:v_en], copy = True).reshape(batch_size, self.in_depth, -1)
                mask = np.zeros_like(a_slice)
                n_idx, c_idx = np.indices((batch_size, self.in_depth))
                idxs = np.argmax(a_slice, axis = 2)
                mask[n_idx, c_idx, idxs] = 1
                mask = mask.reshape(batch_size, self.in_depth, self.filter_row, self.filter_col)
                out[:, :, h_st:h_en, v_st:v_en] = delta[:, :, i:i+1, j:j+1]*mask
        return out
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        # print(X.shape)
        return X.reshape(X.shape[0], -1)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(activation_prev.shape)
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    Y = X.copy()
    Y[Y<0] = 0
    return Y
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    grad = np.zeros_like(X)
    grad[X>0] = 1
    return grad*delta
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    Y = np.exp(X)
    return Y/np.sum(Y, axis = 1, keepdims = True)
    # END TODO  

def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    grad = X * (delta - np.sum(X * delta, axis = 1, keepdims = True))
    return grad
    # END TODO
