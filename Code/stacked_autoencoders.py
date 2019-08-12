

from load_mnist import mnist
import numpy as np
import matplotlib.pyplot as plt
import pdb
from deepMultiClassNetwork_starter import  multi_layer_network
import deepMultiClassNetwork_starter
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def relu(Z):
    '''
    computes relu activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def tanh(Z):
	'''
	computes tanh activation of Z

	Inputs: 
		Z is a numpy.ndarray (n, m)

	Returns: 
		A is activation. numpy.ndarray (n, m)
		cache is a dictionary with {"Z", Z}
	'''
	A = np.tanh(Z)
	cache = {}
	cache["Z"] = Z
	return A, cache

def tanh_der(dA, cache):
	'''
	computes derivative of tanh activation

	Inputs: 
		dA is the derivative from subsequent layer. numpy.ndarray (n, m)
		cache is a dictionary with {"Z", Z}, where Z was the input 
		to the activation layer during forward propagation

	Returns: 
		dZ is the derivative. numpy.ndarray (n,m)
	'''
	### CODE HERE
	Z = cache["Z"]
	A, Cache = tanh(Z)
	dZ = dA *(1-A*A)
	return dZ

def sigmoid(Z):
	'''
	computes sigmoid activation of Z

	Inputs: 
		Z is a numpy.ndarray (n, m)

	Returns: 
		A is activation. numpy.ndarray (n, m)
		cache is a dictionary with {"Z", Z}
	'''
	A = 1/(1+np.exp(-Z))
	cache = {}
	cache["Z"] = Z
	return A, cache

def sigmoid_der(dA, cache):
	'''
	computes derivative of sigmoid activation

	Inputs: 
		dA is the derivative from subsequent layer. numpy.ndarray (n, m)
		cache is a dictionary with {"Z", Z}, where Z was the input 
		to the activation layer during forward propagation

	Returns: 
		dZ is the derivative. numpy.ndarray (n,m)
	'''
	### CODE HERE
	Z = cache['Z']
	A, cache = sigmoid(Z)


	dZ = dA * A * (1-A)
	return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
	'''
	Initializes the weights of the 2 layer network

	Inputs: 
		n_in input dimensions (first layer)
		n_h hidden layer dimensions
		n_fin final layer dimensions

	Returns:
		dictionary of parameters
	'''
	# initialize network parameters
	### CODE HERE

	parameters = {}
	W1=np.random.randn(n_h,n_in) * np.sqrt(1/n_h)
	W2=np.random.randn(n_fin,n_h) * np.sqrt(1/n_fin)
	b1=np.random.randn(n_h,1) * np.sqrt(1/n_h)
	b2=np.random.randn(n_fin,1) * np.sqrt(1/n_fin)
	parameters["W1"] = W1
	parameters["b1"] = b1
	parameters["W2"] = W2
	parameters["b2"] = b2

	return parameters

def linear_forward(A, W, b):
	'''
	Input A propagates through the layer 
	Z = WA + b is the output of this layer. 

	Inputs: 
		A - numpy.ndarray (n,m) the input to the layer
		W - numpy.ndarray (n_out, n) the weights of the layer
		b - numpy.ndarray (n_out, 1) the bias of the layer

	Returns:
		Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
		cache - a dictionary containing the inputs A, W and b
		to be used for derivative
	'''
	### CODE HERE
	Z = np.dot(W,A)+b
	cache = {}
	cache["A"] = A
	# cache["W"] = W
	# cache["b"] = b
	return Z, cache

def layer_forward(A_prev, W, b, activation):
	'''
	Input A_prev propagates through the layer and the activation

	Inputs: 
		A_prev - numpy.ndarray (n,m) the input to the layer
		W - numpy.ndarray (n_out, n) the weights of the layer
		b - numpy.ndarray (n_out, 1) the bias of the layer
		activation - is the string that specifies the activation function

	Returns:
		A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
		g is the activation function
		cache - a dictionary containing the cache from the linear and the nonlinear propagation
		to be used for derivative
	'''
	Z, lin_cache = linear_forward(A_prev, W, b)
	if activation == "sigmoid":
		A, act_cache = sigmoid(Z)
	elif activation == "tanh":
		A, act_cache = tanh(Z)
	
	cache = {}
	cache["lin_cache"] = lin_cache
	cache["act_cache"] = act_cache

	return A, cache

def reconstruction_loss(A2, Y):
	'''
	Estimates the cost with prediction A2

	Inputs:
		A2 - numpy.ndarray (1,m) of activations from the last layer
		Y - numpy.ndarray (1,m) of labels
	
	Returns:
		cost of the objective function
	'''
	### CODE HERE
	cost=np.sum((A2-Y)**2)/Y.shape[1]
	#cost=np.sum((A2-Y)**2)/Y.shape[1]
	#return cost

	#cost=np.sum(np.sqrt(np.sum((A2-Y)**2,axis=0)))/Y.shape[1]
	return cost

def linear_backward(dZ, cache, W, b):
	'''
	Backward propagation through the linear layer

	Inputs:
		dZ - numpy.ndarray (n,m) derivative dL/dz 
		cache - a dictionary containing the inputs A
			where Z = WA + b,    
			Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
		W - numpy.ndarray (n,p)  
		b - numpy.ndarray (n, 1)

	Returns:
		dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
		dW - numpy.ndarray (n,p) the gradient of W 
		db - numpy.ndarray (n, 1) the gradient of b
	'''
	# CODE HERE
	A = cache["A"]
	dW = np.dot(dZ,A.T)/A.shape[1]
	db = np.sum(dZ,axis=1,keepdims=True)/A.shape[1]
	dA_prev = np.dot(W.T,dZ)
	return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
	'''
	Backward propagation through the activation and linear layer

	Inputs:
		dA - numpy.ndarray (n,m) the derivative to the previous layer
		cache - dictionary containing the linear_cache and the activation_cache
		W - numpy.ndarray (n,p)  
		b - numpy.ndarray (n, 1)
	
	Returns:
		dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
		dW - numpy.ndarray (n,p) the gradient of W 
		db - numpy.ndarray (n, 1) the gradient of b
	'''
	lin_cache = cache["lin_cache"]
	act_cache = cache["act_cache"]

	if activation == "sigmoid":
		dZ = sigmoid_der(dA, act_cache)
	elif activation == "tanh":
		dZ = tanh_der(dA, act_cache)
	dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
	return dA_prev, dW, db

def classify(X, parameters):
	'''
	Network prediction for inputs X

	Inputs: 
		X - numpy.ndarray (n,m) with n features and m samples
		parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
	Returns:
		YPred - numpy.ndarray (1,m) of predictions
	'''
	### CODE HERE
	A1, cache1 = layer_forward(X, parameters["W1"], parameters["b1"], "sigmoid")
	YPred, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
	return YPred

def two_layer_network(X, Y,net_dims, num_iterations=2000, learning_rate=0.1):
	'''
	Creates the 2 layer network and trains the network

	Inputs:
		X - numpy.ndarray (n,m) of training data
		Y - numpy.ndarray (1,m) of training data labels
		net_dims - tuple of layer dimensions
		num_iterations - num of epochs to train
		learning_rate - step size for gradient descent
	
	Returns:
		costs - list of costs over training
		parameters - dictionary of trained network parameters
	'''

	n_in, n_h, n_fin = net_dims
	print(net_dims)
	parameters = initialize_2layer_weights(n_in, n_h, n_fin)
	print(parameters["W1"].shape)
	print(parameters["b1"].shape)
	print(parameters["W2"].shape)
	print(parameters["b2"].shape)
	
	A0 = X
	
	costs = []
	fig = plt.figure()
	count = 1
	for ii in range(num_iterations):
		# Forward propagation
		### CODE HERE
		A1, C1 = layer_forward(A0,parameters["W1"],parameters["b1"],"sigmoid")
		A2, C2 = layer_forward(A1,parameters["W2"],parameters["b2"],"sigmoid")
		# cost estimation
		### CODE HERE
		cost = reconstruction_loss(A2,X)
		# Backward Propagation
		### CODE HERE
		dA2 = -2 * (X-A2)
		dA1, dW2, dB2 = layer_backward(dA2,C2,parameters["W2"],parameters["b2"],"sigmoid")
		dA0, dW1, dB1 = layer_backward(dA1,C1,parameters["W1"],parameters["b1"],"sigmoid")


		#update parameters
		### CODE HERE
		parameters["W1"] = parameters["W1"] - (learning_rate * dW1)
		parameters["b1"] = parameters["b1"] - (learning_rate * dB1)
		parameters["W2"] = parameters["W2"] - (learning_rate * dW2)
		parameters["b2"] = parameters["b2"] - (learning_rate * dB2)

		if ii % 10 == 0:
			costs.append(cost)
			print("Cost at iteration %i is: %f" %(ii, cost))
	print(A2.shape)
	# for s in range(10):
	# 	shape = np.sqrt(A2.shape[1])
	# 	ind=np.argwhere(Y[0]==s)[0][0]
	# 	img = (A2.T[ind]).reshape(shape,shape)
	# 	plt.subplot(5,2,count)
	# 	count+=1
	# 	plt.xticks([])
	# 	plt.yticks([])
	# 	plt.grid(False)
	# 	plt.imshow(img, cmap=plt.cm.binary)
	# plt.show()
	
	return costs, parameters, A1

def main():
	# getting the subset dataset from MNIST
	# binary classification for digits 1 and 7

	validation_costs=[]
	test_accuracies=[]

	# getting the subset dataset from MNIST
	data, data_label, test_data, test_label = \
			mnist(noTrSamples=6000,noTsSamples=50,\
			digit_range=[0,1,2,3,4,5,6,7,8,9],\
			noTrPerClass=600, noTsPerClass=5)

	data_last, data_last_label, x, y = \
			mnist(noTrSamples=10,noTsSamples=50,\
			digit_range=[0,1,2,3,4,5,6,7,8,9],\
			noTrPerClass=1, noTsPerClass=5)





	# test_data, test_label, test_data_1, test_labe_1 = \
	# 		mnist(noTrSamples=10000,noTsSamples=1000,\
	# 		digit_range=[0,1,2,3,4,5,6,7,8,9],\
	# 		noTrPerClass=1000, noTsPerClass=100)
	par = 1

	if par ==1 : 

	
		# initialize learning rate and num_iterations
		num_iterations = 3000

		n_in, m = data.shape
		learning_rate = 0.2

		net_dims = [784, 500, 784]
		costs_1, parameters_1, A_1 = two_layer_network(data, data_label,net_dims, \
					num_iterations=num_iterations, learning_rate=learning_rate)
		learning_rate = 0.4
		num_iterations = 1500
		net_dims = [500, 200, 500]
		costs_2,parameters_2, A_2 = two_layer_network(A_1, data_label,net_dims, \
					num_iterations=num_iterations, learning_rate=learning_rate)

		net_dims = [200,100,200]
		costs_3, parameters_3, A_3 = two_layer_network(A_2, data_label,net_dims, \
		 			num_iterations=num_iterations, learning_rate=learning_rate)


		parameters = {}
		parameters["W1"] = parameters_1["W1"]
		parameters["b1"] = parameters_1["b1"]
		parameters["W2"] = parameters_2["W1"]
		parameters["b2"] = parameters_2["b1"]
		parameters["W3"] = parameters_3["W1"]
		parameters["b3"] = parameters_3["b1"]
		print(parameters["W1"].shape)
		print(parameters["W2"].shape)
		#print(parameters["W3"].shape)
		print(parameters)

		net_dims = [784, 500,200,100, 10]
		learning_rate = 1
		num_iterations = 5000
		costs_4, parameters_4 = multi_layer_network(parameters, data_last, data_last_label, net_dims, \
													num_iterations=num_iterations, learning_rate=learning_rate)

		train_Pred = deepMultiClassNetwork_starter.classify(data, parameters_4)
		test_Pred = deepMultiClassNetwork_starter.classify(test_data, parameters_4)
		print("shape-DATA", data_label.shape)
		print("shape-TRAIN", train_Pred.shape)
		print("count:",np.count_nonzero(train_Pred - data_label))
		trAcc = (1 - np.count_nonzero(train_Pred - data_label) / float(train_Pred.shape[1])) * 100
		teAcc = (1 - np.count_nonzero(test_Pred - test_label) / float(test_Pred.shape[1])) * 100

		print("Accuracy for training set is {0:0.3f} %".format(trAcc))
		print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
	if par == 2:

		clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
		clf.fit(data.T, data_label.T)
		out=clf.predict(test_data.T)
		print(out)
		print ("Accuracy",(np.sum(out == test_label)/test_label.shape[1]) * 100)

	if par ==3:
		clf = LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial').fit(data.T, data_label.T)
		score = clf.score(test_data.T,test_label.T)
		print(score)
	if par == 4:
		clf = GaussianNB()
		clf.fit(data.T, data_label.T)
		score = clf.score(test_data.T,test_label.T)
		print(score)

		clf = BernoulliNB()
		clf.fit(data.T, data_label.T)
		score = clf.score(test_data.T,test_label.T)
		print(score)

		clf = MultinomialNB()
		clf.fit(data.T, data_label.T)
		score = clf.score(test_data.T,test_label.T)
		print(score)
	if par == 5:
		clf = DecisionTreeClassifier(criterion='gini',random_state=0)
		clf.fit(data.T, data_label.T)
		score = clf.score(test_data.T,test_label.T)
		clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
		clf.fit(data.T, np.ravel(data_label.T))
		score = clf.score(test_data.T,np.ravel(test_label.T))
		print(score)











	

if __name__ == "__main__":
	main()



