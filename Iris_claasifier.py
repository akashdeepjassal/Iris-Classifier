import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from sklearn import datasets
from sklearn.cross_validation import train_test_split
#import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
target = iris.target

# Convert labels into one-hot vectors
num_labels = len(np.unique(target))
labels = np.eye(num_labels)[target]

# Keeping 20% of data samples as test set
train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size = 0.20)

x_size = train_X[0].shape[0] # size of input layer - "4"
h_size = 100 # size of hidden layers(100 nodes)
y_size = train_y[0].shape[0] # size of output layer - "3"

alpha = 0.02 # Learning rate


# Floating type symbolic expression for training features
X = T.fmatrix(name="X")

# Floating type symbolic expression for training targets
y = T.fmatrix(name="y")

W1_rand = 1/float(np.sqrt(x_size)) # Initialization limit for W1

W2_rand = 1/float(np.sqrt(h_size)) # Initialization limit for W2

# Theano Shared variables for neural network parameters 

# Weight for connections between input and hidden layer
W1 = theano.shared(np.random.uniform(low = -W1_rand, high = W1_rand, size = (x_size, h_size)), name = "W1")

# Bias weights for hidden layer
b1 = theano.shared(np.zeros(h_size), name='b1')

# Weight for connections between input and hidden layer
W2 = theano.shared(np.random.uniform(low = -W2_rand, high = W2_rand, size = (h_size, y_size)), name = "W2")

b2 = theano.shared(np.zeros(y_size), name='b2')



# Forward Propagation
z1 = T.dot(X, W1) + b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1, W2) + b2
y_hat = T.nnet.softmax(z2)

# the loss function we want to optimize
loss = T.nnet.categorical_crossentropy(y_hat, y).mean()

# Returns a target prediction
prediction = T.argmax(y_hat, axis=1)

# Theano functions that can be called from our Python code
forward_prop = theano.function([X], y_hat)
calculate_loss = theano.function([X, y], loss)
predict = theano.function([X], prediction)

# Defines automatic differentiation of all weight w.r.t loss
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

# Gradient step
gradient_step = theano.function(
    [X, y],
    updates=((W2, W2 - alpha * dW2),
             (W1, W1 - alpha * dW1),
             (b2, b2 - alpha * db2),
             (b1, b1 - alpha * db1)))


epochs = 500

for epoch in np.arange(epochs):
    
    # One gradient step with complete training set
    gradient_step(np.array(train_X, 'float32'), np.array(train_y, 'float32'))
    
    if epoch % 10 == 0 or epoch < 10:
        
        # Get the loss
        current_loss = calculate_loss(np.array(train_X, 'float32'), np.array(train_y, 'float32'))
        
        # Get the accuracy between predicted and real target values
        accuracy = np.mean(np.argmax(test_y, axis=1) == predict(np.array(test_X, 'float32')))
        
        print("Epoch -", epoch, " |\t Loss: ", current_loss, " |\t Accuracy: ", accuracy)
       