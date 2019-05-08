# Deep Learning project: Detect Maliaria from a given photo
# Data source: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
# Project start: 2019 04 
# Last update: 2019 04 30
# Project directory: C:\Users\vb100\Desktop\Python\Tests\23_maliaria
# Author: Vytautas Bielinskas, vytautas.bielinskas@gmail.com

#------------------------------------------------------------------------------
# Import modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.python.framework import ops
import glob
import h5py
import PIL
import random
from PIL import Image
from PIL import ImageOps
from numpy import asarray
from random import randint
import os

# # # Constants and Parameters # # #
project_directory = r'C:\Users\vb100\Desktop\Python\Tests\23_maliaria'

# # # Functions and Procedures # # #
# | Set Project Directory
def change_directory(directory):
    '''
    Args:
        directory : string
    '''
    os.chdir(directory)
    print('We are working on:\n{}\n'.format(directory))
    return None

# | Read Train and Test Data
def read_data():
    '''
    Description: This procedure read Train and Test Data (images) an convert it
                 to numerical values by stacking it to Matrixes.
    '''
    print('| Reading Data is Being Started.')    
    
    # Checking PIL version
    print('Pillow Version: {}'.format(PIL.__version__))
    print('PIL Version: {}'.format(PIL.VERSION))
    
    # Set Constants
    SEPARATOR = '*'*25
    DATA_DIR = 'cell_images'
    DATA_CLASS_1 = 'Parasitized'
    DATA_CLASS_2 = 'Uninfected'
    IMAGE_FORMAT = 'png'
    NEW_HEIGHT = int(134/2)
    NEW_WIDTH = int(134/2)
    
    # Define Variables
    training_set = []
    current_class = None
    d = {}
    
    print('\n{} Reading Class <1> {}'.format(SEPARATOR, SEPARATOR))
    current_class = 1 # <1 : 'infected'> For using in Tenorflow ANN
    
    # Read Class 1
    class_1_dir = './{}/{}'.format(DATA_DIR, DATA_CLASS_1)
    
    # Change Working Directory to Class 1 Direcory <Infected>
    os.chdir(class_1_dir)
        
    for image_file in glob.glob('*.{}'.format(IMAGE_FORMAT)):
        
        d = {} # Reserve free space for a new Image
        
        print('Reading File: {}'.format(image_file))
    
        # Open an Iterated Image File and Convert it to Grayscale
        opened_image_file = Image.open(image_file)
        opened_image_file = ImageOps.grayscale(opened_image_file)
        
        # Summarize Some Details About the Image
        print('| Image Size (before): {}.'.format(opened_image_file.size))
        
        # Convert Image to Numpy Array
        opened_image_file = asarray(
                opened_image_file.resize((NEW_HEIGHT, NEW_WIDTH))
                )
        
        # Print Data Facts
        print('| Image Size (after): {}'.format(opened_image_file.size))
        
        d['Class'] = current_class
        d['Values'] = opened_image_file
        training_set.append(dict(d))

    # Read Class 2
    print('\n{} Reading Class <2> {}'.format(SEPARATOR, SEPARATOR))
    current_class = 0 # <0 : 'uninfected'> For using in Tenorflow ANN
        
    # Change Working Directory to Class 2 Directory <Uninfected>
    os.chdir('{}\\{}'.format(os.path.dirname(os.getcwd()),
                             DATA_CLASS_2))
    
    for image_file in glob.glob('*.{}'.format(IMAGE_FORMAT)):
        
        d = {} # Reserve free space for a new Image
        
        print('Reading File: {}'.format(image_file))
        
        # Open an Iterated Image File and Convert it to Grayscale
        opened_image_file = Image.open(image_file)
        opened_image_file = ImageOps.grayscale(opened_image_file)
        
        # Summarize Some Details About the Image
        #print('| Image Format: {}.'.format(opened_image_file.format))
        print('| Image Size (before): {}.'.format(opened_image_file.size))
        
        # Convert Image to Numpy Array
        opened_image_file = asarray(
                opened_image_file.resize((NEW_HEIGHT, NEW_WIDTH))
                )
        
        # Print Data Facts
        print('| Image Size (after): {}'.format(opened_image_file.size))
        
        d['Class'] = current_class
        d['Values'] = opened_image_file
        training_set.append(dict(d))
    
    return training_set

def shuffle_data(data):
    '''
    This function will shuffle a given data by random factor in order to
    mix classes for get realistics Test and Training set (see other defs).
    '''
    
    # Set Constants
    SEPARATOR = '*'*25
    
    print('\n{} Shuffling Data {}'.format(SEPARATOR, SEPARATOR)) 
    return random.sample(data, len(data))

def create_placeholders(n_x, n_y):
    '''
    Creates the placeholders for Tensorflow session.
    
    Args:
        n_x : scalar, size of image vector (num_px × num_px = 134 × 134 = 17956*)
        n_y : scalar, number of classes (from 0 to 0, so --> 1 #Only 1 Class)
        
    Returns:
        X : placeholder for the data input, of shape [n_x, None] and type <float>
        Y : placeholder for the input labels, of shape [n_y, None] and dtype <float>
        
    Tips:
        - We will use None because it lets us be flexible on the number of 
          examples we will for the placeholders.
    '''
    
    X = tf.placeholder(tf.float32, [n_x, None], name = 'X')
    Y = tf.placeholder(tf.float32, [n_y, None], name = 'Y')
    
    return X, Y

def initialize_parameters():
    '''
    Initialize parameters to build a ANN with Tensorflow. The shapes are:
        W1 : [30, 134 × 134 = 17956]*
        b1 : [30, 1]
        W2 : [15, 30]
        b2 : [15, 1]
        W3 : [7, 15]
        b3 : [7, 1]
        W4 : [1, 1]
        b4 : [1, 1]
        
    Returns:
        parameters - a dictionary of tensors containing W1, b1, W2, b2, W3, b3,
                     W4, b4.
    '''
    
    # Define some constants before
    SEED = 19890528
    
    tf.set_random_seed(SEED)
    
    W1 = tf.get_variable('W1', [30, 4489], initializer=tf.contrib.layers.xavier_initializer(seed=SEED))
    b1 = tf.get_variable('b1', [30, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [15, 30], initializer=tf.contrib.layers.xavier_initializer(seed=SEED))
    b2 = tf.get_variable('b2', [15, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [7, 15], initializer=tf.contrib.layers.xavier_initializer(seed=SEED))
    b3 = tf.get_variable('b3', [7, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable('W4', [1, 7], initializer=tf.contrib.layers.xavier_initializer(seed=SEED))
    b4 = tf.get_variable('b4', [1, 1], initializer=tf.zeros_initializer())
    
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3,
                  'W4': W4,
                  'b4': b4
            }

    return parameters

def forward_propagation(X, parameters):
    '''
    Implements the forward propagation for the model:
        LINEAR >> RELU >> LINEAR >> RELU >> LINEAR >> RELU >> LINEAR >> RELU >> LINEAR >> SIGMOID
    
    Arguments:
        X – Input dataset placeholder, of shape (input size, number of examples)
        parameters - Python dictionary containing parameters: W1, b1 – W4, b4
                     the shapes are given in initialize_parameters.
                     
    Returns:
        Z4 - the output of the last linear unit
    '''
    
    # Retrieve the parameters from the dictionary 'parameters'
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    return Z4

def compute_cost(Z4, Y):
    '''
    Compute the cost for the model.
    
    Arguments:
        Z4 - output of forward propagation (output of the last LINEAR unit), of shape (1, number_of_examples)
        Y - 'true' label vector placeholder, same as Z4
        
    Return:
        cost - Tensor of the Cost function
    '''
    
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if non-malaria, 1 if malaria), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1000, minibatch_size=512, print_cost=True):

    """
    Implements a four-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()
    tf.set_random_seed(19890528)
    seed = 55
    (n_x, m) = X_train.shape
    n_y =  Y_train.shape[0]
    costs = []
    
    # Create Placeholders of shape(n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in Tensorflow Graph
    Z4 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow
    cost = compute_cost(Z4, Y)
    
    # Backpropagation: Define the Tensorflow optimizer. I will use AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start the session to compute the Tensorflow graph
    with tf.Session() as sess:
        
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            #print('X_train: {}, Y_train: {}'.format(X_train.shape, Y_train.shape))
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the 'optimizer' and the 'cost'
                # the feedict should contain a minibatch for (X, Y)
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X,
                                                        Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
                
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
            
        # Plot the Cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iteration (per tens)')
        plt.title('Learning rate = {}'.format(str(learning_rate)))
        plt.ylim((0, 1))
        plt.show()
            
        # Save parameters in a variable
        parameters = sess.run(parameters)
        print('Parameters have been trained!')
            
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
            
        # Calculate accuracy on the Test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            
    return parameters        
    
def plot_thumb(data):
    
    '''
    This procedure will plot all thumbs. bu a given data (List of Numpy
    Array: [[WIDTH, HEIGHT], [WIDTH, HEIGHT], ...] with labels)
    '''
    random_index = randint(0, len(data)-1)
    print('Random Index: {}'.format(random_index))
    image_to_show = data[random_index]['Values']
    plt.imshow(image_to_show, cmap='hot', interpolation='nearest')
    plt.show()
    
    class_to_show = data[random_index]['Class']
    print('Selected Class: {}'.format(class_to_show))
    
    return None

def vectorizing_data(data):
    '''
    This function will construct vector X from a given data by performing
    reshaping technique (stacking values vertically) on a dictionary.
    '''

    # Set Constants
    SEPARATOR = '*'*25
    
    # Vectorizing data X
    # Set variables
    l_X = []     # List of vectorized values X
    l_y = []     # List of vectorized values y
    
    print('\n{} Stacking Verticaly (X) and (y) {}'.format(SEPARATOR, SEPARATOR))    
    
    for this_example in range(0, len(data), 1):
        vectorized_data_X = data[this_example]['Values'].ravel()
        vectorized_data_y = int(data[this_example]['Class'])
        l_X.append(vectorized_data_X)
        l_y.append(vectorized_data_y)
        
    X = np.stack(l_X, axis=1)
    y = np.reshape(np.hstack(np.array(l_y)), (-1, 1)).T
    
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))
    
    return X, y

def split_train_test_sets(X, y, ratio):
    '''  
    This function will split a given data (vectorized data) to train and test
    sets based on given ratio.
    '''

    # Set Constants
    SEPARATOR = '*'*25

    print('\n{} Train and Test sets {}'.format(SEPARATOR, SEPARATOR)) 
    
    X_split = np.array_split(X, [int(X.shape[1]*ratio)], axis=1)
    X_train = X_split[0]
    X_test = X_split[1]
    
    y_train = np.reshape(np.hstack(np.array(y[0][:int(len(y[0])*ratio)])), (-1, 1)).T
    y_test = np.reshape(np.hstack(np.array(y[0][int(len(y[0])*ratio):])), (-1, 1)).T
    
    print('| Train and Test sets ratio is {}%.'.format(ratio*100))
    print('| X_train shape: {}'.format(X_train.shape))
    print('| X_test shape: {}'.format(X_test.shape))
    print('| y_train shape: {}'.format(y_train.shape))
    print('| y_test shape: {}'.format(y_test.shape))

    return X_train, X_test, y_train, y_test

# Set Randomize Factor`
np.random.seed(19890528)

# # # # # Plan of Attack # # # # #
# | Set Working Directory
change_directory(project_directory)

# | Read Train and Test Data
my_data = read_data()

# | Plot A Random Thumb
plot_thumb(my_data)

# | Shuffle The Data
my_data = shuffle_data(my_data)

# | Vectorizing The Data to Vector X
X, y = vectorizing_data(my_data)

# | Split Dataset To Train and Test Sets
TRAIN_SET_RATIO = 0.65
X_train, X_test, y_train, y_test = split_train_test_sets(X, y, ratio=TRAIN_SET_RATIO)

# | Normalize Train and Test sets
print('\n| Normalizing Train set')
X_train =  X_train/255
X_test = X_test/255 

# | Perform the Model
parameters = model(X_train, y_train, X_test, y_test)

'''
© Prepared by Vytautas Bielinskas, 2019
vytautas.bielinskas@gmail.com
https://www.linkedin.com/in/bielinskas/
'''