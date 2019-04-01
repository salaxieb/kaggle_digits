import tensorflow
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import matplotlib.pyplot as plt
import scipy
from scipy import misc

print("imported")

import numpy as np

def Rotate_to_vertical(Pic):
#     plt.imshow(Pic, cmap='gray')
#     plt.show()
    angle = 10
    flag_optimal = False
    Horizontal_Sum = Pic.sum (axis=0)
    current_width = np.count_nonzero(Horizontal_Sum)
    #print (current_width)
    while flag_optimal == False:
        rotated_Pic = scipy.misc.imrotate(Pic, angle, interp='bilinear')
        Horizontal_Sum = rotated_Pic.sum(axis=0)
        rotated_width = np.count_nonzero(Horizontal_Sum)
        #print (rotated_width)
        if rotated_width >= current_width:
            flag_optimal = True
        else:
            Pic = rotated_Pic
            current_width = rotated_width
            
    angle = -10
    flag_optimal = False
    Horizontal_Sum = Pic.sum (axis=0)
    current_width = np.count_nonzero(Horizontal_Sum)
    #print (current_width)
    while flag_optimal == False:
        rotated_Pic = scipy.misc.imrotate(Pic, angle, interp='bilinear')
        Horizontal_Sum = rotated_Pic.sum(axis=0)
        rotated_width = np.count_nonzero(Horizontal_Sum)
        #print (rotated_width)
        if rotated_width >= current_width:
            flag_optimal = True
        else:
            Pic = rotated_Pic
            current_width = rotated_width
    
    
    return Pic

def RotateX (X, num_pics):
    Rotated_X = np.empty((num_pics,28,28))
    j = 0
    for Pic in X:
        Pic = Pic.copy()
        
#         plt.imshow(Pic, cmap='gray')
#         plt.show()
        size_X = np.size(X,2)
        size_Y = np.size(X,1)
        A = Pic > np.ones((size_Y, size_X))*150;    
        Pic = Pic * A
        
        #plt.imshow(Pic, cmap='gray')
        #plt.show()
        Pic = Rotate_to_vertical(Pic)
        
        Rotated_X[j:j+1,:,:] = Pic
        if j % 1000 == 1:
            print ((j-1)/1000)
        j += 1
    
    return Rotated_X

def NormalX (X, num_pics):
    Normilized_X = np.empty((num_pics,28,28))
    j = 0
    for Pic in X:
        Pic = Pic.copy()
        
#         plt.imshow(Pic, cmap='gray')
#         plt.show()
        size_X = np.size(X,2)
        size_Y = np.size(X,1)
        A = Pic > np.ones((size_Y, size_X))*150;    
        Pic = Pic * A
        
        #plt.imshow(Pic, cmap='gray')
        #plt.show()
        Pic = Rotate_to_vertical(Pic)
        
        size_X = np.size(X,2)
        size_Y = np.size(X,1)
        
        A = Pic > np.ones((size_Y, size_X))*50;    
        Pic = Pic * A
        #plt.imshow(Pic, cmap='gray')
        #plt.show()
        
        Horizontal_Sum = Pic.sum(axis=0)
        Vertical_Sum = Pic.sum(axis=1)
        Front_Zeros = size_X - np.trim_zeros(Horizontal_Sum, 'f').size
        Back_Zeros = size_Y - np.trim_zeros(Horizontal_Sum, 'b').size
        Top_Zeros = size_X - np.trim_zeros(Vertical_Sum, 'f').size
        Bottom_Zeros = size_Y - np.trim_zeros(Vertical_Sum, 'b').size
        
        Pic = Pic[Top_Zeros:size_Y-Bottom_Zeros + 1, Front_Zeros:size_X-Back_Zeros + 1]
        Zeros_Hor = Front_Zeros + Back_Zeros
        Zeros_Ver = Top_Zeros + Bottom_Zeros
        
        if Zeros_Ver > 0:
            step = (size_X - Zeros_Ver - 2)/(Zeros_Ver + 1)
            for i in range(Zeros_Ver - 1):
                i_row = round((i+1)*step)+i+1
                Mid_Value = np.add(Pic[i_row:i_row+1,:],Pic[i_row+1:i_row+2,:])/2
                Pic = np.concatenate((Pic[0:i_row+1,:], Mid_Value, Pic[i_row+1:,:]), axis=0)
            
        if Zeros_Hor > 0:
            step = (size_Y - Zeros_Hor - 2)/(Zeros_Hor + 1)
            for i in range(Zeros_Hor - 1):
                i_row = round((i+1)*step)+i+1
                Mid_Value = np.add(Pic[:,i_row:i_row+1],Pic[:,i_row+1:i_row+2])/2
                Pic = np.concatenate((Pic[:,0:i_row+1], Mid_Value, Pic[:,i_row+1:]), axis=1)
        
#         plt.imshow(Pic, cmap='gray')
#         plt.show()
        #print (np.size(Normilized_X, axis = 1))
        #print (np.size(Pic, axis = 1))
        while np.size(Pic, axis = 0) < 28:
            Pic = np.concatenate((Pic, np.zeros((1,28))), axis = 0)
        while np.size(Pic, axis = 1) < 28:
            Pic = np.concatenate((Pic, np.zeros((28,1))), axis = 1)
            
#         Pic = Pic > np.ones((size_Y, size_X))*0.2;
#         Pic[Pic < 50] = 0
#         Pic[Pic >= 50] = 255
#         print (Pic)
#         plt.imshow(Pic, cmap='gray')
#         plt.show()
        Normilized_X[j:j+1,:,:] = Pic
        if j % 1000 == 1:
            print ((j-1)/1000)
        j += 1
    #print (Normilized_X[26000,:,:])
    return Normilized_X

batch_size = 128 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
#hidden_size = 15 # there will be 512 neurons in both hidden layers


height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)

(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

num_train = 5000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

X_train = X_train[:num_train].copy()
y_train = y_train[:num_train].copy()



X_train_N = NormalX (X_train, num_train)
X_test_N = NormalX (X_test, 10000)

X_train_R = RotateX (X_train, num_train)
X_test_R = RotateX (X_test, 10000)

X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

X_train_N /= 255 # Normalise data to [0, 1] range
X_test_N /= 255 # Normalise data to [0, 1] range

X_train_R /= 255 # Normalise data to [0, 1] range
X_test_R /= 255 # Normalise data to [0, 1] range

X_train = X_train.reshape(num_train, height * width) # Flatten data to 1D
X_test = X_test.reshape(num_test, height * width) # Flatten data to 1D

X_train_R = X_train_R.reshape(num_train, height * width) # Flatten data to 1D
X_test_R = X_test_R.reshape(num_test, height * width) # Flatten data to 1D

X_train_N = X_train_N.reshape(num_train, height * width) # Flatten data to 1D
X_test_N = X_test_N.reshape(num_test, height * width) # Flatten data to 1D
    
Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
#resize_Pic(X);
def evaluate_data (X_train, X_test, hidden_size):      
    inp = Input(shape=(height * width,)) # Our input is a 1D vector of size 784
    hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
#     hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
    out = Dense(num_classes, activation='softmax')(hidden_1) # Output softmax layer
    
    model = Model(input=inp, output=out) # To define a model, just specify its input and output layers
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy
    
    model.fit(X_train, Y_train, # Train the model using the training set...
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
    
    print (model.evaluate(X_test, Y_test, verbose=1)) # Evaluate the trained model on the test set!
    return model.evaluate(X_test, Y_test, verbose=1)[1]

results = []
results_n = []
results_r = []
horizontal = []

for iter in range(10, 30):
#     neurons_nb = iter**2
    neurons_nb = iter
    horizontal.append(neurons_nb)
    results.append(evaluate_data(X_train, X_test, neurons_nb))
    results_n.append(evaluate_data(X_train_N, X_test_N, neurons_nb))
    results_r.append(evaluate_data(X_train_R, X_test_R, neurons_nb))

fig, ax = plt.subplots()
ax.plot(horizontal, np.divide(results,results), label="baseline")
ax.plot(horizontal, np.divide(results_n,results), label="results with rotating and stretching")
ax.plot(horizontal, np.divide(results_r,results), label="results with rotating")
ax.set(xlabel='hidden layes size', ylabel='accuracy',
       title='mnist data accuracy results without and with preprocessing')
ax.legend()
ax.grid()

plt.show()