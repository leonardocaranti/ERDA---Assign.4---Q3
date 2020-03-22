import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import callbacks
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Input dataset, with planes 1 - 10 ordered by MTOW
A = np.array([[26.00,31.68,9.86,850,3300,88,36500,2,2,17],
              [28.72,36.24,10.55,850,3300,100,45000,2,2,32],
              [35.80,33.62,12.60,850,3500,142,65317,3,2,1],
              [35.80,42.12,12.60,850,4300,188,76900,3,2,5],
              [60.10,62.80,16.30,920,11500,294,252650,6,2,13],
              [60.10,68.30,17.02,903,12000,344,254100,8,2,8],
              [60.30,58.37,17.40,880,8800,268,233000,7,2,8], 
              [60.30,63.69,16.80,880,8200,292,233000,8,2,5], 
              [60.90,63.80,18.50,900,11800,320,297500,5,2,15],
              [64.44,70.67,19.40,920,11500,408,390100,9,4,5]])

f = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

# Alternative dataset, excluding last column (number of planes which KLM owns)
A_2 = np.array([[26.00,31.68,9.86,850,3300,88,36500,2,2],
              [28.72,36.24,10.55,850,3300,100,45000,2,2],
              [35.80,33.62,12.60,850,3500,142,65317,3,2],
              [35.80,42.12,12.60,850,4300,188,76900,3,2],
              [60.10,62.80,16.30,920,11500,294,252650,6,2],
              [60.10,68.30,17.02,903,12000,344,254100,8,2],
              [60.30,58.37,17.40,880,8800,268,233000,7,2], 
              [60.30,63.69,16.80,880,8200,292,233000,8,2], 
              [60.90,63.80,18.50,900,11800,320,297500,5,2],
              [64.44,70.67,19.40,920,11500,408,390100,9,4]])


# ---------------------- Regression model -------------------------


# * PARAMETER DEFINITION * (you can change!)

# Learning rate: how quickly the weights are adjusted. 
# close to one: adjusted quickly and allows for large variations between each iteration
# close to zero: adjusted much slower and small fluctuations per iteration
lr = 0.001 

# Epoch number and batch size: given a part of the dataset (batch) of definite size (batch size), 
# the algorithm performs a number of iterations (epochs) where he adjusts the weights of the neurons. 
epochss_ = 12000
batch_sz = 32

# Neurons: the size of each hidden layer. You can leave neurons_2 there even if you are only using
# 1 hidden layer.
neurons_1 = 10
neurons_2 = 8

# * NEURAL NETWORK MODEL DEFITION *
def baseline_model():
    # Model architecture. Comment out the neuron_2 line to only have 1 hidden layer. 
    # As this is a regression model the last neuron must have a linear activation function. The
    # other layers can be adjusted to have other ones (I like linear so sometimes I put lienar in everything, 
    # but others use 'relu', 'sigmoid', 'softmax', ...)
    model = Sequential()
    model.add(Dense(neurons_1, input_dim=10, kernel_initializer='normal', activation='linear'))
    #model.add(Dense(neurons_2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Model compilation: choose the loss function to evaluate the error and the method of optimising. 
    # Usually regression models use gradient descent which is SGD here (which takes the derivative of the loss function, 
    # like Dwight explained in class), but somehow this Adam one works way better for me! 
    optimiz = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mean_squared_error', optimizer=optimiz)
    return model

# * EVALUATION USING REGRESSION *
MSEs = [] # Used to append mean squared error at the end, just to check
cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto') # Makes sure to stop the evaluations if the error fluctuates too much
estimator = KerasRegressor(build_fn=baseline_model, epochs=epochss_, batch_size=batch_sz, verbose=2) # Change verbose to 0, 1 or 2 to choose what you want to see printed while the model evaluates
kfold = KFold(n_splits=9) #Splits the dataset in a section of n_splits for training and 10-n_splits for evaluation of the model
results = cross_val_score(estimator, A, f, cv=kfold, fit_params={'callbacks': [cb]}) # Evaluates model 
#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
MSEs.append([results.mean(), results.std()])

# * ESTIMATE On *
estimator.fit(A, f) # Fit the original dataset once again for the optimised model defined above
prediction = estimator.predict(A) # This outputs On
print(np.array(prediction)) # On
print(f) # What On should be 
print("Learning rate: ", lr)
print("Epochs: ", epochss_)
print("Batch size: ", batch_sz)
print("Number of neurons layer 1: ", neurons_1)

# Calculate MSE (I wanted to calulculate it myself to check!)
MSE = 0
for i in range(len(prediction)):
    MSE += (prediction[i]-f[i][0])*(prediction[i]-f[i][0])

print("Mean squared error: ", MSE)
print(MSEs) # I don't know how to read this Mean Squared error number. The second one is supposed to be Standard Deviation. 
