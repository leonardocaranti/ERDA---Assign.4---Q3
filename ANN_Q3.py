from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

A = np.array([[60.30,58.37,17.40,880,8800,268,233000,7,2,8], 
              [60.30,63.69,16.80,880,8200,292,233000,8,2,5], 
              [35.80,33.62,12.60,850,3500,142,65317,3,2,1], 
              [35.80,42.12,12.60,850,4300,188,76900,3,2,5],
              [64.44,70.67,19.40,920,11500,408,390100,9,4,5],
              [60.90,63.80,18.50,900,11800,320,297500,5,2,15], 
              [60.10,62.80,16.30,920,11500,294,252650,6,2,13], 
              [60.10,68.30,17.02,903,12000,344,254100,8,2,8], 
              [26.00,31.68,9.86,850,3300,88,36500,2,2,17],
              [28.72,36.24,10.55,850,3300,100,45000,2,2,32]])

f = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

# Function which generates a number of (num_data_pts)
def create_data(A, f, num_data_pts): #Number of data points must be divisible by 10
    A_new  = np.zeros(shape=(num_data_pts, 10))
    f_new = np.zeros(shape=(num_data_pts, 1))
    for i in range(int(num_data_pts/10)-1):
        #print()
        A_new[i*10:(i+1)*10], f_new[i*10:(i+1)*10] = A, f
        #print(A_new[30])
    return A_new, f_new

data, output = create_data(A, f, 1000)

"""
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
"""

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))  # Change to linear?
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Change to least squares

# fit the keras model on the dataset
model.fit(data, output, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
