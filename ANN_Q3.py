import numpy as np
import random 
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import callbacks
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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

# New model regression
# define base model
lr = 0.0005
epochss_ = 20000
batch_sz = 32
neurons_1 = 10
neurons_2 = 7

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(neurons_1, input_dim=10, kernel_initializer='normal', activation='linear'))
    #model.add(Dense(neurons_2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
	# Compile model
    optimiz = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mean_squared_error', optimizer=optimiz)
    return model

# evaluate model
MSEs = []
cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
estimator = KerasRegressor(build_fn=baseline_model, epochs=epochss_, batch_size=batch_sz, verbose=2)
kfold = KFold(n_splits=8)
results = cross_val_score(estimator, A, f, cv=kfold, fit_params={'callbacks': [cb]})
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
MSEs.append([results.mean(), results.std()])

# Test model
estimator.fit(A, f)
prediction = estimator.predict(A)
#accuracy_score(y, prediction)
print(prediction)
print(f)
print("MES's")
print(MSEs)
