import numpy as np
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:

    def __init__(self):

        self.first_weights=[]
        self.second_weights=[]
        self.first_weights.append(2 * np.random.random((10, 1)) - 1)
        self.first_weights.append(2 * np.random.random((10, 1)) - 1)
        self.first_weights.append(2 * np.random.random((10, 1)) - 1)
        self.second_weights.append(2 * np.random.random((3, 1)) - 1)

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):

        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            output_firstlayer=[]
            output_secondlayer=[]
            newfirstweights=[]
            newsecondweights=[]

            for weights in self.first_weights:
                output_firstlayer.append(self.think(training_inputs,weights))

            inputs=np.empty((0,len(output_firstlayer)))
            for weights in self.second_weights:
                for k in range (len(output_firstlayer[0])):
                    temp_input = np.empty((0,len(output_firstlayer)))
                    for h in range(len(output_firstlayer)):
                        temp_input=np.append(temp_input, [output_firstlayer[h][k]])
                    inputs=np.append(inputs,[temp_input],axis=0)
                output_secondlayer.append(self.think(inputs,weights))

            i=0
            for weights in self.second_weights:
                error= np.sqrt((training_outputs-output_secondlayer[i])*(training_outputs-output_secondlayer[i])) #Error calculated using least squares
                if iteration%100 ==0:
                    print("Error = ", error)
                adjustments = np.dot(inputs.T, error * self.sigmoid_derivative(output_secondlayer[i]))
                newsecondweights.append(weights+adjustments)
                i+=1

            i=0
            for weights in self.first_weights:

                error= training_outputs-output_firstlayer[i]
                adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output_firstlayer[i]))
                newfirstweights.append(weights+adjustments)
                i+=1

            self.first_weights=newfirstweights
            self.second_weights=newsecondweights


    def think(self, inputs,weight):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, weight))

        return output

    def guess(self,inputs):

        inputs = inputs.astype(float)
        outputs=np.empty((0,nbr_firstlayer))
        for weights in self.first_weights:
            outputs=np.append(outputs,[self.sigmoid(np.dot(inputs, weights))])
        for weights in self.second_weights:
            results=self.sigmoid(np.dot(outputs,weights))

        return results


# Run the network
neural_network = NeuralNetwork()
"""
print("Random starting synaptic weights: ")
print(neural_network.first_weights)
print(neural_network.second_weights)
"""
nbr_firstlayer=len(neural_network.first_weights)

training_data = np.array([[60.30,58.37,17.40,880,8800,268,233000,7,2,8], 
              [60.30,63.69,16.80,880,8200,292,233000,8,2,5], 
              [35.80,33.62,12.60,850,3500,142,65317,3,2,1], 
              [35.80,42.12,12.60,850,4300,188,76900,3,2,5],
              [64.44,70.67,19.40,920,11500,408,390100,9,4,5],
              [60.90,63.80,18.50,900,11800,320,297500,5,2,15], 
              [60.10,62.80,16.30,920,11500,294,252650,6,2,13], 
              [60.10,68.30,17.02,903,12000,344,254100,8,2,8], 
              [26.00,31.68,9.86,850,3300,88,36500,2,2,17],
              [28.72,36.24,10.55,850,3300,100,45000,2,2,32]])

training_inputs=StandardScaler().fit_transform(training_data)

training_outputs = np.array([[1,2,3,4,5,6,7,8,9,10]]).T

neural_network.train(training_inputs, training_outputs, 1000)

"""
print("Synaptic weights after training: ")
print(neural_network.first_weights)
print(neural_network.second_weights)
"""

# This is the data of plane 1, so if the model is trained correctly it should output a [1.]
data=np.array([[60.30,58.37,17.40,880,8800,268,233000,7,2,8]]) 
training_data=np.append(training_data,data,axis=0)
training_inputs=StandardScaler().fit_transform(training_data)
data=training_inputs[-1]
print(neural_network.guess(data))

