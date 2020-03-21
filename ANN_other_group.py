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
                error= training_outputs-output_secondlayer[i]
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

if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")

    print(neural_network.first_weights)
    print(neural_network.second_weights)
    nbr_firstlayer=len(neural_network.first_weights)

    training_data= np.array([[60.3,58.37,16.8,880,8800,289,233000,7,49500,361.6],
            [60.3,63.69,16.8,880,8200,292,233000,8,51700,361.6],
            [35.8,33.62,12.6,850,3500,142,65317,3,16500,125],
            [35.8,39.47,12.6,850,4200,186,73700,3,20540,125],
            [35.8,42.12,12.6,850,4300,188,76900,3,20240,125],
            [64.44,70.67,19.4,920,11500,408,390100,9,70620,541.2],
            [60.9,63.8,18.5,900,11800,320,297500,5,51250,427.8],
            [64.8,73.86,18.5,920,12000,408,351543,7,68500,436.8],
            [60.1,62.8,16.3,920,11500,294,252650,6,41050,360.5],
            [26,31.68,9.86,850,3300,88,36500,2,9900,72.7]])

    training_inputs=StandardScaler().fit_transform(training_data)

    training_outputs = np.array([[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]).T

    neural_network.train(training_inputs, training_outputs, 100000)

    print("Synaptic weights after training: ")

    print(neural_network.first_weights)
    print(neural_network.second_weights)

    print("New situation: input data = ", A, B, C,D,E,F,G,H,I,J)

    print("Output data: ")

    A = str(input("Input 1: "))

    B = str(input("Input 2: "))

    C = str(input("Input 3: "))

    D = str(input("Input 3: "))

    E = str(input("Input 3: "))

    F = str(input("Input 3: "))

    G = str(input("Input 3: "))

    H = str(input("Input 3: "))

    I = str(input("Input 3: "))

    J = str(input("Input 3: "))

    data=np.array([A, B, C,D,E,F,G,H,I,J])
    training_data=np.append(training_data,data,axis=0)
    training_inputs=StandardScaler().fit_transform(training_data)
    data=training_inputs[-1]
    print(neural_network.guess(data))

