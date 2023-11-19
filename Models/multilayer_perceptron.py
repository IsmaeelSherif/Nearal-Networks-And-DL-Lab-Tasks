import math
import numpy as np





class MulilayerPerceptron() :
     
    def __init__( self, hasBias, learning_rate, epochs, layers, activation ) :
        self.hasBias = hasBias
        self.learning_rate = learning_rate
        self.epochs = epochs
        # the number of weights is equal to the num of neurons in the previous layer
        self.neurons = []
        for i in range(1, len(layers)):
            weightsNum = layers[i-1]
            layerNeuronsNum = layers[i]
            thisLayerNeurons = [None] * layerNeuronsNum
            for j in range(layerNeuronsNum):
                neuron = self._Neuron(self, weightsNum, activation)
                thisLayerNeurons[j] = neuron
            self.neurons.append(thisLayerNeurons)

        print('architecture layers', len(self.neurons))
        for layer in self.neurons:
            print('layer with', len(layer), 'numOfWeights', len(layer[0].weights))
        for i in range(len(self.neurons)):
            layer = self.neurons[i]
            for j in range(len(layer)):
                neuron = layer[j]
                if i == 0 and j == 0:
                    neuron.weights = [-0.3, 0.21, 0.15]
                if i == 0 and j == 1:
                    neuron.weights = [0.25, -0.4, 0.1]
                if i == 1:
                    neuron.weights = [-0.4, -0.2, 0.3]


    def train(self, input_data, output_data):
        X = np.array(input_data)
        y = np.array(output_data)
        classes_num = np.unique(y).size
        self.classes_num = classes_num

        y_multiClass = [self.__transformY(y, positiveClassIndex=index) for index in range(classes_num-1, -1, -1)]
        y_multiClass = np.array(y_multiClass)


        sampleSize = X.shape[0]

        for epoch_num in range(self.epochs):

            error = 0

            for i in range(sampleSize):
                layersOutputs = self._forward(X[i])
                # print(layersOutputs)

                error += self._backward(layersOutputs, y_multiClass, i)

                # break # to just make 1 sample of x

            print('epoch', epoch_num, 'error', error)

        
        for i in range(len(self.neurons)):
            layer = self.neurons[i]
            for j in range(len(layer)):
                neuron = layer[j]
                print('neuron', i, j, 'weights', neuron.weights)


    def _forward(self, xi):
        previousLayerOutput = xi
        layersOutput = [
            xi.tolist()
        ]
        for layer in self.neurons:
            neuronsNum = len(layer)
            layersOutput.append([0] * neuronsNum)
            
        
        for layerIndex in range(len(self.neurons)):
            layer = self.neurons[layerIndex]
            
            for neuronIndex in range(len(layer)):
            #    print('epoch', epoch_num, 'i', i, 'layer', layerIndex, 'neuronIndex', neuronIndex)
                previousLayerOutput = layersOutput[layerIndex] # no -1, bec it is 0-based
                layersOutput[layerIndex + 1][neuronIndex] = layer[neuronIndex].forward(previousLayerOutput)
        return layersOutput
                


    def _backward(self, layersOutputs, y_multiClass, sampleIndex):

        error = 0
        layersGradients = []
        for layer in self.neurons:
            neuronsNum = len(layer)
            layersGradients.append([0] * neuronsNum)

        for layerIndex in range(len(self.neurons)-1, -1, -1):
            layer = self.neurons[layerIndex]

            for neuronIndex in range(len(layer)):
                
                Y_k_plus1 = layersOutputs[layerIndex + 1][neuronIndex]
                f_dash = Y_k_plus1 * (1 - Y_k_plus1)
                isLastLayer = layerIndex == len(self.neurons) - 1
                if isLastLayer:
                    terminal_gradient = y_multiClass[neuronIndex][sampleIndex] - Y_k_plus1
                    error = terminal_gradient
                    # print('y', y_multiClass[neuronIndex][sampleIndex], 'terminal_gradient', terminal_gradient)
                else:
                    terminal_gradient = 0
                    nextLayer = self.neurons[layerIndex + 1]
                    for nextLayerNeuronIndex in range(len(nextLayer)):
                        weightIndex = neuronIndex + 1 if self.hasBias else neuronIndex
                        weight = nextLayer[nextLayerNeuronIndex].weights[weightIndex]
                        terminal_neuron_gradient = layersGradients[layerIndex + 1][nextLayerNeuronIndex]
                        terminal_gradient += terminal_neuron_gradient * weight


                # print('Y_k_plus1', Y_k_plus1, 'f_dash', f_dash, 'terminal_gradient', terminal_gradient, 'gradient', gradient)
                gradient = f_dash * terminal_gradient
                layersGradients[layerIndex][neuronIndex] = gradient
                neuron = layer[neuronIndex]
                neuron.backward(gradient, layersOutputs[layerIndex])
                

        return error



    def __transformY(self, y, positiveClassIndex):
        y = np.array(y)
        for i in range(len(y)):
            y[i] = 1 if y[i] == positiveClassIndex else 0
        return y




    class _Neuron():
        def __init__( self, network_instance, weightsNum, activation) :
            if(activation == 'sigmoid'):
                self.activationFun = self._sigmoid
            elif(activation == 'tanh'):
                self.activationFun = self._tanh
            else:
                raise('please enter a valid activation: [sigmoid, tanh]')
            
            r = np.random.RandomState(0)
            self.weights = r.random(weightsNum + int(network_instance.hasBias))
            self.network_instance = network_instance
    
    
        def forward(self, x):
            if self.network_instance.hasBias:
                x = self._addBiasToX(x)
                
            ypred = np.dot(x, self.weights)
            activation = self.activationFun(ypred)
            return activation
        
        def backward(self, grdient, x):
            if self.network_instance.hasBias:
                x = self._addBiasToX(x)
    
            self.weights += grdient * self.network_instance.learning_rate * x
            # print('backward', x, 'updated', self.weights)
    
    
        def _sigmoid(self, z):
            return 1 / (1 + math.exp(-z))
        
        def _tanh(self, z):
            expZ = math.exp(z)
            _expZ = math.exp(-z)
            return (expZ - _expZ) / (expZ + _expZ)
        
        def _addBiasToX(self, x):
            # add a 1 at the begining
            return np.insert(x, 0, 1)

    
    




