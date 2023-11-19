import numpy as np


class Perceptron() :
     
    def __init__( self, hasBias, learning_rate, epochs ) :
        self.hasBias = hasBias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __addBiasToX(self, X):
        # add a column of 1's at the begining
        return np.c_[np.ones((X.shape[0], 1)), X]

    def __activationFun(self, Z):
        def actFun(z):
            if z < 0:
                return -1
            else:
                return 1
        if isinstance(Z, float):
            return actFun(Z)
        else:
            for i in range(len(Z)):
                Z[i] = actFun(Z[i])
            return Z
        
    def __transformY(self, y, positiveClassIndex):
        y = np.array(y)
        for i in range(len(y)):
            y[i] = 1 if y[i] == positiveClassIndex else -1
        return y
        
    def train(self, input_data, output_data):
        X = np.array(input_data)
        y = np.array(output_data)
        classes_num = np.unique(y).size
        self.classes_num = classes_num

        y_multiClass = [self.__transformY(y, positiveClassIndex=index) for index in range(classes_num)]
        y_multiClass = np.array(y_multiClass)

        sampleSize = X.shape[0]

        r = np.random.RandomState(0)
        coeffs=r.random(X.shape[1])

        if self.hasBias:
            X = self.__addBiasToX(X)
            bias=r.random_sample()
            coeffs = np.concatenate(([bias], coeffs))
        
        coeffs = np.tile(coeffs, (classes_num, 1))

        for epoch_num in range(self.epochs):
            foundError = [False] * classes_num
            errors = [0] * classes_num
            for i in range(sampleSize):
                xi = X[i]
                for class_index in range(classes_num):
                    yi = y_multiClass[class_index][i]
                    class_coeff = coeffs[class_index]
                    ypred = np.dot(xi, class_coeff)
                    activation = self.__activationFun(ypred)

                    error = yi - activation
                    

                    if error != 0:
                        errors[class_index] += 1
                        foundError[class_index] = True
                        coeffs[class_index] = class_coeff + (self.learning_rate * error * xi)
                    
                    # if epoch_num == 0 and class_index == 1:
                    #     print('yi=', yi, 'ypred', ypred, 'activation', activation,'error', error, 'coeff', coeffs[class_index])
                # print('epoch', epoch_num, coeffs)
            # print('epoch', epoch_num, errors)
            if(True not in foundError):
                print('Stopping at', epoch_num, 'convergence has been reached, Error = 0')
                break
    
        self.weights = coeffs


    def predict( self, X ) :
        if self.hasBias:
            X = self.__addBiasToX(X)

        sample_size = X.shape[0]
        
        classesPredictions = [[0] * sample_size] * self.classes_num
        classesPredictions = np.array(classesPredictions)

        for class_index in range(self.classes_num):
            Z = np.dot( X, self.weights[class_index] )
            classesPredictions[class_index] = self.__activationFun(Z)
            
        finalPredictions = [0] * sample_size
        for i in range(sample_size):
            for class_index in range(self.classes_num):
                if(classesPredictions[class_index][i] == 1):
                    finalPredictions[i] = class_index
                    break

        return finalPredictions