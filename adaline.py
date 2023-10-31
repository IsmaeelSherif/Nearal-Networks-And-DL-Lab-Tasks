import numpy as np


class Adaline() :
     
    def __init__( self, hasBias, learning_rate, epochs, mse_threshold ) :
        self.hasBias = hasBias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mse_threshold = mse_threshold

    def __addBiasToX(self, X):
        # add a column of 1's at the begining
        return np.c_[np.ones((X.shape[0], 1)), X]
    
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
            errors = [0] * classes_num
            for i in range(sampleSize):
                xi = X[i]
                for class_index in range(classes_num):
                    yi = y_multiClass[class_index][i]
                    class_coeff = coeffs[class_index]
                    ypred = np.dot(xi, class_coeff)

                    error = yi - ypred
                    

                    if error != 0:
                        errors[class_index] += pow(error, 2)
                        coeffs[class_index] = class_coeff + (self.learning_rate * error * xi)
                    
                    # if epoch_num == 0 and class_index == 1:
                    #     print('yi=', yi, 'ypred', ypred,'error', error, 'coeff', coeffs[class_index])
                    
            # print('epoch', epoch_num, errors)
            if(all(classMSE <= self.mse_threshold for classMSE in errors)):
                print('Stopping at', epoch_num, 'MSE Threshold has been reached')
                break
    
        self.weights = coeffs


    def predict( self, X ) :

        def actFun(Z):
            M = [0] * len(Z)
            for i in range(len(Z)):
                M[i] = -1 if Z[i] < 0 else 1
            return M
            

        if self.hasBias:
            X = self.__addBiasToX(X)

        sample_size = X.shape[0]
        
        classesPredictions = [[0] * sample_size] * self.classes_num
        classesPredictions = np.array(classesPredictions)

        for class_index in range(self.classes_num):
            Z = np.dot( X, self.weights[class_index] )
            classesPredictions[class_index] = actFun(Z)
        # print(classesPredictions)
            
        finalPredictions = [0] * sample_size
        for i in range(sample_size):
            for class_index in range(self.classes_num):
                if(classesPredictions[class_index][i] == 1):
                    finalPredictions[i] = class_index
                    break

        return finalPredictions