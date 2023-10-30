

class LabelEncoder() :
     
    def __init__( self, column) :
        classLabels = column.unique()
        self.__encodedToLabels = dict(enumerate(classLabels))
        self.__labelToEncode = {v: k for k, v in self.__encodedToLabels.items()}

    def encode(self, label):
        return self.__labelToEncode.get(label)
    
    # def getLabel(self, encodedVal):
    #     return self.classLabels.get(encodedVal)