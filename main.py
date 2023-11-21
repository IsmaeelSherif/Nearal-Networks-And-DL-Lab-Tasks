import pandas as pd
from label_encoder import LabelEncoder
from min_max_normalizer import MinMaxNormalizer

from main_ui import startUI




df = pd.read_excel('Dry_Bean_Dataset.xlsx')

indexes_with_nan = df[df['MinorAxisLength'].isnull()].index

for rowIndex in indexes_with_nan:
    valueBefore = df['MinorAxisLength'][rowIndex - 1]
    valueAfter = df['MinorAxisLength'][rowIndex + 1]
    mean = (valueAfter + valueBefore) / 2
    df.at[rowIndex, 'MinorAxisLength'] = mean

labelEndcoder = LabelEncoder(df['Class'])


allBombay = df[df['Class'] == 'BOMBAY']
allCali = df[df['Class'] == 'CALI']
allSira = df[df['Class'] == 'SIRA']

splitRatio = 0.6
trainBombay = allBombay.sample(frac=splitRatio, random_state=0)
testBombay = allBombay.drop(trainBombay.index)

trainCali = allCali.sample(frac=splitRatio, random_state=0)
testCali = allCali.drop(trainCali.index)

trainSira = allSira.sample(frac=splitRatio, random_state=0)
testSira = allSira.drop(trainSira.index)

trainDf = pd.concat([trainBombay, trainCali, trainSira]).sample(frac=1, random_state=0) #shuffled
testDf = pd.concat([testBombay, testCali, testSira]).sample(frac=1, random_state=0) #shuffled

trainDf['Class'] = trainDf['Class'].apply(lambda x: labelEndcoder.encode(x))
testDf['Class'] = testDf['Class'].apply(lambda x: labelEndcoder.encode(x))

y_train = trainDf['Class']
X_train = trainDf.drop(columns='Class')
y_test = testDf['Class']
X_test = testDf.drop(columns='Class')


# startUI(featureNames=X_train.columns.to_list(), classNames=df['Class'].unique())


areaNormalizer = MinMaxNormalizer()
perimNormalizer = MinMaxNormalizer()
majorNormalizer = MinMaxNormalizer()
minorNormalizer = MinMaxNormalizer()

X_train['Area'] = areaNormalizer.fit_transform(X_train['Area'])
X_train['Perimeter'] = perimNormalizer.fit_transform(X_train['Perimeter'])
X_train['MajorAxisLength'] = majorNormalizer.fit_transform(X_train['MajorAxisLength'])
X_train['MinorAxisLength'] = minorNormalizer.fit_transform(X_train['MinorAxisLength'])

X_test['Area'] = areaNormalizer.transform(X_test['Area'])
X_test['Perimeter'] = perimNormalizer.transform(X_test['Perimeter'])
X_test['MajorAxisLength'] = majorNormalizer.transform(X_test['MajorAxisLength'])
X_test['MinorAxisLength'] = minorNormalizer.transform(X_test['MinorAxisLength'])


# from Models.perceptron import Perceptron
# model = Perceptron(hasBias=True, learning_rate= 0.1, epochs=1000)
# model.train(X_train, y_train)
# pred = model.predict(X_test)


# correct = 0
# import numpy as np
# y_test = np.array(y_test)

# for i in range(len(y_test)):
#     print('y_test', y_test[i], 'pred', pred[i])
#     if(y_test[i] == pred[i]):
#         correct += 1
# print('accur', correct/len(y_test))



# from Models.adaline import Adaline
# model = Adaline(hasBias=True, learning_rate=0.1, epochs=1000, mse_threshold=1)

# model.train(X_train, y_train)
# pred = model.predict(X_test)

# correct = 0
# import numpy as np
# y_test = np.array(y_test)

# for i in range(len(y_test)):
#     print('y_test', y_test[i], 'pred', pred[i])
#     if(y_test[i] == pred[i]):
#         correct += 1
# print('accur', correct/len(y_test))

from Models.multilayer_perceptron import MulilayerPerceptron

model = MulilayerPerceptron(hasBias=True, learning_rate=0.1, epochs=100, layers=[5, 3, 4, 3], activation='sigmoid')

# X_train = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]

# y_train = [0, 1, 1, 0]

model.train(X_train, y_train)