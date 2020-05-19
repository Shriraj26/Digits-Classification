import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("train.csv")

X = dataframe.iloc[:,1:].values
y = dataframe.iloc[:,0].values


#train data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 



#Ann
import keras
from keras.models  import Sequential
from keras.layers import Dense

#Initialization of Ann
ann_class = Sequential()

#First input layer and hidden layer
ann_class.add(Dense(397, input_dim=784, kernel_initializer='uniform', activation='relu'))

#Second hidden layer
ann_class.add(Dense(397, kernel_initializer='uniform', activation='relu'))

#Output Layer
ann_class.add(Dense(10, kernel_initializer='uniform', activation='softmax'))

#Compiling the ann - Applying the stochastic grad descent
ann_class.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

#Training The Ann
ann_class.fit(X_train,y_train, batch_size = 10, epochs = 100)


#plotting the data 
d = X[2]
d.shape = (28,28)
plt.imshow(255-d, cmap = 'gray')
plt.show()
print(classifier.predict([X[2]]))




count = 0

for i in range (0,8400):
    if classifier.predict([X_test[i]]) == y_test[i]:
            count = count + 1
print (count/8400*100)

from sklearn import metrics
acc = metrics.accuracy_score(prediction,y_test)


dataframe_test = pd.read_csv("test.csv")
X_test_pred = dataframe_test.iloc[:,0:].values
y_test_pred = classifier.predict(X_test_pred)

#plotting the data 
d = X_test_pred[1]
d.shape = (28,28)
plt.imshow(255-d, cmap = 'gray')
plt.show()
print(classifier.predict([X_test_pred[1]]))

