#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset 
dataframe = pd.read_csv("train.csv")


X = dataframe.iloc[:,1:].values
y = dataframe.iloc[:,0].values


#train data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 



"""
Using Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators  = 100,criterion  = 'entropy', random_state= 0)
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)




#plotting the data 
d = X[2]
d.shape = (28,28)
plt.imshow(255-d, cmap = 'gray')
plt.show()
print(classifier.predict([X[2]]))



#Accuracy of our result
from sklearn import metrics
acc = metrics.accuracy_score(prediction,y_test)*100

