#SIMPLE LINEAR REGRESSION

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


#Splitting dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)


#feature scaling

from sklearn.preprocessing import StandardScaler
"""sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y=StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)"""

#Fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results

Y_pred = regressor.predict(X_test)


#Visualizing the training set results

plt.scatter(X_train,Y_train, color ='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Years Of Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the test set results

plt.scatter(X_test,Y_test, color ='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Years Of Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()












