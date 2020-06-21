import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:,:-1].values
Y = datasets.iloc[:,-1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding Categorical data - Independent Variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Encoding Categorical Data - Dependent Variables
from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()
Y = le.fit_transform(Y)
#Splitting dataset into Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state =1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:]  = sc.fit_transform(X_train[:,3:])
X_test[:,3:]  = sc.fit_transform(X_test[:,3:])



print(X_train)
print("")
print(Y_train)
print(" ")
print(X_test)
print()
print(Y_test)













