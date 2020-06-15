# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('data.csv')


X = dataset.iloc[:, :3]

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
alg = LinearRegression()

#Fitting model with trainig data
alg.fit(X, y)

# Saving model to disk
pickle.dump(alg, open('model.pkl','wb'))
