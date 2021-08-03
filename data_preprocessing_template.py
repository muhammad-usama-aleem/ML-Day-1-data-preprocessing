# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')  # getting data file, which contains 4 column
x = dataset.iloc[:, :-1].values  # assigning first 0, 1, 2 column to the x variable
y = dataset.iloc[:, 3].values  # putting last column to the y variable


# Splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# feature scaling
from sklearn.preprocessing import StandardScaler

# we scale our data so higher square root difference does not dominate the lower one.
# here are are only scaling x data, not y data because in current case y data is already scaled (0-1)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
