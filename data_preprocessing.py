# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')  # getting data file, which contains 4 column
x = dataset.iloc[:, :-1].values  # assigning first 0, 1, 2 column to the x variable
y = dataset.iloc[:, 3].values  # putting last column to the y variable

# taking care of missing data in the data. in our case missing data is in 2nd and third row
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')  # missing value will be replaced by mean
impute = impute.fit(x[:, 1:3])  # it calculate the mean of the 2nd and 3rd column 
x[:, 1:3] = impute.transform(x[:, 1:3])  # it puts the means at the missing value position

# Encoding categorical data
# in our task we will be changing the first column(text) with the encoding data(number)
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])  # this command gives number alphabetical wise
# to the different number of catergory in the data
# france  --> 0
# spain   --> 2
# germany --> 1
# our text column lookes currently like this [0 2 1 2 1 0 2 0 1 0]
# this gives arise to another error, which will be solved be using onehotencoder

# to use onehotencoder to convert one column into number of column depending on the data, we need to use
# ColumnTransformer
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'  # Leave the rest of the columns untouched
)
x = onehotencoder.fit_transform(x)  # this command apply the above mentioned algorithm to the 'x'
# now our data looks like
#   fra spn gmny
# [[1.0 0.0 0.0]
# [0.0 0.0 1.0]
# [0.0 1.0 0.0]
# [0.0 0.0 1.0]
# [0.0 1.0 0.0]
# [1.0 0.0 0.0]
# [0.0 0.0 1.0]
# [1.0 0.0 0.0]
# [0.0 1.0 0.0]
# [1.0 0.0 0.0]]
# print(x[:, 0: 3])


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
