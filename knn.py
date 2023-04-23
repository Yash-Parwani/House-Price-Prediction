# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YTqxSxPrVK790Cep7V2LzYL-nXten6OA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('new.csv',encoding='ISO-8859-1')

# Drop irrelevant columns
df = df.drop(['url','id','tradeTime', 'DOM','floor'], axis=1)

# Check for missing values
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()
# remove the rows with 'Î´Öª' in the constructionTime column
df = df[df['constructionTime'].apply(lambda x: str(x).isdigit())]

# select the features and target variable
X = df[['Lng', 'Lat', 'Cid', 'followers', 'square']]
y = df['totalPrice']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a KNN model and fit the training data
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# make predictions on the testing data
y_pred = knn.predict(X_test)

# calculate mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Calculate accuracy score
accuracy = knn.score(X_test, y_test)
print("Accuracy Score: {:.2f}".format(accuracy))
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices (KNN)")
plt.show()