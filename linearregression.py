# -*- coding: utf-8 -*-
"""LinearRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YTqxSxPrVK790Cep7V2LzYL-nXten6OA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('new.csv',encoding='ISO-8859-1')

# Drop irrelevant columns
df = df.drop(['url','id','tradeTime', 'Cid', 'DOM','floor'], axis=1)

# Check for missing values
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()
df = df[df['constructionTime'] != 'Î´Öª']

# Set the target variable
target = 'totalPrice'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values for Housing Prices")
plt.show()

# Print the MSE and R-squared score
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")