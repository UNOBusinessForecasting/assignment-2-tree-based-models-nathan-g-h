# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
data = pd.read_csv(url)

data = data.drop(columns=['id', 'DateTime'])

# Separate the dependent and independent variables
X = data.drop(columns=['meal'])
Y = data['meal']

# Split inyo training and testing data
x, xt, y, yt = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=500, 
                               max_depth=100, 
                               max_features=40, 
                               min_samples_leaf=5, 
                               n_jobs=-1, 
                               random_state=42)
modelFit = model.fit(x, y)

# determine accuracy
predictions = model.predict(xt)
accuracy = accuracy_score(yt, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

test_data = test_data.drop(columns=['id', 'DateTime', 'meal'])

pred = model.predict(test_data)

# print(f"Predictions: {pred}")
