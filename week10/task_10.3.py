# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:12:14 2024

@author: ATIV
"""

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus
import os

# Load the data
tennis_data = pd.read_csv('play_tennis.csv')

# Convert categorical variables to numerical
for column in tennis_data.columns:
    if tennis_data[column].dtype == type(object):
        le = LabelEncoder()
        tennis_data[column] = le.fit_transform(tennis_data[column])

# Split the data into features and target variable
X = tennis_data.drop('PlayTennis', axis=1)
y = tennis_data['PlayTennis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Make predictions
dt_prediction = dt_clf.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, dt_prediction))

# Print the classification report
print(classification_report(y_test, dt_prediction))

# Print the accuracy
print("Accuracy: ", accuracy_score(y_test, dt_prediction))

# Visualize the decision tree
feature_names = tennis_data.columns[:-1]
target_names = ['No', 'Yes']
dot_data = tree.export_graphviz(dt_clf, out_file=None, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())

