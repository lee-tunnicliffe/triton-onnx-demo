# Import required libraries
import numpy
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import signal
import subprocess

# Generate dummy data to perform binary classification
seed = 7
features = 9 # number of sample features
samples = 10000 # number of samples
X = numpy.random.rand(samples, features).astype('float32')
Y = numpy.random.randint(2, size=samples)

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.2f}".format(accuracy * 100.0))

model.save_model('models/triton_xgboost_model/1/xgboost.json')
