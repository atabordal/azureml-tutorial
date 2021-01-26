
import os
import joblib
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print('Libraries Imported')

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--penalty', type=str, dest='penalty', default='l2', help='penalty')
args = parser.parse_args()

data_folder = args.data_folder
penalty = args.penalty

print('Data folder:', data_folder)

filepath = os.path.join(data_folder, 'diabetes.csv')

df_diabetes = pd.read_csv(filepath)
#Features data
X0= df_diabetes.loc[:,  df_diabetes.columns != 'Outcome']
#label data
y= df_diabetes[['Outcome']]

# Scaler the data 
names = X0.columns
scaler = StandardScaler()
X = scaler.fit_transform(X0)
X = pd.DataFrame(X, columns=names)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.3)

# Adjuting model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty=penalty,random_state=0)
clf.fit(X_train, np.ravel(y_train))
print('Regressionn Model Trained.')

# Predict using the test data...
print('Running the test dataset through...')
y_predtest = clf.predict(X_test)
print('Test dataset scored.')

# calculate accuracy on the prediction
acc= clf.score(X_test, y_test)
print("accuracy = ",acc * 100,"%")

# note file saved in the outputs folder is automatically uploaded into experiment record
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=clf, filename='outputs/diabetesmodel.pkl')
X_validate.to_json('outputs/validation_data.json', orient="split")
