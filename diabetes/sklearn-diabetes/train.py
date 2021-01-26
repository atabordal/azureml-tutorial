
import os
import joblib
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from azureml.core import Run
print('Libraries Imported')

# ***  Azure Machine Learning service specfic code starts... ***

# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--penalty', type=str, dest='penalty', default='l2', help='penalty')
args = parser.parse_args()


data_folder = args.data_folder
penalty = args.penalty

print('Data folder:', data_folder)

# get hold of the current run
run = Run.get_context()

# ***  Azure Machine Learning service specfic code ends. ***

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

# ***  Azure Machine Learning service specfic code starts... ***
run.log('data_dir', data_folder)
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)

# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/diabetes_model.pkl')
X_validate.to_json('outputs/validation_data.json', orient="split")

# ***  Azure Machine Learning service specfic code ends. ***
