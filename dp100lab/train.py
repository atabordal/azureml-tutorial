
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from azureml.core import Run
print('Libraries Imported')

# ***  Azure Machine Learning service specfic code starts... ***

# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--maxdepth', type=float, dest='max_depth', default=14, help='max_depth')
args = parser.parse_args()


data_folder = args.data_folder
max_depth = args.max_depth

print('Data folder:', data_folder)

# get hold of the current run
run = Run.get_context()

# ***  Azure Machine Learning service specfic code ends. ***

# filepath = data_folder + '/BikeModelFeatures.csv'
filepath = os.path.join(data_folder, 'BikeFeatures.csv')

df_features = pd.read_csv(filepath, sep=';')

# load train and test set into numpy arrays

X_train_all , X_test_all = train_test_split(df_features.values,test_size=0.2)      
#test_size=0.5(whole_data)

# Column 0 has the value we want to predict
X_train_all[:,0]

y_train = X_train_all[:,0]

y_test = X_test_all[:,0]

X_train_all.shape

X_train = X_train_all[:,1:11]
X_train.shape

X_test = X_test_all[:,1:11]
X_test

# Set the max_depth model hyperparameter to = max_depth which is the parameter value we created in the Azure ML service specific code above, i.e. , max_depth = max_depth 
# print('trainng RandomForestClassifier...')
#classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42, max_depth = max_depth)
classifier =  AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=10, max_depth = 20), n_estimators= 100, learning_rate=0.2)
classifier.fit(X_train, y_train)
print('Classifier Model Trained.')


# Predict using the test data...
print('Running the test dataset through...')
y_predtest = classifier.predict(X_test)
print('Test dataset scored.')

# calculate accuracy on the prediction
acc = np.average(y_predtest == y_test)
print('Accuracy is', acc)


# ***  Azure Machine Learning service specfic code starts... ***
run.log('data_dir', data_folder)
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)

# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=classifier, filename='outputs/biketypemodel.pkl')

# ***  Azure Machine Learning service specfic code ends. ***
