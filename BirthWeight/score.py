import os
import json
import joblib
import numpy as np
import xgboost as xgb


def init():
    #Load the model
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs/model.bst')
    model = xgb.Booster()
    model.load_model(model_path)

def run(raw_data):
    # Load the new Data
    data = np.array(json.loads(raw_data)['data'])
    data = xgb.DMatrix(data)
    
    # make prediction
    y_hat = model.predict(data)
    #Load Scaler
    sc_y = joblib.load(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs/std_scaler_y.bin'))
    y_hat = sc_y.inverse_transform(y_hat)
    # Pounds to kg
    y_hat = np.multiply(y_hat, 0.454)

    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
