import joblib
import numpy as np
from catboost import CatBoostRegressor




def get_prediction(data,model):
    res = model.predict(data)
    return res[0]
