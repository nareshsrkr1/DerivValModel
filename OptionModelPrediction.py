import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import numpy as np

# Load the model
loaded_model = tf.keras.models.load_model("option_pricing_model_savedmodel")
print('model loaded')

# Load the scaler
scaler = StandardScaler()
scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)

def model_custom_predict(Spot, Strike, Maturity, Volatility):
    inputs_to_model = pd.DataFrame({'Spot Price': [Spot / Strike],
                                    'Strike Price': [Strike],
                                    'Maturity': [Maturity / 365],
                                    'Volatility': [Volatility / 100]})

    new_inputs_scaled = scaler.transform(inputs_to_model)
    value = loaded_model.predict(new_inputs_scaled)

    return value * Strike

print(model_custom_predict(50, 45, 90, 40))
