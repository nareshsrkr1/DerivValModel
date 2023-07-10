import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load the model
# loaded_model = tf.keras.models.load_model("option_pricing_model_saved_updated")
loaded_model = tf.keras.models.load_model("option_pricing_model_saved_model")

print('Model loaded')

# Load the scaler
# scaler_filename = "scaler.save"
scaler_filename = "scaler_model.save"

scaler = joblib.load(scaler_filename)



# def model_custom_predict(Spot_Price, Strike_Price, Maturity, risk_free_interest,Volatility, ):
#     inputs_to_model = pd.DataFrame({'Spot_Price': [Spot_Price / Strike_Price],
#                                     'Strike_Price': [Strike_Price],
#                                     'Maturity': [Maturity],
#                                     'risk_free_interest': [risk_free_interest],
#                                     'Volatility': [Volatility]
#                                    })
def model_custom_predict(Spot, Strike, Maturity, Volatility):
    inputs_to_model = pd.DataFrame({'Spot Price': [Spot / Strike],
                                    'Strike Price': [Strike],
                                    'Maturity': [Maturity / 365],
                                    'Volatility': [Volatility / 100]})

    new_inputs_scaled = scaler.transform(inputs_to_model)
    value = loaded_model.predict(new_inputs_scaled)

    return value * Strike

# print(model_custom_predict(2500, 2450.5, 2, 0.01,0.6))
print(model_custom_predict(50, 45, 90, 40))

# 2023-07-10 02:00:46 INFO: Monte Carlo Call Option Value: 853.5145263671875
# print(model_custom_predict(50, 45, 90, 10))
# 1900,2031.1,2,0.01,0.6,594.1515
# 50,35.0,1,0.01,0.1,15.3446

