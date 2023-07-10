import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, filename='model_training.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Spot_Price,Strike_Price,Maturity,risk_free_interest,Volatility,Call_Premium
def load_dataset(filename):
    df = pd.read_csv(filename)
    # df['risk_free_interest']=df['risk_free_interest']
    # df['Volatility'] = df['Volatility']
    # df['Maturity'] = df['Maturity']
    # df['Spot_Price'] = df['Spot_Price'] / df['Strike_Price']
    # df['Call_Premium'] = df['Call_Premium'] / df['Strike_Price']

    df['Volatility'] = df['Volatility'] / 100
    df['Maturity'] = df['Maturity'] / 365
    df['Spot Price'] = df['Spot Price'] / df['Strike Price']
    df['Call_Premium'] = df['Call_Premium'] / df['Strike Price']

    X = df.drop('Call_Premium', axis=1)
    Y = df['Call_Premium']
    return X, Y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_model(input_dim):
    nodes = 120
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation(custom_activation))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def custom_activation(x):
    return K.exp(x)

def train_model(model, X_train_scaled, Y_train, validation_split=0.1, batch_size=64, epochs=100):
    try:
        model_history = model.fit(X_train_scaled, Y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=2)
        return model_history
    except Exception as e:
        logging.error("An error occurred during model training: %s", str(e))

def save_model(model, scaler, model_filename, scaler_filename):
    try:
        model.save(model_filename, save_format='tf')
        logging.info("Model saved successfully.")
        joblib.dump(scaler, scaler_filename)
        logging.info("Scaler saved successfully.")
    except Exception as e:
        logging.error("An error occurred during model and scaler saving: %s", str(e))

def evaluate_model(model, X_test_scaled, Y_test):
    try:
        loss = model.evaluate(X_test_scaled, Y_test)
        logging.info("Test loss: %s", loss)
    except Exception as e:
        logging.error("An error occurred during model evaluation: %s", str(e))
if __name__ == '__main__':
    logging.info("Loading dataset...")
    # X, Y = load_dataset('InputDataSet_updated.csv')
    X, Y = load_dataset('InputDataSet_before.csv')

    logging.info("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    logging.info("Scaling data...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    logging.info("Building model...")
    model = build_model(X_train_scaled.shape[1])

    logging.info("Training model...")
    num_epochs = 100
    model_history = train_model(model, X_train_scaled, Y_train, epochs=num_epochs)

    logging.info("Saving model and scaler...")
    save_model(model, scaler, "option_pricing_model_saved_model", "scaler_model.save")

    logging.info("Evaluating model...")
    evaluate_model(model, X_test_scaled, Y_test)

