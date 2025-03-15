import joblib
import numpy as np
import pandas as pd  # Import pandas

# Base path for the model
MODEL_BASE_PATH = "core/utils/ml/model"

# Load the model once using joblib
model = joblib.load(f"{MODEL_BASE_PATH}/model_speech_defect_logistic_regression.pkl")
label_encoder = joblib.load(f"{MODEL_BASE_PATH}/label_speech_defect_logistic_regression.pkl")

def predict_speech(features):
    """
    Predicts speech characteristics based on features.
    :param features: List or numpy array of features [jitter, shimmer, speech_rate, pause, ...]
    :return: Prediction result
    """
    
    # Convert features list into a DataFrame
    feature_columns = ["Jitter (%)","Shimmer (%)","WPM (Words per Minute)","Pauses (Duration in sec)"]  # Make sure column names match training data
    features_df = pd.DataFrame([features], columns=feature_columns)

    encoded_prediction = model.predict(features_df)[0]  # Get encoded value
    decoded_label = label_encoder.inverse_transform([encoded_prediction])[0]  # Decode label
    
    return decoded_label  # Return actual label