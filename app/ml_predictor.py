"""
Flask Model Prediction Helper

This file loads the trained model and prepares user input from the Flask form
so the model can predict machine failure risk.
"""

from pathlib import Path

import joblib
import pandas as pd


# Set the project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Set the path to the trained model file
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# Define the same feature columns used during model training
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_model():
    """
    Load the trained machine learning model from the models folder.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run Step 4 model training first."
        )

    return joblib.load(MODEL_PATH)


def build_input_dataframe(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """
    Convert user input from the web form into a pandas DataFrame.
    """
    input_data = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": float(air_temperature),
                "Process temperature [K]": float(process_temperature),
                "Rotational speed [rpm]": float(rotational_speed),
                "Torque [Nm]": float(torque),
                "Tool wear [min]": float(tool_wear),
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    return input_data


def interpret_prediction(prediction):
    """
    Convert the model output into a readable message for the user.
    """
    if prediction == 1:
        return {
            "status": "Machine Failure Risk Detected",
            "recommendation": (
                "Maintenance inspection is recommended. "
                "The machine may be at risk of failure."
            ),
        }

    return {
        "status": "No Machine Failure Detected",
        "recommendation": (
            "Machine condition appears normal. "
            "Continue regular monitoring."
        ),
    }


def predict_machine_failure(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """
    Predict machine failure risk using the trained model.
    """
    # Load the saved machine learning pipeline
    model = load_model()

    # Convert form input into the correct model input format
    input_data = build_input_dataframe(
        machine_type=machine_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )

    # Predict class label: 0 means no failure, 1 means failure risk
    prediction = int(model.predict(input_data)[0])

    # Predict probability of machine failure
    failure_probability = float(model.predict_proba(input_data)[0][1]) * 100

    # Convert prediction into user-friendly result text
    interpretation = interpret_prediction(prediction)

    return {
        "prediction": prediction,
        "failure_probability": round(failure_probability, 2),
        "status": interpretation["status"],
        "recommendation": interpretation["recommendation"],
    }