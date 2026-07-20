from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def predict_machine_failure(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([
        {
            "Type": machine_type,
            "Air temperature [K]": float(air_temperature),
            "Process temperature [K]": float(process_temperature),
            "Rotational speed [rpm]": float(rotational_speed),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear),
        }
    ])

    prediction = model.predict(input_data)[0]
    failure_probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        status = "Machine Failure Risk Detected"
        recommendation = "Maintenance inspection is recommended."
    else:
        status = "No Machine Failure Detected"
        recommendation = "Machine condition appears normal."

    return {
        "prediction": int(prediction),
        "failure_probability": round(float(failure_probability) * 100, 2),
        "status": status,
        "recommendation": recommendation,
    }