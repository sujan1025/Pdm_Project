"""
Machine Learning Prediction Module

This file loads the trained machine learning model and uses it to predict
machine failure risk from user input values. It also provides a simple
rule-based explanation to support maintenance decision-making.
"""

from pathlib import Path
import math
import joblib
import pandas as pd


# Get the main project folder path
BASE_DIR = Path(__file__).resolve().parents[1]

# Location of the saved trained model
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def load_model():
    """
    Load the trained machine learning model from the models folder.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

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
    Convert user input values into a dataframe format expected by the model.
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
        ]
    )

    return input_data


def analyse_failure_reason(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    failure_probability,
):
    """
    Provide a likely explanation for the machine failure risk.

    This is not a direct causal diagnosis. It is a decision-support explanation
    based on input conditions and common predictive maintenance indicators.
    """

    reasons = []

    # Calculate temperature difference
    temperature_difference = process_temperature - air_temperature

    # Calculate approximate mechanical power in kilowatts
    mechanical_power_kw = (torque * rotational_speed * 2 * math.pi) / 60000

    # Estimate overstrain using tool wear and torque
    overstrain_index = tool_wear * torque

    # Machine-type-based overstrain threshold
    if machine_type == "L":
        overstrain_threshold = 11000
    elif machine_type == "M":
        overstrain_threshold = 12000
    else:
        overstrain_threshold = 13000

    # Tool wear condition
    if tool_wear >= 200:
        reasons.append(
            "High tool wear detected. The tool has been used for a long period, "
            "which may increase friction, reduce cutting efficiency and raise failure risk."
        )

    # Torque condition
    if torque >= 60:
        reasons.append(
            "High torque detected. The machine may be operating under heavy mechanical load, "
            "which can increase stress on components."
        )

    # Rotational speed condition
    if rotational_speed >= 1800:
        reasons.append(
            "High rotational speed detected. Running at high speed may increase vibration, "
            "heat generation and mechanical stress."
        )

    # Heat dissipation style condition
    if temperature_difference < 8.6 and rotational_speed < 1380:
        reasons.append(
            "Possible heat dissipation issue detected. The process and air temperature gap is low "
            "while rotational speed is also low, which may indicate poor cooling behaviour."
        )

    # Power condition
    if mechanical_power_kw < 3.5:
        reasons.append(
            "Low mechanical power pattern detected. The combination of torque and rotational speed "
            "may indicate abnormal operating behaviour."
        )

    elif mechanical_power_kw > 9.0:
        reasons.append(
            "High mechanical power pattern detected. The combination of torque and rotational speed "
            "suggests excessive load on the machine."
        )

    # Overstrain condition
    if overstrain_index > overstrain_threshold:
        reasons.append(
            "Possible overstrain condition detected. The combination of tool wear and torque is high "
            "for this machine type, which may increase failure risk."
        )

    # Combined risk pattern
    if failure_probability >= 70 and not reasons:
        reasons.append(
            "The model detected a high-risk combination of operating conditions. Although one single "
            "input is not extreme, the combined pattern of temperature, speed, torque and tool wear "
            "suggests possible machine failure risk."
        )

    elif failure_probability >= 40 and not reasons:
        reasons.append(
            "The model detected a moderate risk pattern. The current operating conditions should be "
            "monitored because the combined input values may indicate early warning behaviour."
        )

    elif not reasons:
        reasons.append(
            "No major abnormal operating condition was detected from the input values. The machine "
            "appears to be operating within a normal range."
        )

    # Select maintenance priority
    if failure_probability >= 70:
        priority = "High Priority"
        action = (
            "Immediate maintenance inspection is recommended. Check tool condition, torque load, "
            "cooling performance and machine vibration before continuing long operation."
        )

    elif failure_probability >= 40:
        priority = "Medium Priority"
        action = (
            "Schedule a preventive maintenance check. Continue monitoring the machine and review "
            "tool wear, temperature behaviour and load conditions."
        )

    else:
        priority = "Low Priority"
        action = (
            "No immediate maintenance action is required. Continue normal monitoring and routine "
            "maintenance practice."
        )

    return {
        "main_reason": reasons[0],
        "all_reasons": reasons,
        "maintenance_priority": priority,
        "maintenance_action": action,
        "mechanical_power_kw": round(mechanical_power_kw, 2),
        "temperature_difference": round(temperature_difference, 2),
        "overstrain_index": round(overstrain_index, 2),
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
    Predict machine failure risk and return prediction, probability,
    explanation and maintenance recommendation.
    """

    # Load the trained model
    model = load_model()

    # Build dataframe for prediction
    input_data = build_input_dataframe(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability of failure class
    failure_probability = model.predict_proba(input_data)[0][1]
    failure_probability_percent = round(float(failure_probability) * 100, 2)

    # Create explanation for the result
    explanation = analyse_failure_reason(
        machine_type=machine_type,
        air_temperature=float(air_temperature),
        process_temperature=float(process_temperature),
        rotational_speed=float(rotational_speed),
        torque=float(torque),
        tool_wear=float(tool_wear),
        failure_probability=failure_probability_percent,
    )

    # Prepare prediction status
    if prediction == 1:
        status = "Machine Failure Risk Detected"
    else:
        status = "No Machine Failure Detected"

    # Combine recommendation with reason
    recommendation = (
        f"{explanation['maintenance_action']} "
        f"Likely reason: {explanation['main_reason']}"
    )

    return {
        "prediction": int(prediction),
        "failure_probability": failure_probability_percent,
        "status": status,
        "recommendation": recommendation,
        "failure_reason": explanation["main_reason"],
        "all_reasons": explanation["all_reasons"],
        "maintenance_priority": explanation["maintenance_priority"],
        "maintenance_action": explanation["maintenance_action"],
        "mechanical_power_kw": explanation["mechanical_power_kw"],
        "temperature_difference": explanation["temperature_difference"],
        "overstrain_index": explanation["overstrain_index"],
    }