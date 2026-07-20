from flask import Blueprint, render_template, request
from app.ml_predictor import predict_machine_failure


main = Blueprint("main", __name__)


@main.route("/")
def dashboard():
    return render_template("dashboard.html")


@main.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            machine_type = request.form.get("machine_type")
            air_temperature = request.form.get("air_temperature")
            process_temperature = request.form.get("process_temperature")
            rotational_speed = request.form.get("rotational_speed")
            torque = request.form.get("torque")
            tool_wear = request.form.get("tool_wear")

            print("FORM DATA RECEIVED:")
            print(machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear)

            result = predict_machine_failure(
                machine_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            )

            input_values = {
                "Machine Type": machine_type,
                "Air Temperature [K]": air_temperature,
                "Process Temperature [K]": process_temperature,
                "Rotational Speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool Wear [min]": tool_wear,
            }

            return render_template(
                "result.html",
                result=result,
                input_values=input_values
            )

        except Exception as e:
            print("PREDICTION ERROR:", e)
            return render_template("predict.html", error=str(e))

    return render_template("predict.html")