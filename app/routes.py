from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.ml_predictor import predict_machine_failure
from app.models import PredictionHistory, User, db


main = Blueprint("main", __name__)


def admin_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("You do not have permission to access this page.", "error")
            return redirect(url_for("main.dashboard"))
        return route_function(*args, **kwargs)

    return wrapper


def parse_float(value, field_name, min_value=None, max_value=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid number.")

    if min_value is not None and number < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}.")

    if max_value is not None and number > max_value:
        raise ValueError(f"{field_name} must be no more than {max_value}.")

    return number


@main.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Login successful.", "success")
            return redirect(url_for("main.dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@main.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("main.login"))


@main.route("/")
@login_required
def dashboard():
    if current_user.role == "admin":
        total_predictions = PredictionHistory.query.count()
        recent_predictions = (
            PredictionHistory.query
            .order_by(PredictionHistory.created_at.desc())
            .limit(5)
            .all()
        )
    else:
        total_predictions = PredictionHistory.query.filter_by(
            user_id=current_user.id
        ).count()

        recent_predictions = (
            PredictionHistory.query
            .filter_by(user_id=current_user.id)
            .order_by(PredictionHistory.created_at.desc())
            .limit(5)
            .all()
        )

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        recent_predictions=recent_predictions,
    )


@main.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            machine_type = request.form.get("machine_type")

            if machine_type not in ["L", "M", "H"]:
                raise ValueError("Machine type must be L, M, or H.")

            air_temperature = parse_float(
                request.form.get("air_temperature"),
                "Air temperature",
                295.3,
                304.5,
            )

            process_temperature = parse_float(
                request.form.get("process_temperature"),
                "Process temperature",
                305.7,
                313.8,
            )

            rotational_speed = parse_float(
                request.form.get("rotational_speed"),
                "Rotational speed",
                1168,
                2886,
            )

            torque = parse_float(
                request.form.get("torque"),
                "Torque",
                3.8,
                76.6,
            )

            tool_wear = parse_float(
                request.form.get("tool_wear"),
                "Tool wear",
                0,
                253,
            )

            result = predict_machine_failure(
                machine_type=machine_type,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
            )

            input_values = {
                "Machine Type": machine_type,
                "Air Temperature [K]": air_temperature,
                "Process Temperature [K]": process_temperature,
                "Rotational Speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool Wear [min]": tool_wear,
            }

            history_record = PredictionHistory(
                user_id=current_user.id,
                machine_type=machine_type,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
                prediction=result["prediction"],
                failure_probability=result["failure_probability"],
                status=result["status"],
                recommendation=result["recommendation"],
            )

            db.session.add(history_record)
            db.session.commit()

            return render_template(
                "result.html",
                result=result,
                input_values=input_values,
            )

        except Exception as e:
            return render_template("predict.html", error=str(e))

    return render_template("predict.html")


@main.route("/history")
@login_required
def history():
    if current_user.role == "admin":
        records = (
            PredictionHistory.query
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )
    else:
        records = (
            PredictionHistory.query
            .filter_by(user_id=current_user.id)
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )

    return render_template("history.html", records=records)


@main.route("/admin/users")
@login_required
@admin_required
def users():
    all_users = User.query.order_by(User.username.asc()).all()
    return render_template("users.html", users=all_users)