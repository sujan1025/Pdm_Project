"""
Flask Routes

This file controls the web pages of the application, including login,
dashboard, prediction, history reporting, and admin user management.
"""

from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.ml_predictor import predict_machine_failure
from app.models import PredictionHistory, User, db


main = Blueprint("main", __name__)


def admin_required(route_function):
    """
    Allow only admin users to access selected routes.
    """

    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("You do not have permission to access this page.", "error")
            return redirect(url_for("main.dashboard"))

        return route_function(*args, **kwargs)

    return wrapper


def parse_float(value, field_name, min_value=None, max_value=None):
    """
    Convert form input into a float and check whether it is within a valid range.
    """

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
    """
    Show the login page and process login form submission.
    """

    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)

            # No login-success flash message is used here.
            # This prevents old login messages from appearing after logout.
            return redirect(url_for("main.dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@main.route("/logout")
@login_required
def logout():
    """
    Log out the current user and show only the logout message.
    """

    # Remove any old flash messages before showing the logout message
    session.pop("_flashes", None)

    logout_user()

    flash("You have been logged out.", "success")

    return redirect(url_for("main.login"))


@main.route("/")
@login_required
def dashboard():
    """
    Show the dashboard after login.
    """

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
    """
    Show the prediction form and process machine failure predictions.
    """

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

        except Exception as error:
            return render_template("predict.html", error=str(error))

    return render_template("predict.html")


@main.route("/history")
@login_required
def history():
    """
    Show prediction history with graph-based reporting.
    """

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

    total_records = len(records)

    failure_records = sum(1 for record in records if record.prediction == 1)
    normal_records = total_records - failure_records

    if total_records > 0:
        normal_percentage = round((normal_records / total_records) * 100, 2)
        failure_percentage = round((failure_records / total_records) * 100, 2)

        average_probability = round(
            sum(record.failure_probability for record in records) / total_records,
            2,
        )

        highest_probability = round(
            max(record.failure_probability for record in records),
            2,
        )

    else:
        normal_percentage = 0
        failure_percentage = 0
        average_probability = 0
        highest_probability = 0

    recent_for_chart = list(reversed(records[:10]))

    trend_points = []
    point_count = len(recent_for_chart)

    for index, record in enumerate(recent_for_chart):
        if point_count == 1:
            x_position = 50
        else:
            x_position = (index / (point_count - 1)) * 100

        y_position = 100 - max(0, min(record.failure_probability, 100))

        trend_points.append(
            {
                "x": round(x_position, 2),
                "y": round(y_position, 2),
                "probability": round(record.failure_probability, 2),
                "label": record.created_at.strftime("%d %b %H:%M"),
            }
        )

    svg_points = " ".join(
        f"{point['x']},{point['y']}" for point in trend_points
    )

    return render_template(
        "history.html",
        records=records,
        total_records=total_records,
        failure_records=failure_records,
        normal_records=normal_records,
        normal_percentage=normal_percentage,
        failure_percentage=failure_percentage,
        average_probability=average_probability,
        highest_probability=highest_probability,
        trend_points=trend_points,
        svg_points=svg_points,
    )


@main.route("/admin/users")
@login_required
@admin_required
def users():
    """
    Show admin-only user management page.
    """

    all_users = User.query.order_by(User.username.asc()).all()

    return render_template("users.html", users=all_users)