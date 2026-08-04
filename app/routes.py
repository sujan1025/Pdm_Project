"""
Flask Routes

This file controls the web pages of the application, including login,
dashboard, prediction, history, and admin user management.
"""

from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.ml_predictor import predict_machine_failure
from app.models import PredictionHistory, User, db


# Create a Flask blueprint to organise routes
main = Blueprint("main", __name__)


def admin_required(route_function):
    """
    Allow only admin users to access selected routes.
    """

    @wraps(route_function)
    def wrapper(*args, **kwargs):
        # Check whether the logged-in user is an admin
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("You do not have permission to access this page.", "error")
            return redirect(url_for("main.dashboard"))

        # Continue to the requested route if the user is admin
        return route_function(*args, **kwargs)

    return wrapper


def parse_float(value, field_name, min_value=None, max_value=None):
    """
    Convert form input into a float and check if it is within a valid range.
    """

    # Try to convert the form value into a number
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid number.")

    # Check minimum allowed value
    if min_value is not None and number < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}.")

    # Check maximum allowed value
    if max_value is not None and number > max_value:
        raise ValueError(f"{field_name} must be no more than {max_value}.")

    return number


@main.route("/login", methods=["GET", "POST"])
def login():
    """
    Show the login page and process login form submission.
    """

    # If user is already logged in, send them to the dashboard
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    # Process the login form after the user submits it
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Search for the username in the database
        user = User.query.filter_by(username=username).first()

        # Check whether the user exists and the password is correct
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful.", "success")
            return redirect(url_for("main.dashboard"))

        # Show error message if login details are wrong
        flash("Invalid username or password.", "error")

    # Show login page for GET request
    return render_template("login.html")


@main.route("/logout")
@login_required
def logout():
    """
    Log out the current user.
    """

    # End the current user session
    logout_user()

    # Show logout message and redirect to login page
    flash("You have been logged out.", "success")
    return redirect(url_for("main.login"))


@main.route("/")
@login_required
def dashboard():
    """
    Show the dashboard after login.
    """

    # Admin users can see prediction records from all users
    if current_user.role == "admin":
        total_predictions = PredictionHistory.query.count()

        recent_predictions = (
            PredictionHistory.query
            .order_by(PredictionHistory.created_at.desc())
            .limit(5)
            .all()
        )

    # Engineer users can see only their own prediction records
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

    # Send dashboard data to the dashboard page
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

    # Process prediction form after user submits it
    if request.method == "POST":
        try:
            # Get machine type from the form
            machine_type = request.form.get("machine_type")

            # Check that machine type is one of the expected values
            if machine_type not in ["L", "M", "H"]:
                raise ValueError("Machine type must be L, M, or H.")

            # Get and validate air temperature
            air_temperature = parse_float(
                request.form.get("air_temperature"),
                "Air temperature",
                295.3,
                304.5,
            )

            # Get and validate process temperature
            process_temperature = parse_float(
                request.form.get("process_temperature"),
                "Process temperature",
                305.7,
                313.8,
            )

            # Get and validate rotational speed
            rotational_speed = parse_float(
                request.form.get("rotational_speed"),
                "Rotational speed",
                1168,
                2886,
            )

            # Get and validate torque
            torque = parse_float(
                request.form.get("torque"),
                "Torque",
                3.8,
                76.6,
            )

            # Get and validate tool wear
            tool_wear = parse_float(
                request.form.get("tool_wear"),
                "Tool wear",
                0,
                253,
            )

            # Send the validated input values to the trained model
            result = predict_machine_failure(
                machine_type=machine_type,
                air_temperature=air_temperature,
                process_temperature=process_temperature,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear,
            )

            # Store input values so they can be displayed on the result page
            input_values = {
                "Machine Type": machine_type,
                "Air Temperature [K]": air_temperature,
                "Process Temperature [K]": process_temperature,
                "Rotational Speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool Wear [min]": tool_wear,
            }

            # Create a prediction history record for the database
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

            # Save the prediction result in the database
            db.session.add(history_record)
            db.session.commit()

            # Show the prediction result page
            return render_template(
                "result.html",
                result=result,
                input_values=input_values,
            )

        # Show validation or prediction errors on the prediction page
        except Exception as error:
            return render_template("predict.html", error=str(error))

    # Show prediction form for GET request
    return render_template("predict.html")


@main.route("/history")
@login_required
def history():
    """
    Show prediction history records.
    """

    # Admin users can see all prediction records
    if current_user.role == "admin":
        records = (
            PredictionHistory.query
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )

    # Engineer users can see only their own prediction records
    else:
        records = (
            PredictionHistory.query
            .filter_by(user_id=current_user.id)
            .order_by(PredictionHistory.created_at.desc())
            .all()
        )

    # Send records to the history page
    return render_template("history.html", records=records)


@main.route("/admin/users")
@login_required
@admin_required
def users():
    """
    Show admin-only user management page.
    """

    # Get all users from the database
    all_users = User.query.order_by(User.username.asc()).all()

    # Send user list to the users page
    return render_template("users.html", users=all_users)