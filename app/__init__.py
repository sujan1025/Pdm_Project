"""
Flask Application Setup

This file creates the Flask app, connects the database,
sets up login management, enables CSRF protection, and registers routes.
"""

import os

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from app.models import User, db


# Create Flask-Login manager for user sessions
login_manager = LoginManager()

# Create CSRF protection object for form security
csrf = CSRFProtect()


def create_app():
    """
    Create and configure the Flask application.
    """
    # Create the Flask app
    app = Flask(__name__)

    # Set secret key for sessions and CSRF protection
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY",
        "temporary-secret-key-change-later",
    )

    # Set SQLite database location
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pdm_app.db"

    # Disable modification tracking to reduce overhead
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Connect the database to the Flask app
    db.init_app(app)

    # Connect login manager to the Flask app
    login_manager.init_app(app)

    # Enable CSRF protection for forms
    csrf.init_app(app)

    # Send users to login page if they try to access protected pages
    login_manager.login_view = "main.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    @login_manager.user_loader
    def load_user(user_id):
        """
        Load the logged-in user from the database using their user ID.
        """
        return User.query.get(int(user_id))

    # Import and register application routes
    from app.routes import main

    app.register_blueprint(main)

    return app