"""
Database Initialisation

This file creates the SQLite database tables and adds demo users
for testing the secure Flask application.
"""

from app import create_app
from app.models import User, db


# Create the Flask application
app = create_app()


def create_user(username, password, role):
    """
    Create a user account if the username does not already exist.
    """

    # Check whether the user already exists
    existing_user = User.query.filter_by(username=username).first()

    # Do not create duplicate users
    if existing_user:
        print(f"User already exists: {username}")
        return

    # Create a new user object
    user = User(username=username, role=role)

    # Hash and store the password
    user.set_password(password)

    # Add the user to the database
    db.session.add(user)
    db.session.commit()

    print(f"Created user: {username} | role: {role}")


# Run database setup inside Flask application context
with app.app_context():
    # Create database tables from the models
    db.create_all()

    # Create demo admin user
    create_user("admin", "admin123", "admin")

    # Create demo engineer user
    create_user("engineer", "engineer123", "engineer")

    print("\nDatabase initialised successfully.")
    print("Login accounts:")
    print("Admin    -> username: admin    password: admin123")
    print("Engineer -> username: engineer password: engineer123")