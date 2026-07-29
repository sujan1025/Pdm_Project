from app import create_app
from app.models import User, db


app = create_app()


def create_user(username, password, role):
    existing_user = User.query.filter_by(username=username).first()

    if existing_user:
        print(f"User already exists: {username}")
        return

    user = User(username=username, role=role)
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    print(f"Created user: {username} | role: {role}")


with app.app_context():
    db.create_all()

    create_user("admin", "admin123", "admin")
    create_user("engineer", "engineer123", "engineer")

    print("\nDatabase initialised successfully.")
    print("Login accounts:")
    print("Admin    -> username: admin    password: admin123")
    print("Engineer -> username: engineer password: engineer123")