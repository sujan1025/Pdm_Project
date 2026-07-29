from datetime import datetime
import bcrypt

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="engineer")

    predictions = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        password_bytes = password.encode("utf-8")
        hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
        self.password_hash = hashed_password.decode("utf-8")

    def check_password(self, password):
        password_bytes = password.encode("utf-8")
        hashed_bytes = self.password_hash.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    machine_type = db.Column(db.String(1), nullable=False)
    air_temperature = db.Column(db.Float, nullable=False)
    process_temperature = db.Column(db.Float, nullable=False)
    rotational_speed = db.Column(db.Float, nullable=False)
    torque = db.Column(db.Float, nullable=False)
    tool_wear = db.Column(db.Float, nullable=False)

    prediction = db.Column(db.Integer, nullable=False)
    failure_probability = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(120), nullable=False)
    recommendation = db.Column(db.String(255), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)