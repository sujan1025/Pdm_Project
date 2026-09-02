# Predictive Maintenance Decision Support System

A secure Flask web application for early machine failure detection using machine learning.

## Overview

This project is a machine learning-based decision support system designed to support predictive maintenance. It uses the AI4I 2020 Predictive Maintenance Dataset to train and evaluate machine learning models for identifying machine failure risk from operating condition data.

The selected model is integrated into a Flask web application where authorised users can submit machine readings, receive prediction results, view failure probability, review maintenance recommendations, and monitor previous assessments through a reporting dashboard.

The system is developed as a local academic prototype. It does not replace engineering judgement or connect to live industrial sensors. Its purpose is to demonstrate how machine learning, secure web development, and dashboard reporting can support early machine failure detection.

## Features

## Machine Failure Prediction

- Predicts machine failure risk from operating condition data
- Uses temperature, rotational speed, torque, tool wear, and machine type
- Displays prediction status and failure probability
- Provides maintenance priority and decision-support indicators
- Stores prediction history for later review

## Decision Support Dashboard

- Dashboard overview for authenticated users
- Prediction result page with risk status and probability
- History page with previous assessments
- Circular graph for failure-risk and normal outcomes
- Probability trend graph for recent predictions
- Admin-only user management page

## User Authentication

- Login-based access control
- Role-based access for admin and engineer users
- Password hashing using bcrypt
- Session management using Flask-Login
- Admin-only protected routes

## Security Measures

- CSRF protection using Flask-WTF
- Server-side input validation
- Passwords stored as hashed values
- Protected Flask routes
- Role-based access enforced server-side
- OWASP ZAP passive scan evidence included

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

Input features used by the model:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Target variable:

- Machine failure

Excluded columns:

- UDI
- Product ID
- TWF
- HDF
- PWF
- OSF
- RNF

Identifier columns were removed because they do not represent machine operating behaviour. Failure mode columns were excluded to avoid data leakage.

## Machine Learning Workflow

1. Load the AI4I 2020 dataset
2. Perform exploratory data analysis and cleaning
3. Remove identifiers and leakage-related columns
4. Apply preprocessing and feature transformation
5. Split the dataset using stratified train-test splitting
6. Apply SMOTE to the training data only
7. Train Logistic Regression, Random Forest, and Gradient Boosting models
8. Tune models using GridSearchCV and stratified cross-validation
9. Evaluate models using classification metrics
10. Select and save the best model for Flask integration
11. Evaluate model performance across machine types

## Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting

Gradient Boosting was selected as the final model because it achieved the strongest balance between recall, precision, and F1-score.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8280 | 0.1443 | 0.8235 | 0.2456 | 0.9075 |
| Random Forest | 0.9655 | 0.4950 | 0.7353 | 0.5917 | 0.9726 |
| Gradient Boosting | 0.9740 | 0.5851 | 0.8088 | 0.6790 | 0.9712 |

## Fairness Evaluation

| Machine Type | Total Records | Actual Failures | Predicted Failures | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| H | 214 | 5 | 2 | 1.0000 | 0.4000 | 0.5714 | 0.9866 |
| L | 1170 | 38 | 62 | 0.5161 | 0.8421 | 0.6400 | 0.9745 |
| M | 616 | 25 | 30 | 0.7000 | 0.8400 | 0.7636 | 0.9648 |

## Technologies Used

## Backend and Machine Learning

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- joblib
- Flask-Login
- Flask-WTF
- Flask-SQLAlchemy
- bcrypt

## Database

- SQLite

SQLite is used as a lightweight local database for storing user accounts and prediction history through Flask-SQLAlchemy. No separate SQL server or database application is required for this prototype.

## Frontend

- HTML
- CSS
- Jinja2 templates

## Development and Security Tools

- VS Code
- Git and GitHub
- OWASP ZAP

## Project Structure

pdm_project/
├── app/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── predict.html
│   │   ├── result.html
│   │   ├── history.html
│   │   └── users.html
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   ├── forms.py
│   └── ml_predictor.py
├── data/
│   ├── ai4i2020.csv
│   └── ai4i2020_cleaned.csv
├── figures/
├── instance/
│   └── pdm_app.db
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── reports/
├── eda_cleaning.py
├── preprocessing.py
├── model_training.py
├── final_evaluation.py
├── init_db.py
├── run.py
├── requirements.txt
└── README.md

## Application Workflow

1. Dataset is prepared and cleaned
2. Preprocessing and feature transformation are applied
3. Machine learning models are trained and evaluated
4. The best model is saved for web application integration
5. SQLite database is initialised for users and prediction history
6. Users log in to the Flask application
7. Users submit machine operating readings
8. The system returns prediction results and stores the assessment
9. The dashboard displays historical prediction activity

## Application Pages

- `/login` - login page
- `/` - dashboard page
- `/predict` - machine failure risk analysis page
- `/history` - prediction history and reporting dashboard
- `/admin/users` - admin-only user management page
- `/logout` - logout route

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project workflow:

```bash
python eda_cleaning.py
python preprocessing.py
python model_training.py
python final_evaluation.py
python init_db.py
python run.py
```

Open the application:

```text
http://127.0.0.1:5000
```

## Security Testing

OWASP ZAP was used to perform passive security testing on the local Flask application.

Local application tested:

```text
http://127.0.0.1:5000
```

Scoped local Flask application results:

| Risk Level | Number of Alerts |
|---|---:|
| High | 0 |
| Medium | 2 |
| Low | 3 |
| Informational | 2 |
| Total | 7 |

The scan found no high-risk alerts for the local Flask application. The remaining findings relate mainly to security headers, cookie attributes, session observations, and deployment hardening.

## Limitations

- The dataset is synthetic and may not fully represent real industrial machinery
- The application runs locally and does not connect to live sensor streams
- SQLite is suitable for the prototype but not intended as the final production database
- Failure reason analysis provides likely indicators, not confirmed physical causes
- Further production hardening is required before deployment
- Model performance varies across machine types, especially H-type machines

## Future Work

- Integrate live sensor or IoT data
- Improve model explainability
- Add automated unit and integration tests
- Apply additional security hardening
- Use a production-ready database and deployment server
- Collect feedback from maintenance engineers or technical users
- Validate the system using real industrial maintenance data

## Use Of AI

- This assignment use the AI in the following ways for the purpose of completing the assignment more efficiently:
- Research : unserstanding best practice for secure web application developement with flask web development and machine learning workflow design.
- Feedback and suggestions : checked for implemented approach and suggestions for improvement.
- Editing : reviewing readme file for better readiability and structure.

## Tools Used

- ChatGPT - for code review, understanding concepts, and reviewing documentation.


## Author

Sujan Lamsal  
MSc Computer Science  
Leeds Trinity University