"""
Model Training and Comparison

This file trains and compares Logistic Regression, Random Forest,
and Gradient Boosting models. The best model is selected using F1-score
and saved for use in the Flask web application.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Set project folders
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("figures")

# Create folders if they do not already exist
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv(DATA_DIR / "ai4i2020_cleaned.csv")

print("STEP 4: MODEL TRAINING STARTED")
print("-" * 50)

# Select input features for prediction
feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Select target column
target_column = "Machine failure"

X = df[feature_columns]
y = df[target_column]

# Identify categorical and numerical columns
categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Split the data into training and testing sets
# Stratify keeps the rare failure class proportion stable
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Create preprocessing step for all models
preprocessor = ColumnTransformer(
    transformers=[
        ("type_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num_scaler", StandardScaler(), numerical_features),
    ]
)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Define hyperparameter settings for each model
parameter_grids = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5],
    },
}

# Use stratified cross-validation because the dataset is imbalanced
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []
trained_models = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")

    # Create a full machine learning pipeline
    # The pipeline applies preprocessing, SMOTE, and model training together
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ]
    )

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grids[model_name],
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
    )

    # Train the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best fitted model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Make predictions on the unchanged test set
    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    # Store results for comparison
    results.append(
        {
            "Model": model_name,
            "Best CV F1": grid_search.best_score_,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "Best Parameters": grid_search.best_params_,
        }
    )

    trained_models[model_name] = best_model

    # Save classification report for this model
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    safe_model_name = model_name.lower().replace(" ", "_")
    report_df.to_csv(REPORTS_DIR / f"classification_report_{safe_model_name}.csv")

    # Create and save confusion matrix chart
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_model_name}.png")
    plt.close()

    print(f"{model_name} completed.")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Convert model results into a DataFrame
results_df = pd.DataFrame(results)

# Save model comparison results
results_df.to_csv(REPORTS_DIR / "model_comparison_results.csv", index=False)

# Select the best model using test F1-score
best_model_row = results_df.sort_values(by="F1-score", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
final_model = trained_models[best_model_name]

# Save the best model for the Flask application
joblib.dump(final_model, MODELS_DIR / "best_model.pkl")

# Save a text summary of the best model
with open(REPORTS_DIR / "best_model_summary.txt", "w", encoding="utf-8") as file:
    file.write("BEST MODEL SUMMARY\n")
    file.write("-" * 50 + "\n\n")

    file.write("Best model selected by test F1-score:\n")
    file.write(f"{best_model_name}\n\n")

    file.write("Model comparison results:\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\n")

# Save ROC curve comparison for all models
plt.figure(figsize=(8, 6))

for model_name, model in trained_models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=model_name,
        ax=plt.gca(),
    )

plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curve_comparison.png")
plt.close()

print("\nModel comparison results saved at:")
print(REPORTS_DIR / "model_comparison_results.csv")

print("\nBest model saved at:")
print(MODELS_DIR / "best_model.pkl")

print("\nBest model selected:")
print(best_model_name)
