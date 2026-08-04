"""
Final Model Evaluation and Fairness Analysis

This file loads the saved best model, evaluates it on the unchanged test set,
performs fairness analysis by machine type, and saves final evaluation outputs.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# Set project folders
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("figures")

# Create folders if they do not already exist
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv(DATA_DIR / "ai4i2020_cleaned.csv")


# Select input features used during training
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

# Recreate the same train-test split used in Step 4
# This ensures the final evaluation uses the same unchanged test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Load the best trained model saved from Step 4
best_model = joblib.load(MODELS_DIR / "best_model.pkl")

# Make predictions on the unchanged test set
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

# Calculate final evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\nFINAL TEST SET PERFORMANCE")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save final metrics
metrics_df = pd.DataFrame(
    [
        {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
        }
    ]
)

metrics_df.to_csv(REPORTS_DIR / "final_model_metrics.csv", index=False)

# Save final classification report
classification_report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
).transpose()

classification_report_df.to_csv(REPORTS_DIR / "final_classification_report.csv")

# Create and save final confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)

cm_df.to_csv(REPORTS_DIR / "final_confusion_matrix.csv", index=False)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Final Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "final_confusion_matrix_gradient_boosting.png")
plt.close()

# Create and save ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(6, 5))
plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final ROC Curve - Gradient Boosting")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "final_roc_curve_gradient_boosting.png")
plt.close()

# Perform fairness analysis across machine types
# This checks whether the model behaves differently for L, M and H machines
fairness_results = []

for machine_type in ["L", "M", "H"]:
    # Select test records that belong to the current machine type
    type_mask = X_test["Type"] == machine_type

    X_type = X_test[type_mask]
    y_type = y_test[type_mask]

    # Make predictions for this machine type only
    y_type_pred = best_model.predict(X_type)
    y_type_probability = best_model.predict_proba(X_type)[:, 1]

    # Calculate performance metrics for this machine type
    type_precision = precision_score(y_type, y_type_pred, zero_division=0)
    type_recall = recall_score(y_type, y_type_pred, zero_division=0)
    type_f1 = f1_score(y_type, y_type_pred, zero_division=0)

    # ROC-AUC can only be calculated if both classes exist in the subset
    if len(y_type.unique()) > 1:
        type_roc_auc = roc_auc_score(y_type, y_type_probability)
    else:
        type_roc_auc = None

    fairness_results.append(
        {
            "Machine Type": machine_type,
            "Total Records": len(y_type),
            "Actual Failures": int(y_type.sum()),
            "Predicted Failures": int(y_type_pred.sum()),
            "Precision": type_precision,
            "Recall": type_recall,
            "F1-score": type_f1,
            "ROC-AUC": type_roc_auc,
        }
    )

# Save fairness analysis
fairness_df = pd.DataFrame(fairness_results)
fairness_df.to_csv(REPORTS_DIR / "fairness_analysis_by_machine_type.csv", index=False)

print("\nFAIRNESS ANALYSIS BY MACHINE TYPE")
print(fairness_df)

# Create recall comparison chart by machine type
plt.figure(figsize=(6, 4))
sns.barplot(data=fairness_df, x="Machine Type", y="Recall")
plt.title("Recall by Machine Type")
plt.xlabel("Machine Type")
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fairness_recall_by_machine_type.png")
plt.close()

# Create F1-score comparison chart by machine type
plt.figure(figsize=(6, 4))
sns.barplot(data=fairness_df, x="Machine Type", y="F1-score")
plt.title("F1-score by Machine Type")
plt.xlabel("Machine Type")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fairness_f1_by_machine_type.png")
plt.close()

# Extract feature importance from the trained classifier if available
classifier = best_model.named_steps["classifier"]
preprocessor = best_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

if hasattr(classifier, "feature_importances_"):
    feature_importance = classifier.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_importance,
        }
    ).sort_values(by="Importance", ascending=False)

    feature_importance_df.to_csv(
        REPORTS_DIR / "feature_importance_gradient_boosting.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=feature_importance_df,
        x="Importance",
        y="Feature",
    )
    plt.title("Feature Importance - Gradient Boosting")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_gradient_boosting.png")
    plt.close()

# Save final evaluation summary
with open(REPORTS_DIR / "final_evaluation_summary.txt", "w", encoding="utf-8") as file:
    file.write("FINAL MODEL EVALUATION SUMMARY\n")
    file.write("-" * 50 + "\n\n")

    file.write("Final test set performance:\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

    file.write("Fairness analysis by machine type:\n")
    file.write(fairness_df.to_string(index=False))
    file.write("\n")

print("\nFinal evaluation reports saved in:")
print(REPORTS_DIR)

print("\nFinal evaluation figures saved in:")
print(FIGURES_DIR)
