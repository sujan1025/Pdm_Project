from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# Paths
DATA_PATH = Path("data/ai4i2020_cleaned.csv")
MODEL_PATH = Path("models/best_model.pkl")

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nSTEP 5: FINAL MODEL EVALUATION AND FAIRNESS ANALYSIS")
    print("----------------------------------------------------")

    if not DATA_PATH.exists():
        print("ERROR: Cleaned dataset not found.")
        print("Please run Step 2 first.")
        return

    if not MODEL_PATH.exists():
        print("ERROR: Best model not found.")
        print("Please run Step 4 first.")
        return

    # Load dataset and best model
    df = pd.read_csv(DATA_PATH)
    best_model = joblib.load(MODEL_PATH)

    print("Cleaned dataset loaded:", df.shape)
    print("Best model loaded:", MODEL_PATH)

    # Features and target
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    target_column = "Machine failure"

    X = df[feature_columns]
    y = df[target_column]

    # Use same 80:20 stratified split as Step 4
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Predict on test data only
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Final metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nFINAL TEST SET PERFORMANCE")
    print("--------------------------")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    # Save final metrics
    final_metrics = pd.DataFrame([
        {
            "Model": "Gradient Boosting",
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
        }
    ])

    final_metrics.to_csv(REPORTS_DIR / "final_model_metrics.csv", index=False)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "final_classification_report.csv")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Failure", "Actual Failure"],
        columns=["Predicted No Failure", "Predicted Failure"],
    )
    cm_df.to_csv(REPORTS_DIR / "final_confusion_matrix.csv")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"],
    )
    disp.plot()
    plt.title("Final Confusion Matrix - Gradient Boosting")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_confusion_matrix_gradient_boosting.png")
    plt.close()

    # ROC curve
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Final ROC Curve - Gradient Boosting")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_roc_curve_gradient_boosting.png")
    plt.close()

    # Fairness analysis across product types L, M, H
    fairness_data = X_test.copy()
    fairness_data["Actual"] = y_test.values
    fairness_data["Predicted"] = y_pred
    fairness_data["Failure Probability"] = y_proba

    fairness_results = []

    for machine_type in sorted(fairness_data["Type"].unique()):
        subset = fairness_data[fairness_data["Type"] == machine_type]

        actual = subset["Actual"]
        predicted = subset["Predicted"]
        probability = subset["Failure Probability"]

        # ROC-AUC needs both classes to exist in the subgroup
        if actual.nunique() == 2:
            subgroup_roc_auc = roc_auc_score(actual, probability)
        else:
            subgroup_roc_auc = None

        fairness_results.append(
            {
                "Machine Type": machine_type,
                "Total Test Records": len(subset),
                "Actual Failures": int(actual.sum()),
                "Predicted Failures": int(predicted.sum()),
                "Accuracy": round(accuracy_score(actual, predicted), 4),
                "Precision": round(precision_score(actual, predicted, zero_division=0), 4),
                "Recall": round(recall_score(actual, predicted, zero_division=0), 4),
                "F1-score": round(f1_score(actual, predicted, zero_division=0), 4),
                "ROC-AUC": round(subgroup_roc_auc, 4) if subgroup_roc_auc is not None else "N/A",
            }
        )

    fairness_df = pd.DataFrame(fairness_results)
    fairness_df.to_csv(REPORTS_DIR / "fairness_analysis_by_machine_type.csv", index=False)

    print("\nFAIRNESS ANALYSIS BY MACHINE TYPE")
    print("---------------------------------")
    print(fairness_df)

    # Fairness chart: Recall by machine type
    plt.figure(figsize=(6, 4))
    plt.bar(fairness_df["Machine Type"], fairness_df["Recall"])
    plt.title("Recall by Machine Type")
    plt.xlabel("Machine Type")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_recall_by_machine_type.png")
    plt.close()

    # Fairness chart: F1-score by machine type
    plt.figure(figsize=(6, 4))
    plt.bar(fairness_df["Machine Type"], fairness_df["F1-score"])
    plt.title("F1-score by Machine Type")
    plt.xlabel("Machine Type")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_f1_by_machine_type.png")
    plt.close()

    # Feature importance, if available
    try:
        classifier = best_model.named_steps["classifier"]
        preprocessor = best_model.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()
        cleaned_feature_names = [
            name.replace("num__", "").replace("cat__", "")
            for name in feature_names
        ]

        importances = classifier.feature_importances_

        feature_importance_df = pd.DataFrame(
            {
                "Feature": cleaned_feature_names,
                "Importance": importances,
            }
        ).sort_values(by="Importance", ascending=False)

        feature_importance_df.to_csv(
            REPORTS_DIR / "feature_importance_gradient_boosting.csv",
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.barh(
            feature_importance_df["Feature"],
            feature_importance_df["Importance"],
        )
        plt.title("Feature Importance - Gradient Boosting")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance_gradient_boosting.png")
        plt.close()

        print("\nFeature importance saved successfully.")

    except Exception as e:
        print("\nFeature importance could not be created.")
        print("Reason:", e)

    # Save written evaluation summary
    summary_path = REPORTS_DIR / "final_evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 5 FINAL EVALUATION SUMMARY\n")
        f.write("===============================\n\n")

        f.write("Selected final model:\n")
        f.write("Gradient Boosting\n\n")

        f.write("Final test set metrics:\n")
        f.write(final_metrics.to_string(index=False))
        f.write("\n\n")

        f.write("Confusion matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\n")

        f.write("Fairness analysis by machine type:\n")
        f.write(fairness_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation:\n")
        f.write(
            "The final Gradient Boosting model was evaluated on the unchanged test set. "
            "The test set was not oversampled, so the results represent realistic performance "
            "on imbalanced data. Recall and F1-score are important because missed machine failures "
            "are more serious than false alarms. Fairness analysis was performed across L, M and H "
            "machine types to check whether performance differed across product categories.\n"
        )

    print("\nSTEP 5 COMPLETED SUCCESSFULLY")
    print("-----------------------------")
    print("Final metrics saved at: reports/final_model_metrics.csv")
    print("Final classification report saved at: reports/final_classification_report.csv")
    print("Final confusion matrix saved at: reports/final_confusion_matrix.csv")
    print("Fairness analysis saved at: reports/fairness_analysis_by_machine_type.csv")
    print("Final evaluation summary saved at: reports/final_evaluation_summary.txt")
    print("Figures saved inside: reports/figures")


if __name__ == "__main__":
    main()