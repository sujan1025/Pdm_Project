from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# Paths
DATA_PATH = Path("data/ai4i2020_cleaned.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print("ERROR: Cleaned dataset not found.")
        return

    print("\nMODEL TRAINING AND COMPARISON STARTED")
    print("---------------------------------------------")

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Input features
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Target
    target_column = "Machine failure"

    # These are not used as input features because they are failure-mode labels
    leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    print("\nFeatures used:")
    for col in feature_columns:
        print("-", col)

    print("\nTarget column:")
    print("-", target_column)

    print("\nExcluded leakage columns:")
    for col in leakage_columns:
        print("-", col)

    X = df[feature_columns]
    y = df[target_column]

    print("\nOriginal target distribution:")
    print(y.value_counts())

    # 80:20 stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # Feature groups
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    categorical_features = ["Type"]

    # OneHotEncoder compatibility for different scikit-learn versions
    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", onehot_encoder, categorical_features),
        ]
    )

    # Cross-validation setup
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Models and parameter grids
    models = {
        "Logistic Regression": {
            "classifier": LogisticRegression(max_iter=2000, solver="liblinear", random_state=42),
            "params": {
                "classifier__C": [0.1, 1, 10],
            },
        },
        "Random Forest": {
            "classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_split": [2, 5],
            },
        },
        "Gradient Boosting": {
            "classifier": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }

    results = []
    best_model_name = None
    best_model = None
    best_f1 = -1

    plt.figure(figsize=(7, 5))

    for model_name, model_info in models.items():
        print("\n=============================================")
        print(f"Training model: {model_name}")
        print("=============================================")

        # Pipeline:
        # preprocessing -> SMOTE -> classifier
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", model_info["classifier"]),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_info["params"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        # Train model using GridSearchCV
        grid_search.fit(X_train, y_train)

        print("\nBest parameters:")
        print(grid_search.best_params_)

        print("\nBest cross-validation F1-score:")
        print(round(grid_search.best_score_, 4))

        trained_model = grid_search.best_estimator_

        # Test set prediction
        y_pred = trained_model.predict(X_test)
        y_proba = trained_model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        print("\nTest set results:")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Failure", "Failure"],
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_model_name = model_name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_model_name}.png")
        plt.close()

        # ROC curve
        RocCurveDisplay.from_estimator(trained_model, X_test, y_test, name=model_name)

        # Save classification report
        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(REPORTS_DIR / f"classification_report_{safe_model_name}.csv")

        # Store results
        results.append(
            {
                "Model": model_name,
                "Best CV F1-score": round(grid_search.best_score_, 4),
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "Best Parameters": grid_search.best_params_,
            }
        )

        # Select best model based on test F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = trained_model

    # Save ROC curve figure for all models
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png")
    plt.close()

    # Save model comparison results
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "model_comparison_results.csv", index=False)

    # Save best model
    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, best_model_path)

    # Save best model summary
    summary_path = REPORTS_DIR / "best_model_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODEL TRAINING SUMMARY\n")
        f.write("=============================\n\n")

        f.write("Models compared:\n")
        f.write("- Logistic Regression\n")
        f.write("- Random Forest\n")
        f.write("- Gradient Boosting\n\n")

        f.write("Evaluation metrics:\n")
        f.write("- Accuracy\n")
        f.write("- Precision\n")
        f.write("- Recall\n")
        f.write("- F1-score\n")
        f.write("- ROC-AUC\n\n")

        f.write("Train-test split:\n")
        f.write("80% training and 20% testing with stratified sampling.\n\n")

        f.write("Class imbalance handling:\n")
        f.write("SMOTE was applied only inside the training pipeline.\n")
        f.write("The test set was not oversampled.\n\n")

        f.write("Best model selected by test F1-score:\n")
        f.write(f"{best_model_name}\n\n")

        f.write("Model comparison results:\n")
        f.write(results_df.to_string(index=False))

    print("\n-----------------------------")
    print("Model comparison saved at: reports/model_comparison_results.csv")
    print("Best model summary saved at: reports/best_model_summary.txt")
    print(f"Best model saved at: {best_model_path}")
    print("Confusion matrices and ROC curve saved inside: reports/figures")
    print("\nBest model:", best_model_name)
    print("Best F1-score:", round(best_f1, 4))


if __name__ == "__main__":
    main()