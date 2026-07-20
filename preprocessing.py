from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# File paths
DATA_PATH = Path("data/ai4i2020_cleaned.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # Check cleaned dataset exists
    if not DATA_PATH.exists():
        print("ERROR: Cleaned dataset not found.")
        return

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    print("\nPREPROCESSING STARTED")
    print("-----------------------------")
    print("Cleaned dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Input features for prediction
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # Target column
    target_column = "Machine failure"

    # Leakage columns: useful for EDA, but not used as prediction input
    leakage_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    print("\nInput features used for model:")
    for col in feature_columns:
        print("-", col)

    print("\nTarget column:")
    print("-", target_column)

    print("\nColumns excluded to avoid data leakage:")
    for col in leakage_columns:
        print("-", col)

    # Create X and y
    X = df[feature_columns]
    y = df[target_column]

    print("\nOriginal target distribution:")
    print(y.value_counts())
    print("\nOriginal target percentage:")
    print((y.value_counts(normalize=True) * 100).round(2))

    # Train-test split
    # Stratify keeps the same failure/non-failure ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    print("\nTraining target distribution before SMOTE:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # Define numerical and categorical features
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

    # Preprocessing:
    # - StandardScaler scales numerical features
    # - OneHotEncoder converts Type L/M/H into numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", onehot_encoder, categorical_features),
        ]
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the same fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    print("\nAfter encoding and scaling:")
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    print("\nTraining target distribution after SMOTE:")
    print(y_train_smote.value_counts())

    print("\nSMOTE explanation:")
    print("SMOTE was applied only to the training data.")
    print("The test data was not changed, so evaluation remains realistic.")

    # Save the preprocessor for later model use
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    # Save summary report
    summary_path = REPORTS_DIR / "preprocessing_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING SUMMARY\n")
        f.write("============================\n\n")

        f.write("Dataset used:\n")
        f.write(str(DATA_PATH) + "\n\n")

        f.write("Input features:\n")
        for col in feature_columns:
            f.write(f"- {col}\n")

        f.write("\nTarget column:\n")
        f.write(f"- {target_column}\n")

        f.write("\nExcluded leakage columns:\n")
        for col in leakage_columns:
            f.write(f"- {col}\n")

        f.write("\nOriginal target distribution:\n")
        f.write(y.value_counts().to_string())
        f.write("\n\nOriginal target percentage:\n")
        f.write((y.value_counts(normalize=True) * 100).round(2).to_string())

        f.write("\n\nTrain-test split:\n")
        f.write(f"Training set shape: {X_train.shape}\n")
        f.write(f"Testing set shape: {X_test.shape}\n")

        f.write("\nTraining target distribution before SMOTE:\n")
        f.write(y_train.value_counts().to_string())

        f.write("\n\nTesting target distribution:\n")
        f.write(y_test.value_counts().to_string())

        f.write("\n\nProcessed data shapes:\n")
        f.write(f"Processed training shape: {X_train_processed.shape}\n")
        f.write(f"Processed testing shape: {X_test_processed.shape}\n")

        f.write("\nTraining target distribution after SMOTE:\n")
        f.write(y_train_smote.value_counts().to_string())

        f.write("\n\nExplanation:\n")
        f.write(
            "The Type column was encoded using OneHotEncoder. Numerical machine readings "
            "were scaled using StandardScaler. SMOTE was applied only to the training data "
            "to balance the failure and non-failure classes without changing the test set. "
            "This avoids data leakage and keeps evaluation realistic.\n"
        )

    print("-----------------------------")
    print(f"Preprocessor saved at: {MODELS_DIR / 'preprocessor.pkl'}")
    print(f"Preprocessing summary saved at: {summary_path}")


if __name__ == "__main__":
    main()