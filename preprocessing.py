"""
Data Preprocessing

This file loads the cleaned dataset, selects features and target,
splits the data, applies encoding and scaling, applies SMOTE to the
training data, and saves the preprocessing object.
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Set project folders
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Create folders if they do not already exist
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Load the cleaned dataset created in Step 2
df = pd.read_csv(DATA_DIR / "ai4i2020_cleaned.csv")

print("\nCleaned dataset shape:")
print(df.shape)

# Select input features used for machine failure prediction
feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Select the target column that the model will predict
target_column = "Machine failure"

X = df[feature_columns]
y = df[target_column]

# Define categorical and numerical columns
categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Split the dataset into training and testing sets
# Stratify keeps the same failure ratio in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

print("\nTraining target distribution before SMOTE:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Create preprocessing pipeline
# OneHotEncoder converts machine Type into numerical columns
# StandardScaler scales numerical values to a similar range
preprocessor = ColumnTransformer(
    transformers=[
        ("type_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num_scaler", StandardScaler(), numerical_features),
    ]
)

# Fit the preprocessor on training data and transform training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

# Apply SMOTE only to the training data
# This balances the minority failure class without changing the test set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print("\nTraining target distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Save the fitted preprocessor for later use if needed
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

# Save preprocessing summary for report evidence
with open(REPORTS_DIR / "preprocessing_summary.txt", "w", encoding="utf-8") as file:
    file.write("PREPROCESSING SUMMARY\n")
    file.write("-" * 50 + "\n\n")

    file.write(f"Cleaned dataset shape: {df.shape}\n\n")

    file.write("Selected feature columns:\n")
    for column in feature_columns:
        file.write(f"- {column}\n")

    file.write(f"\nTarget column: {target_column}\n\n")

    file.write(f"Training set shape: {X_train.shape}\n")
    file.write(f"Testing set shape: {X_test.shape}\n\n")

    file.write("Training target distribution before SMOTE:\n")
    file.write(str(y_train.value_counts()))
    file.write("\n\n")

    file.write("Testing target distribution:\n")
    file.write(str(y_test.value_counts()))
    file.write("\n\n")

    file.write("Training target distribution after SMOTE:\n")
    file.write(str(pd.Series(y_train_smote).value_counts()))
    file.write("\n\n")

    file.write("Preprocessing steps used:\n")
    file.write("- One-hot encoding for Type column\n")
    file.write("- Standard scaling for numerical columns\n")
    file.write("- SMOTE applied only on the training set\n")

print("\nPreprocessor saved at:")
print(MODELS_DIR / "preprocessor.pkl")

print("\nPreprocessing summary saved at:")
print(REPORTS_DIR / "preprocessing_summary.txt")
