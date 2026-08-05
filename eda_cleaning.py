"""
Exploratory Data Analysis and Basic Cleaning

This file loads the AI4I 2020 dataset, checks data quality,
creates EDA reports, checks outliers, saves visualisations,
and creates a cleaned dataset for the next preprocessing step.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Set project folders
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("figures")

# Create folders if they do not already exist
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Set dataset path
DATASET_PATH = DATA_DIR / "ai4i2020.csv"

# Load the original AI4I dataset
df = pd.read_csv(DATASET_PATH)

print("STEP 2: EDA AND CLEANING STARTED")
print("-" * 50)

# Display the first few rows to understand the dataset structure
print("\nFirst 5 rows:")
print(df.head())

# Check the number of rows and columns in the dataset
print("\nDataset shape:")
print(df.shape)

# Check data types of all columns
print("\nData types:")
print(df.dtypes)

# Check missing values in each column
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values)

# Check duplicate rows in the dataset
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:")
print(duplicate_count)

# Check the target class distribution
target_counts = df["Machine failure"].value_counts()
target_percentages = df["Machine failure"].value_counts(normalize=True) * 100

print("\nMachine failure distribution:")
print(target_counts)

print("\nMachine failure percentages:")
print(target_percentages)

# Check product type distribution
type_counts = df["Type"].value_counts()
print("\nProduct type distribution:")
print(type_counts)

# Check failure mode counts
failure_mode_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_mode_counts = df[failure_mode_columns].sum()

print("\nFailure mode counts:")
print(failure_mode_counts)

# Select numerical columns for summary, visualisation and outlier checking
numerical_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Create numerical summary for sensor and operating features
numerical_summary = df[numerical_columns].describe()

print("\nNumerical summary:")
print(numerical_summary)

# Check outliers in numerical columns using the IQR method
# Outliers are checked because unusual machine readings may indicate failure risk
outlier_summary = []

for column in numerical_columns:
    # Calculate first quartile and third quartile
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    # Calculate interquartile range
    iqr = q3 - q1

    # Calculate lower and upper outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count values below and above the outlier boundaries
    lower_outliers = df[df[column] < lower_bound]
    upper_outliers = df[df[column] > upper_bound]

    total_outliers = len(lower_outliers) + len(upper_outliers)

    # Store the outlier result for this feature
    outlier_summary.append(
        {
            "Feature": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": total_outliers,
            "Outlier Percentage": round((total_outliers / len(df)) * 100, 2),
        }
    )

# Convert outlier results into a DataFrame
outlier_summary_df = pd.DataFrame(outlier_summary)

print("\nOutlier summary:")
print(outlier_summary_df)

# Save first 20 rows for report evidence
df.head(20).to_csv(REPORTS_DIR / "dataset_first_20_rows.csv", index=False)

# Save numerical summary
numerical_summary.to_csv(REPORTS_DIR / "numerical_summary.csv")

# Save target distribution
target_counts.to_csv(REPORTS_DIR / "machine_failure_counts.csv")

# Save product type distribution
type_counts.to_csv(REPORTS_DIR / "product_type_counts.csv")

# Save failure mode counts
failure_mode_counts.to_csv(REPORTS_DIR / "failure_mode_counts.csv")

# Save outlier summary
outlier_summary_df.to_csv(REPORTS_DIR / "outlier_summary.csv", index=False)

# Save full EDA summary into a text file
with open(REPORTS_DIR / "eda_summary.txt", "w", encoding="utf-8") as file:
    file.write("EDA AND CLEANING SUMMARY\n")
    file.write("-" * 50 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n\n")

    file.write("Data types:\n")
    file.write(str(df.dtypes))
    file.write("\n\n")

    file.write("Missing values:\n")
    file.write(str(missing_values))
    file.write("\n\n")

    file.write(f"Duplicate rows: {duplicate_count}\n\n")

    file.write("Machine failure distribution:\n")
    file.write(str(target_counts))
    file.write("\n\n")

    file.write("Machine failure percentages:\n")
    file.write(str(target_percentages))
    file.write("\n\n")

    file.write("Product type distribution:\n")
    file.write(str(type_counts))
    file.write("\n\n")

    file.write("Failure mode counts:\n")
    file.write(str(failure_mode_counts))
    file.write("\n\n")

    file.write("Numerical summary:\n")
    file.write(numerical_summary.to_string())
    file.write("\n\n")

    file.write("Outlier summary using IQR method:\n")
    file.write(outlier_summary_df.to_string(index=False))
    file.write("\n\n")

    file.write("Outlier handling decision:\n")
    file.write(
        "Outliers were checked but not removed automatically because unusual machine operating values may represent real abnormal behaviour linked "
        "to machine failure risk.\n\n"
    )

# Create histogram charts for numerical columns
for column in numerical_columns:
    # Clean the column name so it can be used as a file name
    file_name = (
        column.replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
        .replace("/", "_")
    )

    # Create a histogram to show the distribution of the feature
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{file_name}_histogram.png")
    plt.close()

# Create boxplots to visually inspect outliers in numerical features
for column in numerical_columns:
    # Clean the column name so it can be used as a file name
    file_name = (
        column.replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
        .replace("/", "_")
    )

    # Create a boxplot to show possible outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{file_name}_boxplot.png")
    plt.close()

# Create target distribution chart
plt.figure(figsize=(6, 4))
sns.countplot(x="Machine failure", data=df)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "machine_failure_distribution.png")
plt.close()

# Create product type distribution chart
plt.figure(figsize=(6, 4))
sns.countplot(x="Type", data=df)
plt.title("Product Type Distribution")
plt.xlabel("Product Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "product_type_distribution.png")
plt.close()

# Create failure mode count chart
plt.figure(figsize=(7, 4))
failure_mode_counts.plot(kind="bar")
plt.title("Failure Mode Counts")
plt.xlabel("Failure Mode")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "failure_mode_counts.png")
plt.close()

# Create correlation heatmap for numerical columns and target
correlation_columns = numerical_columns + ["Machine failure"]

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[correlation_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="Blues",
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
plt.close()

# Remove duplicate rows if any exist
df_cleaned = df.drop_duplicates()

# Remove rows with missing values if any exist
df_cleaned = df_cleaned.dropna()

# Remove identifier columns because they are not useful for prediction
df_cleaned = df_cleaned.drop(columns=["UDI", "Product ID"])

# Save the cleaned dataset for preprocessing and model training
cleaned_path = DATA_DIR / "ai4i2020_cleaned.csv"
df_cleaned.to_csv(cleaned_path, index=False)

print("\nCleaned dataset saved at:")
print(cleaned_path)

print("\nReports saved in:")
print(REPORTS_DIR)

print("\nFigures saved in:")
print(FIGURES_DIR)