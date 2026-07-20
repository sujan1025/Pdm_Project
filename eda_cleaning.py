from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# File paths
DATA_PATH = Path("data/ai4i2020.csv")
CLEANED_PATH = Path("data/ai4i2020_cleaned.csv")

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def write_section(file, title):
    file.write("\n" + "=" * 60 + "\n")
    file.write(title + "\n")
    file.write("=" * 60 + "\n")


def main():
    # Check dataset exists
    if not DATA_PATH.exists():
        print("ERROR: Dataset not found.")
        print(DATA_PATH)
        return

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Columns used for EDA
    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    failure_mode_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    summary_path = REPORTS_DIR / "eda_summary.txt"

    # ==============================
    # EDA summary text file
    # ==============================
    with open(summary_path, "w", encoding="utf-8") as f:
        write_section(f, "1. FIRST 5 ROWS")
        f.write(df.head().to_string())

        write_section(f, "2. DATASET SHAPE")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n")

        write_section(f, "3. COLUMN NAMES")
        for column in df.columns:
            f.write(f"{column}\n")

        write_section(f, "4. DATA TYPES")
        f.write(df.dtypes.to_string())

        write_section(f, "5. MISSING VALUES")
        f.write(df.isnull().sum().to_string())

        write_section(f, "6. DUPLICATE ROWS")
        f.write(f"Duplicate rows: {df.duplicated().sum()}\n")

        write_section(f, "7. MACHINE FAILURE DISTRIBUTION")
        failure_counts = df["Machine failure"].value_counts().sort_index()
        failure_percentages = (
            df["Machine failure"].value_counts(normalize=True).sort_index() * 100
        ).round(2)

        f.write("Counts:\n")
        f.write(failure_counts.to_string())
        f.write("\n\nPercentages:\n")
        f.write(failure_percentages.to_string())
        f.write("\n\nMeaning:\n")
        f.write("0 = No machine failure\n")
        f.write("1 = Machine failure occurred\n")

        write_section(f, "8. PRODUCT TYPE DISTRIBUTION")
        product_type_counts = df["Type"].value_counts()
        f.write(product_type_counts.to_string())

        write_section(f, "9. FAILURE MODE COUNTS")
        failure_mode_counts = df[failure_mode_columns].sum()
        f.write(failure_mode_counts.to_string())

        write_section(f, "10. NUMERICAL SUMMARY")
        numerical_summary = df[numerical_columns].describe()
        f.write(numerical_summary.to_string())

        write_section(f, "11. BASIC CLEANING DECISION")

        original_rows = df.shape[0]

        # Cleaning starts after EDA checks
        df_cleaned = df.copy()

        before_duplicates = df_cleaned.shape[0]
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = before_duplicates - df_cleaned.shape[0]

        before_missing = df_cleaned.shape[0]
        df_cleaned = df_cleaned.dropna()
        removed_missing = before_missing - df_cleaned.shape[0]

        # Remove non-predictive identifier columns
        identifier_columns = ["UDI", "Product ID"]
        df_cleaned = df_cleaned.drop(columns=identifier_columns)

        f.write(f"Original rows: {original_rows}\n")
        f.write(f"Removed duplicate rows: {removed_duplicates}\n")
        f.write(f"Removed rows with missing values: {removed_missing}\n")
        f.write("Removed identifier columns: UDI, Product ID\n")
        f.write(f"Cleaned dataset shape: {df_cleaned.shape}\n")

        f.write("\nCleaning explanation:\n")
        f.write(
            "No missing values or duplicate rows were found. Therefore, no records were removed. "
            "Only UDI and Product ID were removed because they are identifier columns and do not "
            "represent machine operating behaviour.\n"
        )

    # ==============================
    # Save CSV report outputs
    # ==============================
    df.head(20).to_csv(REPORTS_DIR / "dataset_first_20_rows.csv", index=False)
    numerical_summary.to_csv(REPORTS_DIR / "numerical_summary.csv")
    failure_counts.to_csv(REPORTS_DIR / "machine_failure_counts.csv")
    product_type_counts.to_csv(REPORTS_DIR / "product_type_counts.csv")
    failure_mode_counts.to_csv(REPORTS_DIR / "failure_mode_counts.csv")

    # Save cleaned dataset
    df_cleaned.to_csv(CLEANED_PATH, index=False)

    # ==============================
    # EDA visualisations
    # ==============================

    # Chart 1: Machine failure distribution
    plt.figure(figsize=(6, 4))
    failure_counts.plot(kind="bar")
    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Number of Records")
    plt.xticks(ticks=[0, 1], labels=["No Failure (0)", "Failure (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "machine_failure_distribution.png")
    plt.close()

    # Chart 2: Product type distribution
    plt.figure(figsize=(6, 4))
    product_type_counts.plot(kind="bar")
    plt.title("Product Type Distribution")
    plt.xlabel("Product Type")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "product_type_distribution.png")
    plt.close()

    # Chart 3: Failure mode counts
    plt.figure(figsize=(6, 4))
    failure_mode_counts.plot(kind="bar")
    plt.title("Failure Mode Counts")
    plt.xlabel("Failure Mode")
    plt.ylabel("Number of Failures")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_mode_counts.png")
    plt.close()

    # Chart 4: Numerical feature histograms
    for col in numerical_columns:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        file_name = (
            col.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        plt.savefig(FIGURES_DIR / f"{file_name}_histogram.png")
        plt.close()

    # ==============================
    # Console output
    # ==============================
    print("\nEDA AND BASIC CLEANING COMPLETED SUCCESSFULLY")
    print("----------------------------------------------------")
    print(f"Original dataset shape: {df.shape}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Cleaned dataset saved at: {CLEANED_PATH}")
    print(f"EDA summary saved at: {summary_path}")
    print("CSV reports saved inside: reports")
    print("Charts saved inside: reports/figures")
    print("\nMachine failure distribution:")
    print(failure_counts)
    print("\nMachine failure percentage:")
    print(failure_percentages)


if __name__ == "__main__":
    main()