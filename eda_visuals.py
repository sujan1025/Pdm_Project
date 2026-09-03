import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File and output locations
DATA_PATH = Path("data/ai4i2020.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

# Create report folders if they do not exist
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    print("\nStarting Step 2.2: EDA Visualisation...")

    # Important dataset columns
    target_col = "Machine failure"
    type_col = "Type"
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    # Numerical features used for analysis
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # 1. Show machine failure distribution
    failure_counts = df[target_col].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    failure_counts.plot(kind="bar")
    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Number of Records")
    plt.xticks(ticks=[0, 1], labels=["No Failure (0)", "Failure (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "machine_failure_distribution.png")
    plt.close()

    # 2. Show product type distribution
    product_counts = df[type_col].value_counts()

    plt.figure(figsize=(6, 4))
    product_counts.plot(kind="bar")
    plt.title("Product Type Distribution")
    plt.xlabel("Product Type")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "product_type_distribution.png")
    plt.close()

    # 3. Show counts of each failure type
    failure_mode_counts = df[failure_modes].sum()

    plt.figure(figsize=(6, 4))
    failure_mode_counts.plot(kind="bar")
    plt.title("Failure Mode Counts")
    plt.xlabel("Failure Mode")
    plt.ylabel("Number of Failures")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_mode_counts.png")
    plt.close()

    # 4. Create histogram for each numerical feature
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Create a suitable file name for each graph
        file_name = (
            col.replace(" ", "_")
               .replace("[", "")
               .replace("]", "")
               .replace("/", "_")
        )

        plt.savefig(FIGURES_DIR / f"{file_name}_histogram.png")
        plt.close()

    # 5. Save average feature values for failure and non-failure records
    mean_by_failure = df.groupby(target_col)[numerical_cols].mean()
    mean_by_failure.to_csv(REPORTS_DIR / "mean_values_by_failure_status.csv")

    # 6. Save the main EDA findings as a text report
    summary_path = REPORTS_DIR / "eda_findings.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EDA FINDINGS\n")
        f.write("=====================\n\n")

        f.write("Dataset shape:\n")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")

        f.write("Missing values:\n")
        f.write(f"{df.isnull().sum().sum()} total missing values\n\n")

        f.write("Duplicate rows:\n")
        f.write(f"{df.duplicated().sum()} duplicate rows\n\n")

        f.write("Machine failure distribution:\n")
        f.write(df[target_col].value_counts().sort_index().to_string())

        f.write("\n\nMachine failure percentage:\n")
        f.write(
            (
                df[target_col]
                .value_counts(normalize=True)
                .sort_index() * 100
            ).round(2).to_string()
        )
        f.write("\n\n")

        f.write("Product type distribution:\n")
        f.write(product_counts.to_string())
        f.write("\n\n")

        f.write("Failure mode counts:\n")
        f.write(failure_mode_counts.to_string())
        f.write("\n\n")

        f.write("Numerical feature summary:\n")
        f.write(df[numerical_cols].describe().to_string())
        f.write("\n\n")

        f.write("Interpretation:\n")
        f.write("- The dataset contains 10,000 records and 14 columns.\n")
        f.write("- There are no missing values and no duplicate rows.\n")
        f.write("- The target variable is highly imbalanced.\n")
        f.write("- Numerical histograms were created for the five main sensor features.\n")
        f.write("- These plots help understand feature distributions before preprocessing.\n")

    # Show completion information
    print("\nEDA VISUALISATION COMPLETED")
    print("------------------------------------")
    print("Charts saved inside: reports/figures")
    print("EDA summary saved as: reports/eda_findings.txt")
    print("Mean values saved as: reports/mean_values_by_failure_status.csv")


# Run the script directly
if __name__ == "__main__":
    main()