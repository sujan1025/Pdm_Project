import pandas as pd
from pathlib import Path

# Dataset path
DATA_PATH = Path("data/ai4i2020.csv")

def main():
    # Check dataset exists
    if not DATA_PATH.exists():
        print("ERROR: Dataset not found.")
        print("Make sure the file is saved as data/ai4i2020.csv")
        return

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("\n==============================")
    print("FIRST 5 ROWS")
    print("==============================")
    print(df.head())

    print("\n==============================")
    print("DATASET SHAPE")
    print("==============================")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n==============================")
    print("COLUMN NAMES")
    print("==============================")
    print(df.columns.tolist())

    print("\n==============================")
    print("DATA TYPES")
    print("==============================")
    print(df.dtypes)

    print("\n==============================")
    print("MISSING VALUES")
    print("==============================")
    print(df.isnull().sum())

    print("\n==============================")
    print("DUPLICATE ROWS")
    print("==============================")
    print(df.duplicated().sum())

    print("\n==============================")
    print("MACHINE FAILURE DISTRIBUTION")
    print("==============================")
    print(df["Machine failure"].value_counts())
    print("\nPercentage:")
    print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    print("\nMeaning:")
    print("0 = No machine failure")
    print("1 = Machine failure occurred")

    print("\n==============================")
    print("PRODUCT TYPE DISTRIBUTION")
    print("==============================")
    print(df["Type"].value_counts())

    print("\n==============================")
    print("FAILURE MODE COUNTS")
    print("==============================")
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    print(df[failure_modes].sum())

    print("\n==============================")
    print("NUMERICAL SUMMARY")
    print("==============================")
    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    print(df[numerical_columns].describe())

    print("\nStep 2 completed successfully.")


if __name__ == "__main__":
    main()