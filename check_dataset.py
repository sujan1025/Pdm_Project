import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/ai4i2020.csv")

def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Make sure it is saved as data/ai4i2020.csv")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n1. FIRST 5 ROWS")
    print(df.head())

    print("\n2. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n3. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n4. DATA TYPES")
    print(df.dtypes)

    print("\n5. MISSING VALUES")
    print(df.isnull().sum())

    print("\n6. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n7. MACHINE FAILURE DISTRIBUTION")
    print(df["Machine failure"].value_counts())
    print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    print("\n8. PRODUCT TYPE DISTRIBUTION")
    print(df["Type"].value_counts())

    print("\n9. FAILURE MODE COUNTS")
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    print(df[failure_modes].sum())

    print("\n10. NUMERICAL SUMMARY")
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    print(df[numerical_cols].describe())

    print("\nStep 2.1 raw dataset checking completed.")

if __name__ == "__main__":
    main()