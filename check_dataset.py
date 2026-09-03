import pandas as pd
from pathlib import Path

# Location of the dataset
DATA_PATH = Path("data/ai4i2020.csv")


def main():
    # Check whether the dataset exists
    if not DATA_PATH.exists():
        print("Dataset not found.")
        return

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Show the first 5 records
    print("\n1. FIRST 5 ROWS")
    print(df.head())

    # Show dataset size
    print("\n2. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Show all column names
    print("\n3. COLUMN NAMES")
    print(df.columns.tolist())

    # Show the type of data in each column
    print("\n4. DATA TYPES")
    print(df.dtypes)

    # Check for missing values
    print("\n5. MISSING VALUES")
    print(df.isnull().sum())

    # Check for duplicate records
    print("\n6. DUPLICATE ROWS")
    print(df.duplicated().sum())

    # Show number and percentage of machine failures
    print("\n7. MACHINE FAILURE DISTRIBUTION")
    print(df["Machine failure"].value_counts())
    print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

    # Show number of machines in each product type
    print("\n8. PRODUCT TYPE DISTRIBUTION")
    print(df["Type"].value_counts())

    # Count each type of machine failure
    print("\n9. FAILURE MODE COUNTS")
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    print(df[failure_modes].sum())

    # Main numerical features used for analysis
    print("\n10. NUMERICAL SUMMARY")
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    # Show basic statistics such as mean, minimum and maximum
    print(df[numerical_cols].describe())


# Run the program
if __name__ == "__main__":
    main()