import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "dataset.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


def preprocess_data(df):

    # Keep group separately for fairness analysis
    groups = df["group"].copy()

    # Features
    X = df[
        [
            "age",
            "income",
            "credit_score",
            "debt_ratio",
            "employment_years"
        ]
    ]

    # Target
    y = df["approved"]

    # Train/test split
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X,
        y,
        groups,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        group_train,
        group_test,
        scaler
    )


if __name__ == "__main__":

    df = load_data()

    print("Dataset shape:")
    print(df.shape)

    print("\nDataset preview:")
    print(df.head())

    print("\nGroup distribution:")
    print(df["group"].value_counts())

    print("\nApproval distribution:")
    print(df["approved"].value_counts())