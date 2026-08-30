import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from preprocessing.preprocess import (
    load_data,
    preprocess_data
)

from models.model import create_model

from evaluation.metrics import (
    calculate_metrics
)

from evaluation.fairness import (
    calculate_group_metrics,
    calculate_fairness_gaps
)


RESULTS_DIR = "results"


def main():

    print("\n==============================")
    print("AI MODEL BIAS DETECTION")
    print("==============================")

    # -------------------------
    # Load data
    # -------------------------

    df = load_data()

    print(
        f"\nDataset size: {df.shape}"
    )

    # -------------------------
    # Preprocessing
    # -------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        group_train,
        group_test,
        scaler
    ) = preprocess_data(df)

    # -------------------------
    # Create model
    # -------------------------

    model = create_model()

    # -------------------------
    # Train
    # -------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # -------------------------
    # Predictions
    # -------------------------

    y_pred = model.predict(
        X_test
    )

    # -------------------------
    # Overall metrics
    # -------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    print("\n========== Overall Metrics ==========")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    # -------------------------
    # Group metrics
    # -------------------------

    group_metrics = calculate_group_metrics(
        y_test,
        y_pred,
        group_test
    )

    print(
        "\n========== Group Metrics =========="
    )

    print(
        group_metrics.to_string(
            index=False
        )
    )

    # -------------------------
    # Fairness gaps
    # -------------------------

    fairness_gaps = calculate_fairness_gaps(
        group_metrics
    )

    print(
        "\n========== Fairness Gaps =========="
    )

    print(
        fairness_gaps.to_string(
            index=False
        )
    )

    # -------------------------
    # Save results
    # -------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    metrics_df = pd.DataFrame({

        "metric": list(
            metrics.keys()
        ),

        "value": list(
            metrics.values()
        )
    })

    metrics_df.to_csv(
        "results/metrics.csv",
        index=False
    )

    fairness_output = pd.concat(
        [
            group_metrics,
            fairness_gaps
        ],
        axis=1
    )

    fairness_output.to_csv(
        "results/fairness_report.csv",
        index=False
    )

    print(
        "\nResults saved successfully."
    )

    print(
        "results/metrics.csv"
    )

    print(
        "results/fairness_report.csv"
    )


if __name__ == "__main__":

    main()