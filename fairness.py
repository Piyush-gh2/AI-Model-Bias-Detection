import pandas as pd


def calculate_group_metrics(
    y_true,
    y_pred,
    groups
):

    data = pd.DataFrame({

        "actual": list(y_true),

        "predicted": list(y_pred),

        "group": list(groups)

    })

    results = []

    for group_name in sorted(
        data["group"].unique()
    ):

        group_data = data[
            data["group"] == group_name
        ]

        actual = group_data["actual"]

        predicted = group_data["predicted"]

        total = len(group_data)

        accuracy = (
            (actual == predicted).sum()
            / total
        )

        positive_predictions = (
            predicted == 1
        ).mean()

        actual_positive_rate = (
            actual == 1
        ).mean()

        true_positive = (
            (actual == 1)
            & (predicted == 1)
        ).sum()

        actual_positive = (
            actual == 1
        ).sum()

        if actual_positive > 0:

            true_positive_rate = (
                true_positive
                / actual_positive
            )

        else:

            true_positive_rate = 0

        results.append({

            "group": group_name,

            "samples": total,

            "accuracy": accuracy,

            "positive_prediction_rate":
                positive_predictions,

            "actual_positive_rate":
                actual_positive_rate,

            "true_positive_rate":
                true_positive_rate
        })

    return pd.DataFrame(results)


def calculate_fairness_gaps(
    group_metrics
):

    if len(group_metrics) < 2:

        return pd.DataFrame()

    numeric_columns = [

        "accuracy",

        "positive_prediction_rate",

        "actual_positive_rate",

        "true_positive_rate"
    ]

    gaps = {}

    for column in numeric_columns:

        maximum = group_metrics[
            column
        ].max()

        minimum = group_metrics[
            column
        ].min()

        gaps[column + "_gap"] = (
            maximum - minimum
        )

    return pd.DataFrame([gaps])