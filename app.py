from flask import Flask, request, render_template_string

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)


HTML = """

<!DOCTYPE html>

<html>

<head>

<title>
AI Model Bias Detection
</title>

</head>

<body>

<h1>
AI Model Bias Detection
</h1>

<form method="POST">

<label>Age:</label>

<input
type="number"
name="age"
required
>

<br><br>

<label>Income:</label>

<input
type="number"
name="income"
required
>

<br><br>

<label>Credit Score:</label>

<input
type="number"
name="credit_score"
required
>

<br><br>

<label>Debt Ratio:</label>

<input
type="number"
step="0.01"
name="debt_ratio"
required
>

<br><br>

<label>Employment Years:</label>

<input
type="number"
step="0.1"
name="employment_years"
required
>

<br><br>

<button type="submit">
Predict
</button>

</form>

{% if prediction %}

<h2>
Prediction: {{ prediction }}
</h2>

{% endif %}

</body>

</html>

"""


def train_model():

    import pandas as pd

    df = pd.read_csv(
        "data/dataset.csv"
    )

    X = df[
        [
            "age",
            "income",
            "credit_score",
            "debt_ratio",
            "employment_years"
        ]
    ]

    y = df["approved"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(
        X_scaled,
        y
    )

    return model, scaler


model, scaler = train_model()


@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    if request.method == "POST":

        values = [

            float(
                request.form["age"]
            ),

            float(
                request.form["income"]
            ),

            float(
                request.form["credit_score"]
            ),

            float(
                request.form["debt_ratio"]
            ),

            float(
                request.form[
                    "employment_years"
                ]
            )
        ]

        X = np.array(
            [values]
        )

        X_scaled = scaler.transform(
            X
        )

        result = model.predict(
            X_scaled
        )[0]

        if result == 1:

            prediction = "Approved"

        else:

            prediction = "Not Approved"

    return render_template_string(
        HTML,
        prediction=prediction
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )