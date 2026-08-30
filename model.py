from sklearn.linear_model import LogisticRegression


def create_model():

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    return model