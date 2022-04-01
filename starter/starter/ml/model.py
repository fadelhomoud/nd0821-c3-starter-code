import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=221)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds, as_df=False):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def slice_performance(data, X, Y, model, cat_features):
    data['preds'] = inference(model, X)
    data['y_true'] = Y
    perf_df = pd.DataFrame()
    for category in cat_features:
        for val in data[category].unique():

            slice_df = data[data[category] == val]
            precision, recall, fbeta = compute_model_metrics(
                slice_df['y_true'], slice_df['preds'])

            perf_df = pd.concat([perf_df, pd.DataFrame(
                {"Category": [category],
                 'Value': [val],
                 'precision': [precision],
                 'recall': [recall],
                 'fbeta': [fbeta]})
            ])

    return perf_df.reset_index(drop=True)
