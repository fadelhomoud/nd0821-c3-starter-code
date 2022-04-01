"""
Code for checking model performance on different slices
"""

import pickle
import os

import pandas as pd

from ml.model import compute_model_metrics, inference
from ml.data import process_data
from train_model import cat_features


def check_slices_performance(data, model_path):

    # Loading model and encoders:
    with open(os.path.join(model_path, "inference_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(os.path.join(model_path, "onehot_encoder.pkl"), "rb") as file:
        encoder = pickle.load(file)

    with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as file:
        lb = pickle.load(file)

    # Processing data:
    X_val, y_val, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Measuring model performance:
    predictions = inference(model, X_val)

    data["pred"] = predictions
    data["label"] = lb.transform(data["salary"])

    # Calculating slice results:
    slice_results = pd.DataFrame()
    for group in cat_features:

        performance = data.groupby(group).apply(
            lambda df: compute_model_metrics(
                df["label"], df["pred"], as_df=True))
        performance = performance.droplevel(1)
        performance.index.name = "group_value"
        performance["group"] = group
        slice_results = slice_results.append(performance)

    slice_results = slice_results.reset_index()
    slice_results = slice_results[[
        "group", "group_value", "precision", "recall", "fbeta"]]

    return slice_results


if __name__ == "__main__":

    # Loading the data:
    data = pd.read_csv("./starter/data/cleaned_census.csv")
    model_path = "./starter/model"

    # Calculating slice performance:
    output = check_slices_performance(data, model_path)

    # Printing results to file:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_rows', None)
    with open("slice_output.txt", 'w') as file:
        print(output.to_string(), file=file)
