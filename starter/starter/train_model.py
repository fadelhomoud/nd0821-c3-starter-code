# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, compute_model_metrics, inference, slice_performance
from starter.ml.data import process_data

# Add code to load in the data.
data = pd.read_csv("starter/data/cleaned_census.csv")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"Precision = {precision} \nRecall = {recall}\nFbeta = {fbeta}")


X_slice, y_slice, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

slices_df = slice_performance(data, X_slice, y_slice, model, cat_features)
with open("slice_output.txt", 'w') as file:
    print(slices_df.to_string(), file=file)


with open("./model/inference_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("./model/onehot_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

with open("./model/label_encoder.pkl", "wb") as file:
    pickle.dump(lb, file)
