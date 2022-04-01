
import pandas as pd
import numpy as np
import pytest
from starter.ml.model import train_model, compute_model_metrics, inference, slice_performance


@pytest.fixture
def data():
    '''
    Dummay data for testing only
    '''
    df = pd.DataFrame(
        {
            'x1': [1, 3, -3, 5, -1, 4],
            'x2': [2, 4, -2, 6, -1, 4],
            'x3': [0, 4, 6, 3, 7, 3],
            'y': [1, 0, 0, 0, 1, 1]
        }
    )
    return df


def test_model_train(data):
    '''
    Test that the model will have the same number of features
    and classes as the input data
    '''

    model = train_model(data.drop(columns=['y']), data.y)

    assert all([model.n_classes_ == 2, model.n_features_in_ == 3])


def test_metrics():
    '''
    Test that the metric calculations are correct using dummay data
    '''
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    prec, rec, fb = compute_model_metrics(y, y_pred)

    assert all([prec == 0.5, rec == 1, round(fb, 3) == 0.667])


def test_inference(data):
    '''
    Test that the inference function will return a numpy array
    with the same length as the input
    '''
    model = train_model(data.drop(columns=['y']), data.y)

    pred = inference(model, data.drop(columns=['y']))

    assert all([isinstance(pred, np.ndarray), len(pred) == data.shape[0]])
