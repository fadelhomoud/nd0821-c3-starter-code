from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    '''
    This test that the message in the root page is as expected
    '''
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() ==  "Welcome to the inference API!"


def test_post_more_than_50k():
    '''
    This test that this input will result in an inference of >50k salary
    '''
    body = {
        'age': 35,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'Masters',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 50000,
        'capital-loss': 3000,
        'hours-per-week': 50,
        'native-country': 'England'
    }

    r = client.post("/inference", json=body)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], list)
    assert r.json()["prediction"][0] == ">50K"


def test_post_less_than_50k():
    '''
    This test that this input will result in an inference of<50k salary
    '''
    body = {
        'age': 20,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'Masters',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 10,
        'native-country': 'England'
    }
    r = client.post("/inference", json=body)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], list)
    assert r.json()["prediction"][0] == "<=50K"


def test_post_input():
    '''
    This test that this input will result in an error given gender and country are not valid inputs
    '''    
    body = {
        'age': 20,
        'workclass': 'Private',
        'fnlwgt': 215646,
        'education': 'Masters',
        'education-num': 12,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'gender': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 10,
        'country': 'England'
    }

    r = client.post("/inference", data=body)
    assert r.status_code != 200