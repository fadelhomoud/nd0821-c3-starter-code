# Put the code for your API here.

import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.data import process_data
from starter.train_model import cat_features


# Loading model and encoders:
with open("./model/inference_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/onehot_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("model/label_encoder.pkl", "rb") as file:
    lb = pickle.load(file)


app = FastAPI()


class PostBody(BaseModel):

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                'age': 27,
                'workclass': 'Private',
                'fnlwgt': 123456,
                'education': 'HS-grad',
                'education-num': 7,
                'marital-status': 'Never-married',
                'occupation': 'Other-service',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Female',
                'capital-gain': 100000,
                'capital-loss': 5000,
                'hours-per-week': 45,
                'native-country': 'England'
            }
        }


@app.get("/")
async def return_greetings():
    return "Welcome to the inference API!"


@app.post("/inference")
async def model_inference(data: PostBody):

    data = data.dict(by_alias=True)

    data, y, encoder_ret, lb_ret = process_data(
        pd.DataFrame(data, index=[0]),
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)

    pred = model.predict(data)
    pred = lb.inverse_transform(pred)

    return {"prediction": pred.tolist()}
