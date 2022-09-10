# -*- coding: utf-8 -*-
"""
Created 09 09 2022

@author: R. LAMJOUN
"""

import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# loading the saved model
model = joblib.load('./model/model.sav')


@app.get("/")
async def get_items():
    """
    GET on the root giving a welcome message.
    """
    return {"message": "Hi, here function to predict..."}
    
    
@app.post('/iris_prediction')
def iris_prediction(input_parameters: ModelInput):
    #
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    #
    l_sepal_length = input_dictionary['sepal_length']
    l_sepal_width = input_dictionary['sepal_width']
    l_petal_length = input_dictionary['petal_length']
    l_petal_width = input_dictionary['petal_width']
    #
    input_list = [l_sepal_length, l_sepal_width, l_petal_length, l_petal_width]
    #
    #
    df = pd.DataFrame([input_list], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    prediction = model.predict(df)
    
    if prediction[0] == 0:
        return "setosa"
    elif (prediction[0] == 1):
        return "versicolor"
    elif (prediction[0] == 2):
        return  "virginica"
    else:
        return "no decision!!"


