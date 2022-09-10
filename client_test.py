import json
import requests

url = 'http://127.0.0.1:8000/iris_prediction'

input_data_test = {
    'sepal_length': 7.2,
    'sepal_width': 3.2,
    'petal_length': 6.0,
    'petal_width': 1.8,
}

input_json = json.dumps(input_data_test)
response = requests.post(url, data=input_json)

print(response.text)
