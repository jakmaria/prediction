from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import pytz

app = Flask(__name__)

def prepare_dataframe(data):
    df = pd.DataFrame(data, columns=['timestamp', 'value'])
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_convert(None)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    params_list = data.get('params', [])

    params = {item['name']: item['value'] for item in params_list}

    # input data preparation
    df = prepare_dataframe(data['data'])

    # Initialize the model with input parameters
    model = Prophet(**params)

    # Fit the model to the input data
    model.fit(df)

    # Create future data frames
    future = model.make_future_dataframe(periods=5, freq='W')

    # Predict future data
    forecast = model.predict(future)

    # Prepare the resulting data to send back to the client
    result = forecast[['ds', 'yhat']].to_dict(orient='records')

    return jsonify(result)
if __name__ == '__main__':
    app.run(port=5000)