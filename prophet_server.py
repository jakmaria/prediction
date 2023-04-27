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

    # Příprava vstupních dat
    df = prepare_dataframe(data['data'])

    # Inicializace modelu s volitelnými parametry
    model = Prophet(**params)

    # Nafitování modelu na vstupní data
    model.fit(df)

    # Vytvoření budoucích datových rámců
    future = model.make_future_dataframe(periods=5, freq='W')

    # Predikce pro budoucí data
    forecast = model.predict(future)

    # Příprava výsledného data pro zaslání zpět klientovi
    result = forecast[['ds', 'yhat']].to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)