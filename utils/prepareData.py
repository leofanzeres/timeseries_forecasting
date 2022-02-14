import json
import pandas as pd

def getWeatherData():

    with open('data/barcelona-weather.json') as json_file:
        data = json.load(json_file)
        df = pd.DataFrame()
        timeStamp = [d['dt_iso'] for d in data if 'dt' in d]
        temperature = [d['main']['temp'] for d in data if 'dt' in d]
        pressure = [d['main']['pressure'] for d in data if 'dt' in d]
        humidity = [d['main']['humidity'] for d in data if 'dt' in d]

        if len(timeStamp) == len(temperature):
            df['time'] = timeStamp
            df['temperature'] = temperature
            df['pressure'] = pressure
            df['humidity'] = humidity

        return df
