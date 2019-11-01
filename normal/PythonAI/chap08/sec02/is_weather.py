import requests
url = 'http://weather.livedoor.com/forecast/webservice/json/v1'
payload = { 'city': '130010' }
weather_data = requests.get(url, params=payload).json()
for weather in weather_data['forecasts']:
    print(
        weather['dateLabel']
        + 'の天気は'
        + weather['telop']
    )
