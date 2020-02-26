import requests
import json
import time # due to API restrictions

delay_condition = 0;
with open(r'C:\Users\coste\PycharmProjects\untitled2\pry.json') as json_file:
    delay_condition += 1
    data = json.load(json_file)
    for p in data:
        # print('To delay einai toso ' + int(delay_condition))
        delay_condition += 1
        if delay_condition == 5:
            print('Reached ')
            time.sleep(70)
            delay_condition = 0
        #print(p['Name'])
        parameters = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': p['Symbol'],
            'outputsize': 'compact',
            'apikey': 'RJANQO4BF951MNGA'
        }
        r = requests.get('https://www.alphavantage.co/query?', parameters)
        #print(r.json())
        filename = p['Symbol'] + '.json'
        with open(filename, "w", encoding="utf-8") as writeJSON:
            json.dump(r.json(), writeJSON, ensure_ascii=False)
