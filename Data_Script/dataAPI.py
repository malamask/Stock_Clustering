import os
import requests
import json
import time # due to API restrictions
import csv
from os.path import join as pjoin

# API call from Alpha Vintage
#Tt returns the first 50 companies of S&P500 for hole the operation period.
#Symbols are loaded from a .json file.
#All the stocks are saved in a folder as .csv files
delay_condition = 0;
companies = 0;
with open(r'C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles\S&P500.json') as json_file:
    delay_condition += 1
    data = json.load(json_file)
    for p in data:
        if companies == 50:
            break
        companies += 1
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
            'outputsize': 'full', #compact for less data
            'apikey': 'RJANQO4BF951MNGA',
            'datatype': 'csv'

        }
        r = requests.get('https://www.alphavantage.co/query?', parameters)
        #print(r.json())###
        filename = p['Symbol'] + '.csv'
        #with open(filename, "w", encoding="utf-8") as writeJSON: C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles
            #json.dump(r.json(), writeJSON, ensure_ascii=False)
        path = r'C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles'

        with open(os.path.join(path,filename), 'w',newline="\n" ) as csvfile:
            writer = csv.writer(csvfile)
            for line in r.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))