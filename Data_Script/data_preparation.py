import os
import glob
import numpy as np
import pandas as pd
from dtaidistance import dtw

path = r'C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles'

stock_files = glob.glob(path + "/*.csv")

stocks = []


# trasform files

main_df = pd.DataFrame()
for filename in stock_files:
    df = pd.read_csv(filename, nrows=200)  # number of daily prices
    print(filename)
    if 'timestamp' in df:

        df.set_index('timestamp', inplace=True)

        df.rename(columns={'close': os.path.basename(filename[:-4])}, inplace=True)

        df.drop(['open', 'high', 'low', 'volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

print(main_df.head())
#main_df = main_df.fillna(main_df.mean())
main_df = main_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
main_df.to_csv('sp500_closes.csv')

series_list = {}  # use python dictionary
series_listH = []
symbol_listH = []
# seperate the
for symbol, value in main_df.iteritems():
    # print(key, value)
    # print(key)
    print()
    print(symbol)
    series_list[symbol] = np.array([], float)
    tempList = np.array([], float)
    for date, price in value.items():  # nan row and nan first values
        # series_list[]
        tempList = np.append(tempList, price)
        print(f"Index : {date}, Value : {price}")
        series_list[symbol] = np.append(series_list[symbol], price)
    print(series_list[symbol])
    series_listH.append(tempList)
    symbol_listH.append(symbol)

# visualization
series_listH = np.asarray(series_listH, dtype=np.float32)

