import os
import csv
from math import sqrt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq
import glob
import datetime as datetime
from matplotlib import pyplot as plt
import pandas_datareader.data as web
from matplotlib import ticker
from sklearn.cluster import KMeans

#start of extra imports
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


#end
r"""
path = r'C:\Users\coste\PycharmProjects\Stock_Clustering\Data_Script'
stock_files = glob.glob(os.path.join(path,"*.csv"))
df = pd.concat((pd.read_csv(f) for f in stock_files))

print(df.size)

for a in df:
    print(df)
"""


path = r'C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles'

stock_files = glob.glob(path + "/*.csv")

stocks = []
"""
for filename in stock_files:
    df = pd.read_csv(filename, index_col = None, header = 0)
    stocks.append(df)
    df.head()

frame = pd.concat(stocks, axis = 0, ignore_index = True, sort=False)

# K-Means clustering
print(df.head())
"""


#trasform files

main_df = pd.DataFrame()
for filename in stock_files:
    df = pd.read_csv(filename,nrows=10)

    df.set_index('timestamp', inplace=True)

    df.rename(columns = {'close': os.path.basename(filename[:-4])} , inplace=True)

    df.drop(['open', 'high', 'low','volume'], 1, inplace = True)

    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df,how='outer')

print(main_df.head())
main_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
main_df.to_csv('sp500_closes.csv')

series_list = {} #use python dictionary

#seperate the
for symbol, value in main_df.iteritems():
    #print(key, value)
    #print(key)
    print()
    print(symbol)
    series_list[symbol] = np.array([], float)
    for date, price in value[1:].items():
        #series_list[]
        print(f"Index : {date}, Value : {price}")
        series_list[symbol] = np.append(series_list[symbol],price)
    print(series_list[symbol])

#start

#calculate DTW

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    x = np.array([1, 2, 3, 3, 7])
    print(x)
    print(series_list['A'])
    y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

    distance, path = fastdtw(x, y, dist=euclidean)

    #print(distance)
   # print(path)