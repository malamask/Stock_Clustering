import os
import csv
from math import sqrt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
import glob
import datetime as datetime
from matplotlib import pyplot as plt
import pandas_datareader.data as web
from matplotlib import ticker
from sklearn.cluster import KMeans

# start of extra imports
from pylab import plot, show
from numpy import vstack, array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans, vq
import pandas as pd
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import pylab as pl
import numpy as np


# end
from tslearn.clustering import TimeSeriesKMeans

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

# trasform files

main_df = pd.DataFrame()
for filename in stock_files:
    df = pd.read_csv(filename, nrows=70)  # number of daily prices
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
print(series_listH)
series_listH = np.asarray(series_listH, dtype=np.float32)
print(series_listH)


# dtw illustration for hierarchical
from dtaidistance import dtw

# print(type(symbol_listH))

ds = dtw.distance_matrix(series_listH)

print(type(ds))

dsC = np.minimum(ds, ds.transpose())  ## create the summetrix distance matrix
np.fill_diagonal(dsC, 0)

import scipy.spatial.distance as ssd

# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(dsC)  # symmetric matrix
print(distArray)
# data_matrix = [[0,0.8,0.9],[0.8,0,0.2],[0.9,0.2,0]]
distList = dsC.tolist()

# hierarchical with DTW

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix,
        **kwargs,
        labels=main_df.columns  # company symbols
    )


model = AgglomerativeClustering(distance_threshold=0, affinity='precomputed', n_clusters=None, linkage='complete')
model = model.fit(distList)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# hierarcical with features


# start = "3/16/2020"
data = pd.read_csv(r"C:\Users\coste\PycharmProjects\Stock_Clustering\sp500_closes.csv", index_col="timestamp")
# print(data)
# data = data.loc[start:]

# Calculating annual mean returns and variances

returns = data.pct_change().mean() * 252
variance = data.pct_change().std() * sqrt(252)
returns.columns = ["Returns"]
variance.columns = ["Variance"]
# Concatenating the returns and variances into a single data-frame
ret_var = pd.concat([returns, variance], axis=1).dropna()
ret_var.columns = ["Returns", "Variance"]


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
model = model.fit(ret_var)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#cophenetic - features

from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist, squareform

Z = single(pdist(ret_var))
cophenet(Z)
print(squareform(cophenet(Z)))
