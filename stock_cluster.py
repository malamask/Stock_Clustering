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

# visualize AMZN, GOOG, GOOGL in the same graph

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['GOOG'], name="GOOG",
                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['GOOGL'], name="GOOGL",
                         line_color='dimgray'))
fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['AMZN'], name="AMZN",
                         line_color='#8c564b'))
fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['ADS'], name="ADS",
                         line_color='#bcbd22'))
fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['LNT'], name="LNT",
                         line_color='#e377c2'))
fig.add_trace(go.Scatter(x=main_df.index.values[1:], y=main_df['MO'], name="MO",
                         line_color='#7f7f7f'))

fig.update_layout(title_text='Time Series with Rangeslider',
                  xaxis_rangeslider_visible=True)
fig.show()

# end of visualization

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pylab as plt

# %pylab inline
# from pyFTS.partitioners import CMeans, Grid, FCM, Huarng, Entropy, Util as pUtil
# from pyFTS.common import Membership as mf
# from pyFTS.benchmarks import benchmarks as bchmk
# from pyFTS.data import Enrollments

# https://github.com/PYFTS/notebooks/blob/master/Partitioners.ipynb

from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

from pyFTS.data import TAIEX

dataset = TAIEX.get_data()
# dataset_diff = tdiff.apply(dataset)
for dataset in series_listH:
    # print(serie)
    dataset_diff = tdiff.apply(dataset)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[10, 5])
    ax[0].plot(dataset)
    ax[1].plot(dataset_diff)
    ax[0].set_xlabel('day number')
    ax[0].set_ylabel('close')
    ax[1].set_xlabel('day number')
    ax[1].set_ylabel('variation')
    plt.show()

data_size = len(series_list)
distance_matrix = np.zeros((data_size - 1, data_size - 1), float)
sum = 0

# hierarchical clustering
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# https://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

l_matrix = []
r"""

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
        labels=main_df.columns
    )
    # print("mas endiafrei")
    # print(linkage_matrix)


series_listH = np.asarray(series_listH, dtype=np.float32)
print(series_listH)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(series_listH)
# model = model.fit(X
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='lastp',
                p=47)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
"""

# K-means with DTW
# https://tslearn.readthedocs.io/en/latest/auto_examples/plot_kmeans.html

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

seed = 0
numpy.random.seed(seed)
print(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print(X_train)
# X_train = X_train[y_train < 4]  # Keep first 3 classes
X_train = series_listH
print(X_train)
# numpy.random.shuffle(X_train)
# Keep only 30 time series
# X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:30])
# Make time series shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
print(X_train.shape)
sz = X_train.shape[1]
print(sz)

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)
i = 0
for yi in range(3):
    plt.subplot(1, 3, 1 + yi)
    for xx in X_train[y_pred == yi]:
        i += 1
        print('seira:')
        print(i)
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(0, 2000)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()


# test area
import plotly.express as px

import pandas as pd

df = pd.read_csv(r'C:\Users\coste\Desktop\10ο Εξάμηνο\GOOG.csv')

fig = px.line(df, x='timestamp', y='close')
fig.show()

main_df.set_index(symbol_listH)

# second Feature k-means

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
# X_train = X_train[y_train < 4]  # Keep first 3 classes
X_train = series_listH
# numpy.random.shuffle(X_train)
# Keep only 50 time series
# X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:30])
# Make time series  shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]

# Create features
# Average yearly return
# Yearly Variance

import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import pylab as pl
import numpy as np

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

print(ret_var)

X = ret_var.values  # Converting ret_var into nummpy arraysse = []for k in range(2,15):
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    sse.append(kmeans.inertia_)  # SSE for each n_clusterspl.plot(range(2,15), sse)
pl.plot(range(2, 15), sse)
pl.title("Elbow Curve")
pl.xlabel("Number of clusters")
pl.ylabel("SSE")
pl.show()

kmeans = KMeans(n_clusters=8).fit(X)

centroids = kmeans.cluster_centers_
pl.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="rainbow")
pl.show()

# ret_var.drop("AWK", inplace=True)
print(returns.idxmax())
X = ret_var.values
kmeans = KMeans(n_clusters=8).fit(X)
centroids = kmeans.cluster_centers_
pl.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="rainbow")
pl.show()

Company = pd.DataFrame(ret_var.index)
cluster_labels = pd.DataFrame(kmeans.labels_)
df = pd.concat([Company, cluster_labels], axis=1)

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

# K-means with DTW
"""
X = distList
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    sse.append(kmeans.inertia_)  # SSE for each n_clusterspl.plot(range(2,15), sse)
pl.plot(range(2, 15), sse)
pl.title("Elbow Curve")
pl.show()

kmeans = KMeans(n_clusters=8).fit(X)

centroids = kmeans.cluster_centers_
pl.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="rainbow")
pl.show()

"""


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

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
model = model.fit(ret_var)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# cophenetic fucntion

# hierarchical cophenetic
"""
X = distList
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    sse.append(kmeans.inertia_)  # SSE for each n_clusterspl.plot(range(2,15), sse)
pl.plot(range(2, 15), sse)
pl.title("Elbow Curve")
pl.show()

kmeans = KMeans(n_clusters=8).fit(X)

centroids = kmeans.cluster_centers_
pl.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="rainbow")
pl.show()

"""

# calculate SSE for time series, giving cluster centers and time series in numpy type
sse = []
for k in range(3, 15):
    sdtw_km = TimeSeriesKMeans(n_clusters=k,
                               metric="softdtw",
                               metric_params={"gamma": .01},
                               verbose=True,
                               random_state=seed)
    y_pred = sdtw_km.fit_predict(X_train)
    centers_array = sdtw_km.cluster_centers_
    clusters = sdtw_km.labels_
    time_series = X_train
    calculated_sse = 0
    nuumber_of_clusters = 2  # input clusters - 1
    current_cluster = 0
    for s1 in centers_array:
        serie_number = 0
        for cluster in clusters:
            print(cluster)
            if cluster == current_cluster:
                s2 = time_series[serie_number]
                series_DTW = dtw.distance(s1, s2)
                calculated_sse = calculated_sse + np.math.pow(series_DTW, 2)
            serie_number += 1
        print('mpike re malaka')
        print(current_cluster)
        print("mexri edw")
        current_cluster += 1
    sse.append(calculated_sse)

pl.plot(range(3,15), sse)
pl.title("Elbow Curve")
pl.xlabel("Number of clusters")
pl.ylabel("SSE")
pl.show()



print(calculated_sse)
print(cluster)

#cophenetic - features

from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist, squareform

Z = single(pdist(ret_var))
cophenet(Z)
print(squareform(cophenet(Z)))


