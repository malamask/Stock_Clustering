import os
import glob
import numpy as np
import pandas as pd
from dtaidistance import dtw
import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import pylab as pl
import numpy as np

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

# K-means with DTW
# https://tslearn.readthedocs.io/en/latest/auto_examples/plot_kmeans.html



seed = 0
numpy.random.seed(seed)
print(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print(X_train)
X_train = series_listH
print(X_train)
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
print(X_train.shape)
sz = X_train.shape[1]
print(sz)

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=8,
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


# Create features
# Average yearly return
# Yearly Variance



data = pd.read_csv(r"C:\Users\coste\PycharmProjects\Stock_Clustering\sp500_closes.csv", index_col="timestamp")


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
        print(current_cluster)
        current_cluster += 1
    sse.append(calculated_sse)

pl.plot(range(3,15), sse)
pl.title("Elbow Curve")
pl.xlabel("Number of clusters")
pl.ylabel("SSE")
pl.show()

#visualize stocks for Kmeans with features
symbol_array = np.array(symbol_listH)
label_array = kmeans.labels_
number_of_clusters = 8

for k in range(0,number_of_clusters):
    plot_df = pd.DataFrame()
    print(k)
    symbol_counter=0;
    for cluster in label_array:
        print(symbol_array[symbol_counter])
        if cluster == k:
            print(symbol_array[symbol_counter])
            if main_df.empty:
                plot_df = main_df[symbol_array[symbol_counter]]
            else:
                plot_df = plot_df.join(main_df[symbol_array[symbol_counter]], how='outer')
            #add the company in plot
        symbol_counter += 1
    plot_df.plot(figsize=(15,8))
    plt.ylabel('Price')
    plt.show()



#visualize stocks for Kmeans with DTW

symbol_array = np.array(symbol_listH)
label_array = sdtw_km.labels_
number_of_clusters = 8

for k in range(0,number_of_clusters):
    plot_df = pd.DataFrame()
    print(k)
    symbol_counter=0;
    for cluster in label_array:
        print(symbol_array[symbol_counter])
        if cluster == k:
            print(symbol_array[symbol_counter])
            if main_df.empty:
                plot_df = main_df[symbol_array[symbol_counter]]
            else:
                plot_df = plot_df.join(main_df[symbol_array[symbol_counter]], how='outer')
            #add the company in plot
        symbol_counter += 1
    plot_df.plot(figsize=(15,8))
    plt.ylabel('Price')
    plt.show()
