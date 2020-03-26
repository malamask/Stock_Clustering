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


#trasform files

main_df = pd.DataFrame()
for filename in stock_files:
    df = pd.read_csv(filename,nrows=70)#number of daily prices

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
series_listH = []
#seperate the
for symbol, value in main_df.iteritems():
    #print(key, value)
    #print(key)
    print()
    print(symbol)
    series_list[symbol] = np.array([], float)
    tempList = np.array([],float)
    for date, price in value[2:].items(): #nan row and nan first values
        #series_list[]
        tempList = np.append(tempList,price)
        print(f"Index : {date}, Value : {price}")
        series_list[symbol] = np.append(series_list[symbol],price)
    print(series_list[symbol])
    series_listH.append(tempList)

#visualization
print(series_listH)
series_listH = np.asarray(series_listH,dtype=np.float32)
print(series_listH)


import warnings
warnings.filterwarnings('ignore')


import matplotlib.pylab as plt

#%pylab inline
#from pyFTS.partitioners import CMeans, Grid, FCM, Huarng, Entropy, Util as pUtil
#from pyFTS.common import Membership as mf
#from pyFTS.benchmarks import benchmarks as bchmk
#from pyFTS.data import Enrollments



from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

from pyFTS.data import TAIEX

dataset = TAIEX.get_data()
#dataset_diff = tdiff.apply(dataset)
for dataset in series_listH:
    #print(serie)
    dataset_diff = tdiff.apply(dataset)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[10,5])
    ax[0].plot(dataset)
    ax[1].plot(dataset_diff)
    plt.show()








#start - DTW only
import itertools

for series1, series2 in itertools.combinations(series_list, 2):
    #print(series1," ", series_list[series1])
    #print("kai")
    #print(series2," ", series_list[series2])

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    x = series_list[series1]
    print(x)
    print(series_list.__sizeof__())
    y = series_list[series2]

    distance, path = fastdtw(x, y, dist=euclidean)

    print(distance)
    print(path)
"""
for series in series_list:
    print(series ,"  ",series_list[series] )
    print("epomeno")
#calculate DTW

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    x = np.array([1, 2, 3, 3, 7])
    print(x)
    print(series_list.__sizeof__())
    y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

    distance, path = fastdtw(x, y, dist=euclidean)

    print(distance)
    print(path)
"""

#hierarchical clustering
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


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
    dendrogram(linkage_matrix, **kwargs)




series_listH = np.asarray(series_listH,dtype=np.float32)
print(series_listH)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(series_listH)
#model = model.fit(X
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=6)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#K-means

"""
from tslearn.generators import random_walks
X = random_walks(n_ts=50, sz=32, d=1)
print(X)
print(series_listH)
km = TimeSeriesKMeans(n_clusters=10, metric="euclidean", max_iter=30,random_state=0).fit(series_listH)
#km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,random_state=0).fit(X)

km.cluster_centers_.shape
#print(series_listH)


"""


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
#X_train = X_train[y_train < 4]  # Keep first 3 classes
X_train = series_listH
print(X_train)
#numpy.random.shuffle(X_train)
# Keep only 30 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:30])
# Make time series shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
print(X_train.shape)
sz = X_train.shape[1]
print(sz)
# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=0)
y_pred = km.fit_predict(X_train)
plt.figure()
for yi in range(3):
    plt.subplot(1, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(0, 200)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")


plt.tight_layout()
plt.show()



#test area
"""
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = series_listH
print(X_train.size)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:10])
X_train = X_train[y_train < 4]  # Keep first 3 classes
X_train = series_listH
print(X_train)
numpy.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:20])

X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
print(X_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:3])
print(X_train)
print(X_train.shape)
X_train = series_listH


X_train = TimeSeriesResampler(sz=10).fit_transform(X_train)
print(X_train.shape)

"""
