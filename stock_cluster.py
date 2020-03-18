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

#Calculate average annual percentage return and volatilities over a theoretical one year period
returns = main_df.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = main_df.pct_change().std() * sqrt(252)

#format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
show()
# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
#show()

#drop the relevant stock from our data
returns.drop('AAL',inplace=True)

#recreate data to feed into the algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
#show()

#hierarchical clustering

import random
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
from dtaidistance import dtw
from plotly.subplots import make_subplots
from scipy import interpolate



NUM_OF_TRAJECTORIES = 200
MIN_LEN_OF_TRAJECTORY = 16
MAX_LEN_OF_TRAJECTORY = 40

THRESHOLD = 1.0

trajectoriesSet = {}
"""
for item in range(NUM_OF_TRAJECTORIES):
    length = random.choice(list(range(MIN_LEN_OF_TRAJECTORY, MAX_LEN_OF_TRAJECTORY + 1)))
    tempTrajectory = np.random.randint(low=-100, high=100, size=int(length / 4)).astype(float) / 100

    oldScale = np.arange(0, int(length / 4))
    interpolationFunction = interpolate.interp1d(oldScale, tempTrajectory)

    newScale = np.linspace(0, int(length / 4) - 1, length)
    tempTrajectory = interpolationFunction(newScale)

    trajectoriesSet[(str(item),)] = [tempTrajectory]
"""
trajectoriesSet = {}
import pandas as pd
data = pd.read_csv("sp500_closes.csv")
dfTemp = pd.DataFrame(data)
for key, value in dfTemp.iteritems():
    print(key,value)
    print()
    oldScale = trajectoriesSet[key]
    print(oldScale)
    trajectoriesSet[key] = interpolate.interp1d(oldScale,value)




print(trajectoriesSet)

trajectories = deepcopy(trajectoriesSet)
distanceMatrixDictionary = {}

iteration = 1
while True:
    distanceMatrix = np.empty((len(trajectories), len(trajectories),))
    distanceMatrix[:] = np.nan

    for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
        tempArray = []

        for index2, (filter2, trajectory2) in enumerate(trajectories.items()):

            if index1 > index2:
                continue

            elif index1 == index2:
                continue

            else:
                unionFilter = filter1 + filter2
                sorted(unionFilter)

                if unionFilter in distanceMatrixDictionary.keys():
                    distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)

                    continue

                metric = []
                for subItem1 in trajectory1:

                    for subItem2 in trajectory2:
                        metric.append(dtw.distance(subItem1, subItem2, psi=1))

                metric = max(metric)

                distanceMatrix[index1][index2] = metric
                distanceMatrixDictionary[unionFilter] = metric

    minValue = np.min(list(distanceMatrixDictionary.values()))

    if minValue > THRESHOLD:
        break

    minIndices = np.where(distanceMatrix == minValue)
    minIndices = list(zip(minIndices[0], minIndices[1]))

    minIndex = minIndices[0]

    filter1 = list(trajectories.keys())[minIndex[0]]
    filter2 = list(trajectories.keys())[minIndex[1]]

    trajectory1 = trajectories.get(filter1)
    trajectory2 = trajectories.get(filter2)

    unionFilter = filter1 + filter2
    sorted(unionFilter)

    trajectoryGroup = trajectory1 + trajectory2

    trajectories = {key: value for key, value in trajectories.items()
                    if all(value not in unionFilter for value in key)}

    distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                if all(value not in unionFilter for value in key)}

    trajectories[unionFilter] = trajectoryGroup

    print(iteration, 'finished!')
    iteration += 1

    if len(list(trajectories.keys())) == 1:
        break

for key, _ in trajectories.items():
    print(key)

for key, value in trajectories.items():

    if len(key) == 1:
        continue

    figure = make_subplots(rows=1, cols=1)
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(value))]

    for index, subValue in enumerate(value):
        figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index], line=dict(width=4), line_shape='spline'),
                         row=1, col=1,
                         )

        '''oldScale = np.arange(0, len(subValue))
        interpolateFunction = interpolate.interp1d(oldScale, subValue)

        newScale = np.linspace(0, len(subValue) - 1, MAX_LEN_OF_TRAJECTORY)
        subValue = interpolateFunction(newScale)

        figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index]), row=1, col=2)'''

    figure.update_layout(showlegend=False, height=600, width=900)
    figure.show()
