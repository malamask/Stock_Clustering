import os
import glob
from pylab import plot, show, mpl
import numpy as np
import pandas as pd
from math import sqrt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from fastcluster import linkage as linkageV
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet,ward

path = r'C:\Users\coste\PycharmProjects\Stock_Clustering\dataFiles'
stock_files = glob.glob(path + "/*.csv")
stocks = []

# trasform files
# The code bellow, load all the csv stock files and create the basic data frame
# With date rows and company-symbol columns
# The main_df data frame is saved as .csv.
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
#Remove the rows with N/A value
main_df = main_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
main_df.to_csv('sp500_closes.csv')



#Create one dictionary list and two parallel lists
# The dictionary list's key is symbol name and the value is a numpy array with daily close prices
series_list = {}  # use python dictionary
series_listH = []
symbol_listH = []
# seperate the
for symbol, value in main_df.iteritems():
    print()
    print(symbol)
    series_list[symbol] = np.array([], float)
    tempList = np.array([], float)
    for date, price in value.items():  # nan row and nan first values
        tempList = np.append(tempList, price)
        print(f"Index : {date}, Value : {price}")
        series_list[symbol] = np.append(series_list[symbol], price)
    print(series_list[symbol])
    series_listH.append(tempList)
    symbol_listH.append(symbol)


series_listH = np.asarray(series_listH, dtype=np.float32)

#hierarchical dendogram
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


# hierarchical with DTW

# dtw illustration for hierarchical
from dtaidistance import dtw

ds = dtw.distance_matrix(series_listH)

dsC = np.minimum(ds, ds.transpose())  ## create the summetrix distance matrix
np.fill_diagonal(dsC, 0)

import scipy.spatial.distance as ssd

# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(dsC)  # symmetric matrix
distList = dsC.tolist()


model = AgglomerativeClustering(distance_threshold=0, affinity='precomputed', n_clusters=None, linkage='complete')
model = model.fit(distList)

plt.title('Hierarchical Clustering Dendrogram with DTW')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#clusters for visualization
clustersDTW = model.children_
nsamplesDTW = len(model.labels_)


#calculate features
data = pd.read_csv(r"C:\Users\coste\PycharmProjects\Stock_Clustering\sp500_closes.csv", index_col="timestamp")

# Calculating annual mean returns and variances

returns = data.pct_change().mean() * 252
variance = data.pct_change().std() * sqrt(252)
returns.columns = ["Returns"]
variance.columns = ["Variance"]
# Concatenating the returns and variances into a single data-frame
ret_var = pd.concat([returns, variance], axis=1).dropna()
ret_var.columns = ["Returns", "Variance"]

# hierarcical with features

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
model = model.fit(ret_var)

plt.title('Hierarchical Clustering Dendrogram with Features')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#clusters of visualization
clustersFeatures = model.children_
nsamplesFeatures = len(model.labels_)

#Evaluation
# create n*n distance matrix with features
Z = squareform(pdist(ret_var)) #nxn distance matrix


#visualize distances General code
## sorting area

def seriation(Z,N,cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat,method="ward"): #default: ward
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkageV(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


#DTW visualization
dist_mat = dsC #distance matrix
corr = main_df.corr()
methods = ["ward","single","average","complete"]
for method in methods:
    print("Method:\t",method)
    N = len(series_listH)
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
    plt.pcolormesh(ordered_dist_mat,cmap=mpl.cm.jet) #use colormap
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    #CPCC calculation
    Z = linkage(corr, method)
    c, coph_dists = cophenet(Z, pdist(corr))
    plt.title("Distance Visualization - DTW  Method: " + method + " CPCC = " + str(c))
    plt.show()


#cophenetic for DTW
#https://silburt.github.io/blog/stock_correlation.html

corr = main_df.corr()
Z = linkage(corr,'ward')
c,coph_dists = cophenet(Z,pdist(corr))



#Features visualization
ret_var_array = ret_var.to_numpy()
dist_mat = pdist(ret_var)
methods = ["ward","single","average","complete"]
for method in methods:
    print("Method:\t",method)
    N = len(ret_var)
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(squareform(dist_mat), method)
    plt.pcolormesh(ordered_dist_mat,cmap=mpl.cm.jet) #use colormap
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    #cpcc calculation
    Z = linkage(pdist(ret_var_array), method)
    c, coph_dists = cophenet(Z, pdist(ret_var))
    #end
    plt.title("Distance Visualization - Features Method: " + method + " CPCC = " + str(c))
    plt.show()


#cophenetic for features

ret_var_array = ret_var.to_numpy()
#Z= single(pdist(ret_var_array))
#Z = ward(ret_var_array)
Z = linkage(pdist(ret_var_array),'ward')
c,coph_dists = cophenet(Z,pdist(ret_var))
print(c)

# Visualize stocks

def create_symbol_sets(clusters , number_of_samples ,next_pos,symbols):
    symbol_list = []
    new_pair = clusters[next_pos]
    #first_segment
    if new_pair[0] < number_of_samples: # exists in symbol_list
        symbol_list.append(symbols[new_pair[0]])
    else:
        next_pos = new_pair[0] - number_of_samples
        symbol_list.append(create_symbol_sets(clusters , number_of_samples ,next_pos,symbols))

    #second segment
    if new_pair[1] < number_of_samples: # exists in symbol_list
        symbol_list.append(symbols[new_pair[1]])
    else:
        next_pos = new_pair[1] - number_of_samples
        #print(next_pos)
        symbol_list.append(create_symbol_sets(clusters, number_of_samples, next_pos, symbols))

    return symbol_list

#fuction to iterate all the sub lists
def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o


# visualize clusters with features

for i, merge in enumerate(clustersFeatures):
    print(merge)
    plot_df = pd.DataFrame()
    symbol_plots = []
    pair=0
    for child_idx in merge:
        pair+=1
        if child_idx < nsamplesFeatures:
            #print(symbol_listH[child_idx])
            #plot for the pairs
            symbol_plots.append(symbol_listH[child_idx])
        else:
            next_position = child_idx - nsamplesFeatures
            symbol_plots.append(create_symbol_sets(clustersFeatures, nsamplesFeatures,next_position,symbol_listH))

    for s in list(traverse(symbol_plots)):
        if main_df.empty:
            plot_df = main_df[s]
        else:
            plot_df = plot_df.join(main_df[s], how='outer')
    plot_df.plot(figsize=(15, 8))
    plt.title("Stocks of the same cluster - Hierarchical with Features")
    plt.ylabel('Price')
    plt.show()



# visualize clusters with DTW

for i, merge in enumerate(clustersDTW):
    print(i)
    plot_df = pd.DataFrame()
    symbol_plots = []
    pair=0
    for child_idx in merge:
        pair+=1
        if child_idx < nsamplesDTW:
            #print(symbol_listH[child_idx])
            #plot for the pairs
            symbol_plots.append(symbol_listH[child_idx])
        else:
            next_position = child_idx - nsamplesDTW
            symbol_plots.append(create_symbol_sets(clustersDTW, nsamplesDTW,next_position,symbol_listH))

    for s in list(traverse(symbol_plots)):
        if main_df.empty:
            plot_df = main_df[s]
        else:
            plot_df = plot_df.join(main_df[s], how='outer')
    plot_df.plot(figsize=(15, 8))
    plt.title("Stocks of the same cluster - Hierarchical with DTW")
    plt.ylabel('Price')
    plt.show()















