import os
import csv
import pandas as pd
import glob
import datetime as datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from matplotlib import ticker
from sklearn.cluster import KMeans

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
returns = prices_df.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)

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

