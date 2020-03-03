import os
import csv
import pandas as pd
import glob

path = r"C:\Users\coste\PycharmProjects\Stock_Clustering\Data_Script\*.csv"
i=0;
for fname in glob.glob(path):
   i=i+1
   print(i)
   df = pd.read_csv(fname)
   print(df)
   #df.head()
   my_list = list(df.columns)
   for col in my_list:
       print(col)
   print(len(my_list) , my_list)



