import numpy as np
import pandas as pd
dataf = pd.read_csv("assignment1.csv", comment = '#')

X = np.array(dataf.iloc[:, 0]); X = X.reshape(-1, 1)
Y = np.array(dataf.iloc[:, 1]); Y = Y.reshape(-1, 1)

column_max = dataf.max()
dataf_max = column_max.max()

column_min = dataf.min()
dataf_min = column_min.min()

normalized_dataf = (dataf - dataf_min) / (dataf_max - dataf_min)
print("Normalized Data: \n ", normalized_dataf)