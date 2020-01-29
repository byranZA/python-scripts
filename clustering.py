import pandas as pd
import functions 

data = pd.read_excel('BBG-data.xlsx',
                     sheet_name='Stocks-Top40')

df = functions.cleanData(data)

import matplotlib.pyplot as plt
plt.plot(df['BVT SJ Equity'])
