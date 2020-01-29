import pandas as pd


def cleanData(data):
    stock_series = {}
    for c in data.columns:
        if 'Unnamed: ' in c:
            date_col = data[c]
        else:
            temp = pd.to_numeric(data[c])
            temp.index = date_col
            temp = temp.loc[~temp.index.duplicated(keep='first')]
            
            
            stock_series[c] = temp.copy()
    
    df = pd.DataFrame.from_dict(stock_series, 
                 orient='columns')
    df.dropna(how='all', inplace=True)
    
    return df
