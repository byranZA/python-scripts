import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# import data
data = pd.read_excel('data - asset class.xlsx', sheet_name='Asset Returns')
data.index = data['Date']
del data['Date']

data_names = pd.read_excel('data - asset class.xlsx', sheet_name='Names')
data_names.index = data_names['Index']
del data_names['Index']


def portfolio_volatility(w, rets, cov_mat):
    w = w.reshape(-1,1)
    vol = np.sqrt(w.T.dot(cov_mat).dot(w))
    ann_vol = vol*np.sqrt(12)
    return ann_vol

def portfolio_return(w, rets):
    cret = (1+np.mean(rets.dot(w.T)))**12-1
    return cret
    

# work out returns
rets = data.pct_change().dropna().values

# generate random portfolios
import random_portfolios_functions as rfunc
means, stds = rfunc.get_random_portfolios(rets, 500000)
rfunc.plot_random_portfolios(means, stds)
    
target_return = 0.06

# minimise volatility
num_assets = rets.shape[1]
cov_mat = np.cov(rets, rowvar=False)
bounds = tuple((0,1) for asset in range(num_assets))
constraints = ({'type':'eq', 'fun': lambda x: portfolio_return(x, rets) - target_return},
                {'type':'eq', 'fun': lambda x: np.sum(x) - 1})
result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=(rets, cov_mat),
                  method='SLSQP', bounds=bounds, constraints=constraints)

portfolio_volatility(result.x, rets, cov_mat)
portfolio_return(result.x, rets)


opt_rets = []
opt_vol = []
for target_return in np.linspace(0, 0.15,50):
    try:
        num_assets = rets.shape[1]
        cov_mat = np.cov(rets, rowvar=False)
        bounds = tuple((0,1) for asset in range(num_assets))
        constraints = ({'type':'eq', 'fun': lambda x: portfolio_return(x, rets) - target_return},
                        {'type':'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=(rets, cov_mat),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_vol.append(portfolio_volatility(result.x, rets, cov_mat)[0][0])
        opt_rets.append(portfolio_return(result.x, rets))
    except:
        pass


plt.plot(opt_vol, opt_rets)
plt.scatter(stds, means)
