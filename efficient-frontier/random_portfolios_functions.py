import numpy as np
import matplotlib.pyplot as plt

# get random weights
def random_weights(n):
    k = np.random.rand(n)
    return k /sum(k)

# work out mean and vol for random portfolio
def random_portfolio(returns):
    
    p = np.asmatrix(np.mean(returns, axis=0))
    w = np.asmatrix(random_weights(returns.shape[1]))
    C = np.asmatrix(np.cov(returns, rowvar=False))
    
    mu = (1+(w*p.T))**12-1
    sigma = np.sqrt(w*C*w.T)*np.sqrt(12)
    
    return mu, sigma

def get_random_portfolios(rets, n_portfolios=10000):
    means, stds = np.column_stack([
            random_portfolio(rets) for _ in range(n_portfolios)])
    
    return means, stds

def plot_random_portfolios(means, stds):
    plt.scatter(stds, means)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.axvline(x=0, linestyle='--', color='black')
    plt.title("Random portfolios: Mean vs volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Mean returns")
    plt.show()