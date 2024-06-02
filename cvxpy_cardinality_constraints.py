import os
import math
import numpy as np
import cvxpy as cp
import pandas as pd
import time

import matplotlib.pyplot as plt

stocks = [stock.split('.')[0] for stock in sorted(os.listdir('data/australian_stocks'))][0:100]
print(stocks)

np.set_printoptions(precision=6, suppress=True)

dates = pd.date_range('2000-01-01', '2020-03-31')
data = pd.DataFrame({'Time': dates})

for stock in stocks:
    prices = pd.read_csv(f'data/australian_stocks/{stock}.csv', usecols=['Date', 'Adj Close'])
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.rename(columns={'Date': 'Time', 'Adj Close': stock},
                  inplace=True)
    data = pd.merge(data, prices, how='left', on=['Time'], sort=False)

data = data[data['Time'].dt.weekday < 5]  # Remove weekend dates
data = data.dropna(axis=0, how='all')  # Remove empty rows

# last price for each stock is p (prices)
p = data.drop(['Time'], axis=1).tail(1).to_numpy()
# print(p)

# Calculate weekly returns from 1 January 2019 onwards
r = data[(data['Time'].dt.weekday == 4) & (data['Time'] >= '2019-01-01')] \
    .drop(['Time'], axis=1) \
    .pct_change(fill_method='ffill')

sigma = r.cov().to_numpy()
mu = r.mean().to_numpy()

n = len(stocks)


start_time = time.time()

# Solve using convex optimization procedure
x = cp.Variable(n, integer=True)
lambda_val = 0.5
objective = cp.Minimize(lambda_val * cp.quad_form(x, sigma) - (1 - lambda_val) * mu.T @ x)
constraints = [
    x >= 0,
    # cp.sum(x) == 1,
    cp.sum(x) >= 10,
    cp.sum(x) <= 25,
    
    # cp.quad_form(x, sigma) <= 0.05,
]
problem = cp.Problem(objective, constraints)
problem.solve(solver='ECOS_BB')

end_time = time.time()

x_opt = x.value

print(f"Optimal portfolio weights = {x_opt}")
print(f"Total risk of the portfolio = {x_opt.T @ sigma @ x_opt}")
print(f"Total return of the portfolio = {mu.T @ x_opt}")
print(f"Optimal objective value = {objective.value}")
print(f"Total time taken = {end_time - start_time} seconds")

# remove values very close to 0 from the optimal portfolio weights
threshold = 1e-5
mask = x_opt > threshold
x_opt_filtered = x_opt[mask]
stocks_opt_filtered = np.array(stocks)[mask]

plt.bar(stocks_opt_filtered, height=x_opt_filtered)

for i, height in enumerate(x_opt_filtered):
    plt.text(i, height, f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Stocks')
plt.ylabel('Weights')
plt.show()




