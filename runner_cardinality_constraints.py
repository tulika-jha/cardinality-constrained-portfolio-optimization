import os
import math
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time

from metaheuristic_algorithms.pso_cardinality_constraints import PSOPortfolioOptimizationCardinalityConstraints

stocks = [stock.split('.')[0] for stock in sorted(os.listdir('data/australian_stocks'))][0:75]
print(stocks)

np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(threshold=np.inf)

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

# risk aversion parameter (lambda_val)
lambda_val = 0.5
n = len(stocks)


# Optimize using metaheuristic algorithm
# 1. First define an objective function
# 2. Then initialize metaheuristic problem
# 3. Optimize
# 4. Plot result


np.random.seed(2)

def objective_func(x, mu, sigma, lambda_val):
    return lambda_val * x.T @ sigma @ x - (1 - lambda_val) * mu.T @ x

start_time = time.time()

pso = PSOPortfolioOptimizationCardinalityConstraints(
    objective_func=objective_func,
    objective_func_arguments={
        "mu": mu,
        "sigma": sigma,
        "lambda_val": lambda_val,
    },
    portfolio_size=n,
    cardinality_min=2,
    cardinality_max=10,
    num_particles=100,
    num_iterations=100,
    c1=2,
    c2=1,
    w=0.8,
    verbose=False,
)

pso.optimize()

end_time = time.time()

x_opt = pso.gbest
objective_opt = pso.gbest_obj

print(f"Optimal portfolio weights = {x_opt}")
print(f"Total risk of the portfolio = {x_opt.T @ sigma @ x_opt}")
print(f"Total return of the portfolio = {mu.T @ x_opt}")
print(f"Optimal objective value = {objective_opt}")
print(f"Total time taken = {end_time - start_time} seconds")

# remove values very close to 0 from the optimal portfolio weights
threshold = 1e-5
mask = x_opt > threshold
x_opt_filtered = x_opt[mask]
stocks_opt_filtered = np.array(stocks)[mask]

# convergence plot
plt.plot(range(1, pso.num_iterations+1), pso.training_history)
plt.xlabel('iteration')
plt.ylabel('gbest')
plt.show()

plt.bar(stocks_opt_filtered, height=x_opt_filtered)

for i, height in enumerate(x_opt_filtered):
    plt.text(i, height, f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Stocks')
plt.ylabel('Weights')
plt.show()

