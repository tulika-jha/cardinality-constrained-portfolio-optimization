import os
import math
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from metaheuristic_algorithms.pso_cardinality_constraints import PSOPortfolioOptimizationCardinalityConstraints

np.random.seed(0)
np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(threshold=np.inf)


def load_and_process_data():
    """ Load prices, process raw data and return mu, sigma and stock names. """
    stocks = [stock.split('.')[0] for stock in sorted(os.listdir('data/australian_stocks'))][0:100]
    print(stocks)
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
    
    # Calculate weekly returns from 1 January 2019 onwards
    r = data[(data['Time'].dt.weekday == 4) & (data['Time'] >= '2019-01-01')] \
        .drop(['Time'], axis=1) \
        .pct_change(fill_method='ffill')
    
    sigma = r.cov().to_numpy()
    mu = r.mean().to_numpy()
    return mu, sigma, stocks


def solve_with_pso(lambda_val=0.5):
    """ Solve the problem (with hardcoded parameters) using pso and return the optimal and historical prices. """
    
    # load and process data
    mu, sigma, stocks = load_and_process_data()
    n = len(stocks)
    
    # Optimize using metaheuristic algorithm
    # 1. First define an objective function
    # 2. Then initialize metaheuristic problem
    # 3. Optimize
    
    def objective_func(x, mu, sigma, lambda_val):
        return lambda_val * x.T @ sigma @ x - (1 - lambda_val) * mu.T @ x
    
    pso = PSOPortfolioOptimizationCardinalityConstraints(
        objective_func=objective_func,
        objective_func_arguments={
            "mu": mu,
            "sigma": sigma,
            "lambda_val": lambda_val,
        },
        portfolio_size=n,
        cardinality_min=5,
        cardinality_max=100,
        num_particles=100,
        num_iterations=100,
        c1=2,
        c2=1,
        w=0.8,
        verbose=False,
    )
    
    pso.optimize()
    
    x_opt = pso.gbest
    objective_opt = pso.gbest_obj
    training_history = pso.training_history
    risk = x_opt.T @ sigma @ x_opt
    returns = mu.T @ x_opt
    
    return x_opt, objective_opt, training_history, risk, returns


def plot_pareto_curve(solver_function, lambda_vals, risk_filter=np.inf, returns_filter=np.inf, show_optimal_boundary=True):
    """ Plot the Pareto curve for the optimization problem. """
    
    risk_vals = []
    returns_vals = []
    
    for lambda_val in lambda_vals:
        x_opt, objective_opt, training_history, risk, returns = solver_function(lambda_val)
        if risk < risk_filter and returns < returns_filter:
            risk_vals.append(risk)
            returns_vals.append(returns)
    
    fig = plt.figure()
    plt.scatter(risk_vals, returns_vals, color='mediumpurple', s=50, label="Solution using PSO")
    plt.xlabel("Standard deviation (Risk)")
    plt.ylabel("Potential Return")
    plt.title("Risk/Return Tradeoff")
    
    if show_optimal_boundary:
        returns_and_risk = sorted(zip(returns_vals, risk_vals), key=lambda x: x[1])
        returns_vals, risk_vals = zip(*returns_and_risk)
        
        optimal_risk_vals = [risk_vals[0]]
        optimal_returns_vals = [returns_vals[0]]
        
        for i in range(1, len(risk_vals)):
            if returns_vals[i] > optimal_returns_vals[-1]:
                optimal_risk_vals.append(risk_vals[i])
                optimal_returns_vals.append(returns_vals[i])
        print(f"Optimal risk vals = {optimal_risk_vals}, Optmal return vals = {optimal_returns_vals}")
        plt.plot(optimal_risk_vals, optimal_returns_vals, color='lightgreen', label='Optimal risk/return tradeoff')
        plt.legend()
    
    plt.show()


# plot pareto curve for pso
solver_function = solve_with_pso
lambda_vals = np.logspace(-5, 0, 50)
plot_pareto_curve(
    solver_function=solver_function,
    lambda_vals=lambda_vals,
    risk_filter=40,
    returns_filter=1.5,
)
lambda_vals = np.linspace(0, 1, 50)
plot_pareto_curve(
    solver_function=solver_function,
    lambda_vals=lambda_vals,
    risk_filter=4,
    returns_filter=0.7,
)
lambda_vals = (1 - np.logspace(-5, 0, 50))[::-1]
plot_pareto_curve(
    solver_function=solver_function,
    lambda_vals=lambda_vals,
    risk_filter=0.2,
    returns_filter=0.2,
)
print(f"lambda_vals = {lambda_vals}")
