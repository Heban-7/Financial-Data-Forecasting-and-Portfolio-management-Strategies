import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, EfficientFrontier


class PortFolios():
    def __init__(self):
        pass
    
    def generate_random_portfolios(self, data):
        # Calculate expected returns and covariance matrix
        self.mu = expected_returns.mean_historical_return(data)  # Annualized returns
        self.cov_matrix = risk_models.sample_cov(data)  # Annualized covariance

        # Generate random portfolios
        num_portfolios = 10000
        self.assets = data.columns
        n_assets = len(self.assets)

        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(n_assets), num_portfolios)
        returns = np.dot(weights, self.mu)
        stddevs = np.sqrt(np.diag(weights @ self.cov_matrix @ weights.T))
        sharpe_ratios = returns / stddevs  # Assuming risk-free rate = 0

        # Find min volatility and max Sharpe portfolios
        min_vol_idx = np.argmin(stddevs)
        max_sharpe_idx = np.argmax(sharpe_ratios)

        # Random portfolios results
        random_portfolios = [
            {'weights': w, 'return': r, 'stddev': s, 'sharpe': sr}
            for w, r, s, sr in zip(weights, returns, stddevs, sharpe_ratios)
        ]

        # Use PyPortfolioOpt to get EXACT optimal portfolios

        # Minimum Volatility Portfolio
        ef_minvol = EfficientFrontier(self.mu, self.cov_matrix)
        ef_minvol.min_volatility()
        min_vol_weights = ef_minvol.clean_weights()
        min_vol_return, min_vol_stddev, _ = ef_minvol.portfolio_performance()

        # Maximum Sharpe Portfolio
        ef_maxsharpe = EfficientFrontier(self.mu, self.cov_matrix)
        ef_maxsharpe.max_sharpe()
        max_sharpe_weights = ef_maxsharpe.clean_weights()
        max_sharpe_return, max_sharpe_stddev, _ = ef_maxsharpe.portfolio_performance()

        # Compile results into a dictionary
        self.ef_results = {
            'random_portfolios': random_portfolios,
            'min_vol': {
                'weights': np.array([min_vol_weights[asset] for asset in assets]),
                'return': min_vol_return,
                'stddev': min_vol_stddev,
            },
            'max_sharpe': {
                'weights': np.array([max_sharpe_weights[asset] for asset in assets]),
                'return': max_sharpe_return,
                'stddev': max_sharpe_stddev,
            }
        }

    def plot_Efficient_Frontier(self):
        # Plot Efficient Frontier
        plt.figure(figsize=(15, 10))
        returns = [p['return'] for p in self.ef_results['random_portfolios']]
        stddevs = [p['stddev'] for p in self.ef_results['random_portfolios']]

        plt.scatter(stddevs, returns, c=np.array(returns)/np.array(stddevs),
                    marker='o', cmap='viridis', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')

        # Plot optimal portfolios
        plt.scatter(self.ef_results['min_vol']['stddev'], self.ef_results['min_vol']['return'],
                    marker='*', color='r', s=500, label='Minimum Volatility')
        plt.scatter(self.ef_results['max_sharpe']['stddev'], self.ef_results['max_sharpe']['return'],
                    marker='*', color='g', s=500, label='Maximum Sharpe Ratio')

        # Plot individual assets
        for i, asset in enumerate(self.assets):
            asset_vol = np.sqrt(self.cov_matrix.iloc[i, i])
            asset_ret = self.mu[i]
            plt.scatter(asset_vol, asset_ret, marker='o', s=200, color='black')
            plt.annotate(asset, (asset_vol*1.01, asset_ret*1.01))

        plt.title('Efficient Frontier')
        plt.xlabel('Expected Volatility (Standard Deviation)')
        plt.ylabel('Expected Annual Return')
        plt.legend()
        plt.tight_layout()

        # Print Portfolio Weights & Metrics
        print("\nMinimum Volatility Portfolio:")
        print("Expected Return: {:.2%}".format(self.ef_results['min_vol']['return']))
        print("Expected Volatility: {:.2%}".format(self.ef_results['min_vol']['stddev']))
        print("Sharpe Ratio: {:.2f}".format(self.ef_results['min_vol']['return']/ef_results['min_vol']['stddev']))
        print("Asset Allocation:")
        for i, asset in enumerate(self.assets):
            print(f"{asset}: {self.ef_results['min_vol']['weights'][i]:.2%}")

        print("\nMaximum Sharpe Ratio Portfolio:")
        print("Expected Return: {:.2%}".format(self.ef_results['max_sharpe']['return']))
        print("Expected Volatility: {:.2%}".format(self.ef_results['max_sharpe']['stddev']))
        print("Sharpe Ratio: {:.2f}".format(self.ef_results['max_sharpe']['return']/ef_results['max_sharpe']['stddev']))
        print("Asset Allocation:")
        for i, asset in enumerate(self.assets):
            print(f"{asset}: {self.ef_results['max_sharpe']['weights'][i]:.2%}")

        # Plot Asset Allocations
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.pie(self.ef_results['min_vol']['weights'], labels=self.assets, autopct='%1.1f%%')
        plt.title('Minimum Volatility Portfolio Allocation')

        plt.subplot(1, 2, 2)
        plt.pie(self.ef_results['max_sharpe']['weights'], labels=self.assets, autopct='%1.1f%%')
        plt.title('Maximum Sharpe Ratio Portfolio Allocation')

        plt.tight_layout()
        plt.show()


    def portfolio_optimization(self, data):
        # Portfolio Optimization

        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        # Optimize for max Sharpe Ratio
        ef_opt = EfficientFrontier(mu, S)
        weights = ef_opt.max_sharpe()
        cleaned_weights = ef_opt.clean_weights()

        # Print optimized weights
        print("Optimized Portfolio Weights:")
        for k, v in cleaned_weights.items():
            print(f"{k}: {100*v:.2f}%")

        # Create a SECOND ef instance for plotting (do not optimize this one)
        ef_plot = EfficientFrontier(mu, S)

        # Plot Efficient Frontier
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.efficient_frontier(ef_plot, ax=ax, show_assets=True)
        plt.title('Efficient Frontier')
        plt.tight_layout()

        # Plot asset allocation
        plt.figure(figsize=(15, 10))
        plt.pie(cleaned_weights.values(), labels=cleaned_weights.keys(), autopct='%1.1f%%')
        plt.title('Optimized Portfolio Weights')
        plt.tight_layout()