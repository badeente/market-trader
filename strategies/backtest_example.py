import os
from datetime import datetime
from strategies.backtesting_strategy import run_backtest
from strategies.two_legged_pullback_strategy import TwoLeggedPullbackStrategy
from backtesting import Backtest
import pandas as pd

def save_backtest_results(stats, bt, strategy_name: str):
    # Create results directory if it doesn't exist
    results_dir = "strategies/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{strategy_name}_{timestamp}")
    os.makedirs(run_dir)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(run_dir, "statistics.csv"))
    
    # Save trades to CSV
    trades_df = pd.DataFrame(stats._trades)
    trades_df.to_csv(os.path.join(run_dir, "trades.csv"))
    
    # Save summary to text file
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("Backtest Results Summary\n")
        f.write("=======================\n\n")
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Performance Metrics:\n")
        f.write("-----------------\n")
        f.write(f"Return: {stats['Return [%]']:.2f}%\n")
        f.write(f"Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%\n")
        f.write(f"Max. Drawdown: {stats['Max. Drawdown [%]']:.2f}%\n")
        f.write(f"# Trades: {stats['# Trades']}\n")
        f.write(f"Win Rate: {stats['Win Rate [%]']:.2f}%\n")
        f.write(f"Profit Factor: {stats['Profit Factor']:.2f}\n")
        f.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n\n")
        
        f.write("Files:\n")
        f.write("------\n")
        f.write("- statistics.csv: Detailed performance metrics\n")
        f.write("- trades.csv: Individual trade details\n")
    
    return run_dir

def main():
    # Initialize the strategy
    strategy = TwoLeggedPullbackStrategy()
    
    # Run backtest
    stats, bt = run_backtest(
        csv_path="data/clean15.csv",
        strategy=strategy,
        cash=10000,  # Start with $10,000
        commission=0.002  # 0.2% commission per trade
    )
    
    # Save results
    results_dir = save_backtest_results(stats, bt, "TwoLeggedPullback")
    
    # Print path to results
    print(f"\nBacktest results saved to: {results_dir}")
    
    # Print key metrics to console
    print("\nKey Metrics:")
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Max. Drawdown: {stats['Max. Drawdown [%]']:.2f}%")

if __name__ == "__main__":
    main()
