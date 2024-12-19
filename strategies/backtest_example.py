import os
from datetime import datetime
from strategies.backtesting_strategy import run_backtest
from strategies.two_legged_pullback_strategy import TwoLeggedPullbackStrategy
from backtesting import Backtest
import pandas as pd
import glob

def save_backtest_results(stats, bt, strategy_name: str, chunk_number: int):
    # Create results directory if it doesn't exist
    results_dir = "strategies/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{strategy_name}_{timestamp}_chunk_{chunk_number}")
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
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Chunk: {chunk_number}\n\n")
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
    
    # Get all split CSV files
    split_csv_files = sorted(glob.glob('data/split_dev/dev_chunk_*.csv'))
    
    all_stats = []
    all_bt = []
    
    for chunk_number, csv_path in enumerate(split_csv_files, 1):
        # Run backtest for each chunk
        stats, bt = run_backtest(
            csv_path=csv_path,
            strategy=strategy,
            cash=10000,  # Start with $10,000
            commission=0.002  # 0.2% commission per trade
        )
        
        # Save results for each chunk
        results_dir = save_backtest_results(stats, bt, "TwoLeggedPullback", chunk_number)
        
        # Print path to results for each chunk
        print(f"\nBacktest results for chunk {chunk_number} saved to: {results_dir}")
        
        # Collect stats and bt for aggregation
        all_stats.append(stats)
        all_bt.append(bt)
    
    # Aggregate results
    if all_stats:
        aggregated_stats = {
            'Return [%]': sum(stat['Return [%]'] for stat in all_stats),
            'Buy & Hold Return [%]': sum(stat['Buy & Hold Return [%]'] for stat in all_stats),
            'Max. Drawdown [%]': max(stat['Max. Drawdown [%]'] for stat in all_stats),
            '# Trades': sum(stat['# Trades'] for stat in all_stats),
            'Win Rate [%]': sum(stat['Win Rate [%]'] for stat in all_stats) / len(all_stats),
            'Profit Factor': sum(stat['Profit Factor'] for stat in all_stats) / len(all_stats),
            'Sharpe Ratio': sum(stat['Sharpe Ratio'] for stat in all_stats) / len(all_stats)
        }
        
        # Save aggregated results
        aggregated_results_dir = save_backtest_results(aggregated_stats, None, "TwoLeggedPullback_Aggregated", "all")
        
        # Print aggregated results
        print("\nAggregated Key Metrics:")
        print(f"Return: {aggregated_stats['Return [%]']:.2f}%")
        print(f"# Trades: {aggregated_stats['# Trades']}")
        print(f"Win Rate: {aggregated_stats['Win Rate [%]']:.2f}%")
        print(f"Max. Drawdown: {aggregated_stats['Max. Drawdown [%]']:.2f}%")

if __name__ == "__main__":
    main()
