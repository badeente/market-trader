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
    print("Starting backtest process...")
    
    try:
        # Initialize the strategy
        print("Initializing TwoLeggedPullbackStrategy...")
        strategy = TwoLeggedPullbackStrategy()
        
        # Get all split CSV files - fixed pattern to use single underscore
        print("Looking for CSV files in data/split_clean15/...")
        split_csv_files = sorted(glob.glob('data/split_clean15/week_*.csv'))
        print(f"Found {len(split_csv_files)} CSV files")
        
        if not split_csv_files:
            print("ERROR: No CSV files found matching the pattern 'data/split_clean15/week_*.csv'")
            print("Current working directory:", os.getcwd())
            try:
                print("Files in data/split_clean15/:", os.listdir('data/split_clean15'))
            except Exception as e:
                print(f"Error listing directory: {str(e)}")
            return
        
        # Print first few files found to verify pattern
        print("\nFirst few files found:")
        for file in split_csv_files[:5]:
            print(f"- {file}")
        
        all_stats = []
        all_bt = []
        
        for chunk_number, csv_path in enumerate(split_csv_files, 1):
            print(f"\nProcessing chunk {chunk_number}/{len(split_csv_files)}: {csv_path}")
            try:
                # Verify file exists and is readable
                if not os.path.isfile(csv_path):
                    print(f"ERROR: File does not exist: {csv_path}")
                    continue
                    
                # Check file size
                file_size = os.path.getsize(csv_path)
                if file_size == 0:
                    print(f"ERROR: File is empty: {csv_path}")
                    continue
                    
                print(f"File size: {file_size} bytes")
                
                # Run backtest for each chunk
                print("Running backtest...")
                stats, bt = run_backtest(
                    csv_path=csv_path,
                    strategy=strategy,
                    cash=10000,  # Start with $10,000
                    commission=0.002  # 0.2% commission per trade
                )
                print("Backtest completed successfully")
                
                # Save results for each chunk
                print("Saving results...")
                results_dir = save_backtest_results(stats, bt, "TwoLeggedPullback", chunk_number)
                print(f"Results saved to: {results_dir}")
                
                # Collect stats and bt for aggregation
                all_stats.append(stats)
                all_bt.append(bt)
                
            except Exception as e:
                print(f"ERROR processing chunk {chunk_number}:")
                print(f"Exception: {str(e)}")
                import traceback
                print("Traceback:")
                print(traceback.format_exc())
                continue
        
        print(f"\nProcessed {len(all_stats)} chunks successfully out of {len(split_csv_files)} total chunks")
        
        # Aggregate results
        if all_stats:
            print("\nAggregating results...")
            try:
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
                print("Saving aggregated results...")
                aggregated_results_dir = save_backtest_results(aggregated_stats, None, "TwoLeggedPullback_Aggregated", "all")
                print(f"Aggregated results saved to: {aggregated_results_dir}")
                
                # Print aggregated results
                print("\nAggregated Key Metrics:")
                print(f"Return: {aggregated_stats['Return [%]']:.2f}%")
                print(f"# Trades: {aggregated_stats['# Trades']}")
                print(f"Win Rate: {aggregated_stats['Win Rate [%]']:.2f}%")
                print(f"Max. Drawdown: {aggregated_stats['Max. Drawdown [%]']:.2f}%")
            except Exception as e:
                print("ERROR during aggregation:")
                print(f"Exception: {str(e)}")
                import traceback
                print("Traceback:")
                print(traceback.format_exc())
        else:
            print("\nNo results to aggregate - all chunks failed")
    
    except Exception as e:
        print("ERROR in main:")
        print(f"Exception: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
