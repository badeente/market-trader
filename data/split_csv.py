import pandas as pd
import os

# Define the input and output paths; adjust as needed
input_file = 'data/clean15.csv'
output_dir = 'data/split_clean15'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file
df = pd.read_csv(input_file)

# Parse the timestamp column as datetime
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df.dropna(subset=['datetime'], inplace=True)
df.sort_values('datetime', inplace=True)

# Set the timestamp as the DataFrame index
df.set_index('datetime', inplace=True)

# Group by weekly frequency
grouped = df.groupby(pd.Grouper(freq='W'))

# Write each week's data to a separate CSV file
for week_start, chunk_df in grouped:
    if chunk_df.empty:
        continue
    week_str = week_start.strftime('%Y-%m-%d')
    output_file = os.path.join(output_dir, f'week_{week_str}.csv')
    chunk_df.to_csv(output_file, index=False)

print(f"CSV file split into weekly chunks and saved in {output_dir}")
