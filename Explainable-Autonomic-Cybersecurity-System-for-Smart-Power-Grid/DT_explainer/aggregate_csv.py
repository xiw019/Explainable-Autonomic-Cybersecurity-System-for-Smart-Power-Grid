import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to aggregate the dataset and apply specific normalization to 'duration'
def aggregate_data(df, step=20):
    aggregated_data = []
    
    for start_row in range(0, len(df), step):
        end_row = start_row + step
        if end_row > len(df):
            subset = df.iloc[start_row:len(df)]
        else:
            subset = df.iloc[start_row:end_row]
        
        # Continuous columns
        length_mean = subset['length'].mean()
        length_range = subset['length'].max() - subset['length'].min()
        length_median = subset['length'].median()
        ttl_mean = subset['ttl'].mean()
        ttl_range = subset['ttl'].max() - subset['ttl'].min()
        ttl_median = subset['ttl'].median()
        payload_length_mean = subset['payload_length'].mean()
        payload_length_range = subset['payload_length'].max() - subset['payload_length'].min()
        payload_length_median = subset['payload_length'].median()
        
        # Categorical columns using pandas.Series.mode
        most_freq_highest_layer = subset['highest_layer'].mode().iloc[0] if not subset['highest_layer'].mode().empty else 'Unknown'
        most_freq_src_port = subset['src_port'].mode().iloc[0] if not subset['src_port'].mode().empty else 'Unknown'
        most_freq_dst_port = subset['dst_port'].mode().iloc[0] if not subset['dst_port'].mode().empty else 'Unknown'
        most_freq_protocol = subset['protocol'].mode().iloc[0] if not subset['protocol'].mode().empty else 'Unknown'
        most_freq_tcp_flags = subset['tcp_flags'].mode().iloc[0] if not subset['tcp_flags'].mode().empty else 'Unknown'
        
        # Calculate duration as the difference between the last and first timestamp in the subset, multiply by 10000
        duration = (subset.iloc[-1]['timestamp'] - subset.iloc[0]['timestamp']) * 100000
        
        aggregated_row = {
            'timestamp_start': subset.iloc[0]['timestamp'],
            'length_mean': length_mean,
            'length_range': length_range,
            'length_median': length_median,
            'ttl_mean': ttl_mean,
            'ttl_range': ttl_range,
            'ttl_median': ttl_median,
            'payload_length_mean': payload_length_mean,
            'payload_length_range': payload_length_range,
            'payload_length_median': payload_length_median,
            'most_freq_highest_layer': most_freq_highest_layer,
            'most_freq_src_port': most_freq_src_port,
            'most_freq_dst_port': most_freq_dst_port,
            'most_freq_protocol': most_freq_protocol,
            'most_freq_tcp_flags': most_freq_tcp_flags,
            'label': subset['label'].mode().iloc[0] if not subset['label'].mode().empty else 'Unknown',
            'duration': duration
        }
        
        aggregated_data.append(aggregated_row)
    
    aggregated_df = pd.DataFrame(aggregated_data)
    
    # Apply log transformation to 'duration' to shrink differences (adding 1 to avoid log(0))
    aggregated_df['duration'] = np.log1p(aggregated_df['duration'])
    
    # Normalize the transformed 'duration' using Min-Max scaling
    scaler = MinMaxScaler()
    aggregated_df['duration'] = scaler.fit_transform(aggregated_df[['duration']])
    
    # Fill NaN values with 0 for numerical columns and 'Unknown' for categorical columns
    numerical_columns = ['length_mean', 'length_range', 'length_median', 'ttl_mean', 'ttl_range', 'ttl_median', 'payload_length_mean', 'payload_length_range', 'payload_length_median', 'duration']
    categorical_columns = ['most_freq_highest_layer', 'most_freq_src_port', 'most_freq_dst_port', 'most_freq_protocol', 'most_freq_tcp_flags', 'label']
    
    aggregated_df[numerical_columns] = aggregated_df[numerical_columns].fillna(0)
    aggregated_df[categorical_columns] = aggregated_df[categorical_columns].fillna('Unknown')
    
    return aggregated_df

# Example usage
# Load the original dataset
df = pd.read_csv('syn-flood.csv')

# Aggregate the data, normalize 'duration', and fill NaN values appropriately
aggregated_df = aggregate_data(df)

# Save the aggregated dataset to a new CSV file
aggregated_df.to_csv('aggregated-syn-flood.csv', index=False)
