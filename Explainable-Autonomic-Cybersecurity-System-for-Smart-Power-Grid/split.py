import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from joblib import dump

# Load the aggregated dataset
df = pd.read_csv('aggregated-syn-flood.csv')

# Drop the 'timestamp_start' and 'duration' columns as they should not be used for training
df = df.drop(['timestamp_start', 'duration', 'length_mean'], axis=1)

# Encoding categorical features
label_encoder = LabelEncoder()
categorical_columns = ['most_freq_highest_layer', 'most_freq_src_port', 'most_freq_dst_port', 'most_freq_protocol', 'most_freq_tcp_flags']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Separate features (X) and target variable (y)
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier with parameters to control the tree size
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=40,  # Example parameter adjustment
    min_samples_leaf=30,  # Example parameter adjustment
)
dt_model.fit(X_train, y_train)

# Output the tree structure
tree_structure = export_text(dt_model, feature_names=list(X.columns))
print("\nDecision Tree Structure:\n")
print(tree_structure)

# Save the tree structure to a text file
with open('decision_tree_structure.txt', 'w') as f:
    f.write(tree_structure)

print("Decision Tree structure saved to 'decision_tree_structure.txt'.")

# Print example dataset entries and their traces along the tree
num_samples = 20
sample_entries = X_test.sample(num_samples, random_state=42)
for i, example_entry in enumerate(sample_entries.iterrows()):
    example_entry = example_entry[1]  # Extract the entry from the iterrows result
    predicted_class = dt_model.predict([example_entry])[0]
    node_indicator = dt_model.decision_path([example_entry])
    leaf_id = dt_model.apply([example_entry])[0]
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # Store the trace in a text file
    trace_filename = f'decision_tree_trace_{i+1}.txt'
    with open(trace_filename, 'w') as f:
        f.write("Example Dataset Entry:\n")
        f.write(f"All Feature Values: {example_entry}\n")
        f.write(f"Predicted Class: {predicted_class}\n\n")
        f.write("Trace along the tree:\n")
        for node_id in node_index:
            if leaf_id == node_id:
                continue
            if example_entry[dt_model.tree_.feature[node_id]] <= dt_model.tree_.threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            f.write(f"Node {node_id}: Feature '{X.columns[dt_model.tree_.feature[node_id]]}' {threshold_sign} {dt_model.tree_.threshold[node_id]}\n")
        f.write(f"\nPredicted Leaf Node: {leaf_id}")

    print(f"Decision Tree trace for sample {i+1} saved to '{trace_filename}'.")
