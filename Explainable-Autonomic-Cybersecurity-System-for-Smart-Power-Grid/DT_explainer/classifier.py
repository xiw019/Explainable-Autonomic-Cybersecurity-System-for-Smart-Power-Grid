import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Load the aggregated dataset
df = pd.read_csv('aggregated-syn-flood.csv')

# Drop the 'timestamp_start' and 'duration' columns as they should not be used for training
df = df.drop(['timestamp_start','duration','length_mean'], axis=1)

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

# Predict on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save the trained Decision Tree model to a file
model_filename = 'decision_tree_model.joblib'
dump(dt_model, model_filename)

print(f"Model saved to {model_filename}")

# Output the tree structure
tree_structure = export_text(dt_model, feature_names=list(X.columns))
print("\nDecision Tree Structure:\n")
print(tree_structure)

# Save the tree structure to a text file
with open('decision_tree_structure.txt', 'w') as f:
    f.write(tree_structure)

print("Decision Tree structure saved to 'decision_tree_structure.txt'.")
