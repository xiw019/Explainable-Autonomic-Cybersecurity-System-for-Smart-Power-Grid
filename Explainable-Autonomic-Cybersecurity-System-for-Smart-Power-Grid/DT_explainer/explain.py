from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'path_to_your_aggregated_csv_file.csv' is the path to your aggregated dataset
df = pd.read_csv('aggregated-syn-flood.csv')

# Encode categorical features
label_encoders = {}
for column in ['most_freq_highest_layer', 'most_freq_src_port', 'most_freq_dst_port', 'most_freq_protocol', 'most_freq_tcp_flags']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split dataset into features (X) and label (y)
X = df.drop('label', axis=1)
X = X.drop('timestamp', axis=1)  # Remove timestamp feature
y = df['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the decision tree visualization
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=label_encoder.classes_.tolist())
plt.savefig('decision_tree_visualization.png')

# Save the detailed inference process into a text file
with open('detailed_inference_process.txt', 'w') as f:
    def traverse_tree(node, depth=0):
        indent = "  " * depth
        if clf.tree_.feature[node] != -2:
            feature = X.columns[clf.tree_.feature[node]]
            threshold = clf.tree_.threshold[node]
            f.write(f"{indent}if {feature} <= {threshold}:\n")
            traverse_tree(clf.tree_.children_left[node], depth + 1)
            f.write(f"{indent}else:  # if {feature} > {threshold}\n")
            traverse_tree(clf.tree_.children_right[node], depth + 1)
        else:
            class_idx = clf.tree_.value[node][0].argmax()
            class_name = label_encoder.classes_[class_idx]
            f.write(f"{indent}class: {class_name}\n")

    f.write("Decision Tree Inference Process:\n\n")
    traverse_tree(0)
    f.write("\n")

    # Detailed explanation for selected samples
    sample_indices = [0, 10, 20]  # Change these indices as needed
    for index in sample_indices:
        f.write(f"Sample {index}:\n")
        sample_features = X_test.iloc[[index]]
        f.write(f"Features: {sample_features.to_dict(orient='records')[0]}\n")
        node_indicator = clf.decision_path(sample_features)
        leaf_id = clf.apply(sample_features)[0]
        node_index = node_indicator.indices[node_indicator.indptr[index]:node_indicator.indptr[index + 1]]
        f.write("Inference Steps:\n")
        for node_id in node_index:
            if node_id == leaf_id:
                f.write("  Leaf reached.\n")
                break
            f.write(f"  Node {node_id}:\n")
            feature = X.columns[clf.tree_.feature[node_id]]
            threshold = clf.tree_.threshold[node_id]
            if sample_features[feature].values[0] <= threshold:
                f.write(f"    Go to left child: {feature} <= {threshold}\n")
            else:
                f.write(f"    Go to right child: {feature} > {threshold}\n")
        f.write("\n")

# Close the file
f.close()
