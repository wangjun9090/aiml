import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the training dataset
input_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/training_dataset_8.csv'
print(f"Loading training dataset: {input_file}")
try:
    training_df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading training file: {e}")
    raise

print(f"Rows loaded: {len(training_df)}")
print("Columns in training dataset:", list(training_df.columns))

# Step 2: Prepare features and target
# Assuming all columns except 'userid', 'zip', 'plan_id', 'dsnp_type', 'csnp_type', 'state', 'target_persona' are features
feature_columns = [col for col in training_df.columns if col not in ['userid', 'zip', 'plan_id', 'dsnp_type', 'csnp_type', 'state', 'target_persona']]
X = training_df[feature_columns]
y = training_df['target_persona']

# Handle any remaining missing values
X = X.fillna(0)

# Step 3: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Step 4: Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Limit depth to prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42,       # For reproducibility
    class_weight='balanced'  # Adjust for class imbalance
)

rf_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Step 6: Save the model as a pickle file
pickle_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/models/rf_model_csnp_focus.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Random Forest model saved to {pickle_file}")

# Optional: Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))
