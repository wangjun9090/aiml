import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define file path
input_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/032025/training_dataset_0301_0302_2025_1.csv'

# Step 1: Load the optimized dataset
print(f"Loading optimized dataset: {input_file}")
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

print(f"Rows loaded: {len(df)}")
print("Columns:", list(df.columns))

# Step 2: Check for target_persona
if 'target_persona' not in df.columns:
    print(f"ERROR: 'target_persona' not found in dataset. Found columns: {list(df.columns)}")
    raise ValueError("'target_persona' is required for validation")

# Step 3: Filter out classes with insufficient samples
min_samples = 2  # Minimum samples per class for stratification
class_counts = df['target_persona'].value_counts()
rare_classes = class_counts[class_counts < min_samples].index.tolist()

if rare_classes:
    print(f"\nWarning: The following classes have fewer than {min_samples} samples and will be excluded:")
    for cls in rare_classes:
        print(f" - {cls}: {class_counts[cls]} sample(s)")
    df = df[~df['target_persona'].isin(rare_classes)].reset_index(drop=True)
    print(f"Rows after filtering rare classes: {len(df)}")

# Check if any data remains
if len(df) == 0:
    print("ERROR: No data remains after filtering classes with insufficient samples.")
    raise ValueError("Cannot proceed with model training due to insufficient data.")

# Step 4: Prepare features and target
feature_columns = [col for col in df.columns if col not in ['userid', 'zip', 'plan_id', 'state', 'target_persona']]
X = df[feature_columns]
y = df['target_persona']

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Step 5: Split data for training and validation
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print(f"Error during train-test split: {e}")
    print("Falling back to non-stratified split due to class imbalance.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_test)}")

# Step 6: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Step 7: Predict on validation set
y_pred = rf_model.predict(X_test)

# Step 8: Calculate match and accuracy
df_test = pd.DataFrame({'target_persona': y_test, 'predicted_persona': y_pred})
df_test['match'] = df_test['target_persona'] == df_test['predicted_persona']

# Step 9: Overall accuracy
overall_accuracy = df_test['match'].mean()
print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
print(f"Total validation rows: {len(df_test)}")
print(f"Correct predictions: {df_test['match'].sum()}")

# Step 10: Accuracy by persona
print("\nIndividual Persona Accuracy Rates:")
persona_accuracy = df_test.groupby('target_persona')['match'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Count'})
persona_accuracy['Accuracy'] = persona_accuracy['Accuracy'].map('{:.2%}'.format)

# List all expected personas for completeness
expected_personas = ['doctor', 'drug', 'dental', 'otc', 'vision', 'csnp', 'dsnp', 'fitness', 'hearing']
for persona in expected_personas:
    if persona in persona_accuracy.index:
        accuracy = persona_accuracy.loc[persona, 'Accuracy']
        count = persona_accuracy.loc[persona, 'Count']
        print(f"Accuracy for '{persona}': {accuracy} (Count: {count})")
    else:
        print(f"Accuracy for '{persona}': N/A (Count: 0 - not present in target_persona)")

# Full table for reference
print("\nFull Accuracy Table by Target Persona:")
print(persona_accuracy.sort_values(by='Accuracy', ascending=False))

# Step 11: Mismatch analysis (merge with original features for context)
df_test_full = df.loc[y_test.index].copy()  # Get original rows corresponding to test set
df_test_full['predicted_persona'] = df_test['predicted_persona']
df_test_full['match'] = df_test['match']
feature_columns = [col for col in df.columns if col not in ['userid', 'zip', 'plan_id', 'state', 'target_persona']]
print("\nMismatch Analysis (Top 10 mismatches):")
mismatches = df_test_full[~df_test_full['match']][['userid', 'target_persona', 'predicted_persona'] + feature_columns[:5]].head(10)
print(mismatches)

# Step 12: Save validation results
output_validation_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/032025/eval_0301_0302_2025.csv'
df_test_full[['userid', 'target_persona', 'predicted_persona', 'match']].to_csv(output_validation_file, index=False)
print(f"\nValidation results saved to: {output_validation_file}")
