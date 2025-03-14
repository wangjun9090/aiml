import pandas as pd

# Define file paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/data/s-learning-data/behavioral_features_0901_2024_0228_2025.csv'
drug_behavior_file = '/Workspace/Users/jwang77@optumcloud.com/data/s-learning-data/drug_behavior.csv'
output_behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/data/s-learning-data/updated_behavioral_features_0901_2024_0228_2025.csv'

# Step 1: Load the datasets
print(f"Loading original behavioral features file: {behavioral_file}")
behavioral_df = pd.read_csv(behavioral_file)

print(f"Loading normalized drug behavior file: {drug_behavior_file}")
drug_behavior_df = pd.read_csv(drug_behavior_file)

# Step 2: Join Drug Behavior with Behavioral Data
print("Joining drug behavior data with behavioral data...")
# Aggregate drug behavior by userId, summing clicks across all dates
drug_behavior_agg = drug_behavior_df.groupby('userId')[['drug_search_click', 'drug_list_build_click']].sum().reset_index()

# Rename columns to match requested field names
drug_behavior_agg = drug_behavior_agg.rename(columns={
    'drug_search_click': 'drug_search',
    'drug_list_build_click': 'drug_build'
})

# Merge with behavioral_df
behavioral_df = behavioral_df.merge(
    drug_behavior_agg[['userId', 'drug_search', 'drug_build']],
    on='userId',
    how='left'
)

# Fill NaN with 0 for new columns (no drug activity = 0)
behavioral_df['drug_search'] = behavioral_df['drug_search'].fillna(0).astype(int)
behavioral_df['drug_build'] = behavioral_df['drug_build'].fillna(0).astype(int)

# Step 3: Save the updated behavioral dataset
print(f"Saving updated behavioral data to: {output_behavioral_file}")
behavioral_df.to_csv(output_behavioral_file, index=False)

print("Updated behavioral data sample with drug features:")
print(behavioral_df[['userId', 'drug_search', 'drug_build']].head())
print(f"Behavioral data generation complete. Rows processed: {len(behavioral_df)}")
