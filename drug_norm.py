import pandas as pd

# Define file paths
drug_search_file = '/Workspace/Users/jwang77@optumcloud.com/data/s-learning-data/drug_search.csv'
output_drug_behavior_file = '/Workspace/Users/jwang77@optumcloud.com/data/s-learning-data/drug_behavior.csv'

# Load raw drug search data
print(f"Loading drug search file: {drug_search_file}")
drug_search_df = pd.read_csv(drug_search_file)

# Transform Drug Search Data
print("Transforming drug search data...")
# Convert 'time' to date
drug_search_df['date'] = pd.to_datetime(drug_search_df['time']).dt.date

# Count occurrences of each event type directly (no binary conversion)
drug_search_df['drug_search_click'] = (drug_search_df['eventName'] == 'drug_search_click').astype(int)
drug_search_df['drug_list_build_click'] = (drug_search_df['eventName'] == 'drug_list_build_click').astype(int)

# Group by date and rxVisitor, summing the counts
drug_behavior_df = drug_search_df.groupby(['date', 'rxVisitor'])[['drug_search_click', 'drug_list_build_click']].sum().reset_index()
drug_behavior_df = drug_behavior_df.rename(columns={'rxVisitor': 'userId'})  # Map rxVisitor to userId

# Ensure integer type for counts
drug_behavior_df['drug_search_click'] = drug_behavior_df['drug_search_click'].astype(int)
drug_behavior_df['drug_list_build_click'] = drug_behavior_df['drug_list_build_click'].astype(int)

# Save the normalized drug behavior data
print(f"Saving normalized drug behavior data to: {output_drug_behavior_file}")
drug_behavior_df.to_csv(output_drug_behavior_file, index=False)

print("Drug behavior data sample:")
print(drug_behavior_df.head())
print(f"Normalization complete. Rows processed: {len(drug_behavior_df)}")
