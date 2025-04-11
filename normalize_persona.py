import pandas as pd

# Define file paths
input_behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/us_dce_pro_behavioral_features_0301_0302_2025.csv'
output_normalized_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0301_0302_2025.csv'

# Step 1: Load the dataset
print(f"Loading behavioral features file: {input_behavioral_file}")
try:
    behavioral_df = pd.read_csv(input_behavioral_file)
except Exception as e:
    print(f"Error loading behavioral file: {e}")
    raise

print(f"Rows loaded: {len(behavioral_df)}")
print("Sample before normalization:")
print(behavioral_df[['userid', 'persona']].head())

# Step 2: Normalize persona column with duplication for dsnp/csnp
def process_persona(df):
    new_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row['persona']):
            new_rows.append(row)
            continue
        personas = [p.strip() for p in row['persona'].split(',')]
        # Case 1: Contains dsnp or csnp
        if 'dsnp' in personas or 'csnp' in personas:
            # First row: Keep first persona
            first_row = row.copy()
            first_row['persona'] = personas[0]
            new_rows.append(first_row)
            # Second row: dsnp or csnp (dsnp preferred)
            second_row = row.copy()
            second_row['persona'] = 'dsnp' if 'dsnp' in personas else 'csnp'
            new_rows.append(second_row)
        else:
            # Case 2: No dsnp/csnp, keep only first persona
            row_copy = row.copy()
            row_copy['persona'] = personas[0]
            new_rows.append(row_copy)
    return pd.DataFrame(new_rows)

# Step 3: Rename persona values
persona_mapping = {
    'additional-dental': 'dental',
    'doctorPref': 'doctor',
    'additional-fitness': 'fitness',
    'drugPref': 'drug',
    'additional-hearing': 'hearing',
    'additional-vision': 'vision'
}

print("Normalizing persona column...")
normalized_df = process_persona(behavioral_df)

print("Renaming persona values...")
normalized_df['persona'] = normalized_df['persona'].replace(persona_mapping)

# Step 4: Map 'unknown', 'none', and 'healthcare' to blank
print("Mapping 'unknown', 'none', and 'healthcare' to blank...")
normalized_df['persona'] = normalized_df['persona'].replace(['unknown', 'none', 'healthcare'], '')

print(f"Rows after normalization and renaming: {len(normalized_df)}")
print("Sample after normalization and mapping:")
print(normalized_df[['userid', 'persona']].head())

# Step 5: Save the normalized dataset
print(f"Saving normalized behavioral data to: {output_normalized_file}")
normalized_df.to_csv(output_normalized_file, index=False)
print("Normalization and mapping complete.")
