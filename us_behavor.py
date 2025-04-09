import pandas as pd
import numpy as np

# Load the cleaned clickstream data
clickstream_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/us/union_elastic_us_0301_0331_2025.parquet'
print(f"Loading file: {clickstream_file}")

try:
    clickstream_df = pd.read_parquet(clickstream_file)
except Exception as e:
    print(f"Error loading file: {e}")
    raise

print(f"Initial rows loaded: {len(clickstream_df)}")
print("Column names in input file:")
print(list(clickstream_df.columns))
print("Sample of initial data:")
print(clickstream_df.head())

# Check if required columns exist
expected_columns = ['internalUserId', 'startTime', 'userActions.extracted_data.text.topPriority', 
                   'userActions.extracted_data.text.specialneeds_option', 'userActions.extracted_data.text.drugs_option',
                   'userActions.targetUrl', 'duration', 'userActions.extracted_data.text.stateCode']
print("Checking for expected columns:")
for col in expected_columns:
    if col in clickstream_df.columns:
        print(f" - Found: {col}")
    else:
        print(f" - Missing: {col}")

# Filter only for non-null internalUserId
clickstream_df = clickstream_df[clickstream_df['internalUserId'].notna()]
print(f"Rows after filtering non-null internalUserId: {len(clickstream_df)}")
print("Sample after filtering:")
print(clickstream_df.head())

# Reset index to avoid alignment issues
clickstream_df = clickstream_df.reset_index(drop=True)

# Define output columns for behavioral features
output_columns = [
    'userid', 'start_time', 'city', 'state', 'zip', 'plan_id', 'compared_plan_ids',  # Identifiers
    'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
    'query_csnp', 'query_dsnp',  # Queries
    'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug', 'filter_provider', 'filter_vision',
    'filter_csnp', 'filter_dsnp',  # Filters
    'accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
    'accordion_vision', 'accordion_csnp', 'accordion_dsnp',  # Accordions (ignored)
    'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
    'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
    'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
    'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages', 'rel_time_dsnp_pages',  # Time (ignored)
    'total_session_time', 'num_pages_viewed', 'num_plans_selected', 'num_plans_compared',  # Session/Plan Interactions
    'submitted_application',  # New feature
    'persona'  # Label
]

# Initialize output DataFrame with the same index as filtered clickstream_df
output_df = pd.DataFrame(index=clickstream_df.index, columns=output_columns)

# Step 1: Deduplication - Relaxed to only internalUserId and userActions.targetUrl
clickstream_df = clickstream_df.drop_duplicates(subset=['internalUserId', 'userActions.targetUrl'], keep='first')
print(f"Rows after initial deduplication: {len(clickstream_df)}")
print("Sample after initial deduplication:")
print(clickstream_df.head())

# Populate output_df
output_df['userid'] = clickstream_df['internalUserId']
output_df['start_time'] = clickstream_df['startTime']

def extract_zip(url):
    if pd.isna(url):
        return None
    parts = url.split('/')
    for part in parts:
        if len(part) == 5 and part.isdigit():
            return part
    return None

def extract_plan_id(url):
    if pd.isna(url):
        return None
    if 'details.html' in url and 'https://www.uhc.com/medicare/health-plans/details.html' in url:
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part.startswith('H') and len(part) >= 10 and any(c.isdigit() for c in part):
                return part
    if 'plan-summary' in url and '#MEDSUPP' in url:
        return "MEDSUPP"
    return None

def extract_compared_plan_ids(url):
    if pd.isna(url):
        return None
    if 'plan-compare' not in url or '#' not in url:
        return None
    
    temp_buffer = url.split('#')[-1]
    if not temp_buffer:
        return None
    
    plan_ids = temp_buffer.split(',')
    unique_plan_ids = []
    seen = set()
    
    for plan_id in plan_ids:
        plan_id = plan_id.strip()
        if plan_id.startswith('H') and any(c.isdigit() for c in plan_id) and plan_id not in seen:
            unique_plan_ids.append(plan_id)
            seen.add(plan_id)
    
    return ','.join(unique_plan_ids[:3]) if unique_plan_ids else None

output_df['zip'] = clickstream_df['userActions.targetUrl'].apply(extract_zip)
output_df['plan_id'] = clickstream_df['userActions.targetUrl'].apply(extract_plan_id)
output_df['compared_plan_ids'] = clickstream_df['userActions.targetUrl'].apply(extract_compared_plan_ids)

# Debug: Check rows after extraction
print(f"Rows after feature extraction: {len(output_df)}")
print("Sample after extraction:")
print(output_df[['userid', 'start_time', 'compared_plan_ids']].head())

output_df['city'] = clickstream_df.get('city', pd.Series(index=clickstream_df.index))  # Use get() to handle missing column
output_df['state'] = clickstream_df.get('userActions.extracted_data.text.stateCode', pd.Series(index=clickstream_df.index))

# Deduplicate compared_plan_ids within each userid (only if needed, avoid dropping rows)
def deduplicate_compared_plan_ids(df):
    df = df.sort_values('start_time')
    # Only deduplicate if identical to previous row within same userid, but keep all rows
    df['compared_plan_ids'] = df.groupby('userid')['compared_plan_ids'].transform(
        lambda x: x.where(x != x.shift()).fillna(x)
    )
    return df

output_df = deduplicate_compared_plan_ids(output_df)

# Debug: Check rows after deduplication
print(f"Rows after compared_plan_ids deduplication: {len(output_df)}")
print("Sample after deduplication:")
print(output_df[['userid', 'start_time', 'compared_plan_ids']].head())

# Query mappings
query_mappings = {
    'query_dental': r'q1=dental',
    'query_transportation': r'q1=Transportation',
    'query_otc': r'q1=(otc|low income care)',
    'query_drug': r'q1=(drug coverage|prescription)',
    'query_provider': r'q1=(doctor[s]?|provider[s]?)',
    'query_vision': r'q1=vision',
    'query_csnp': r'q1=(csnp|Chronic|Diabete)',
    'query_dsnp': r'q1=(dsnp|medicaid)'
}

for col, pattern in query_mappings.items():
    output_df[col] = clickstream_df['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)

# Filter mappings
filter_mappings = {
    'filter_dental': r'additionalBenefits=dental',
    'filter_transportation': r'additionalBenefits=transportation',
    'filter_otc': r'additionalBenefits=otc',
    'filter_drug': r'additionalBenefits=dsnp',  # Note: This might be a typo; should it be drug-related?
    'filter_vision': r'additionalBenefits=vision',
    'filter_csnp': r'additionalBenefits=csnp',
    'filter_dsnp': r'additionalBenefits=dsnp'
}

for col, pattern in filter_mappings.items():
    output_df[col] = clickstream_df['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)

output_df['filter_provider'] = 0

# Accordions and time fields (ignored)
for col in ['accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug',
            'accordion_provider', 'accordion_vision', 'accordion_csnp', 'accordion_dsnp',
            'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
            'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
            'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages',
            'rel_time_drug_pages', 'rel_time_provider_pages', 'rel_time_vision_pages',
            'rel_time_csnp_pages', 'rel_time_dsnp_pages']:
    output_df[col] = 0.0

# Session and plan interaction metrics
#output_df['total_session_time'] = pd.to_numeric(clickstream_df['duration'], errors='coerce').fillna(0.0)
output_df['num_pages_viewed'] = 1

selection_pattern = 'https://www.uhc.com/medicare/health-plans/details.html'
comparison_pattern = 'plan-compare'

output_df['num_plans_selected'] = clickstream_df['userActions.targetUrl'].str.contains(selection_pattern, case=False, na=False).astype(int)
output_df['num_plans_compared'] = clickstream_df['userActions.targetUrl'].apply(
    lambda x: len(extract_compared_plan_ids(x).split(',')) if extract_compared_plan_ids(x) else 0
)

output_df['submitted_application'] = clickstream_df['userActions.targetUrl'].str.contains(r'online-application.html', case=False, na=False).astype(int)

# Revised Persona Label Logic
# Step 1: Define the topPriority to persona mapping
top_priority_mapping = {
    'doctorPref': 'ma_provider_network',
    'drugPref': 'ma_drug_coverage',
    'additional-vision': 'ma_vision',
    'additional-dental': 'ma_dental_benefit',
    'additional-hearing': 'ma_hearing',
    'additional-fitness': 'ma_fitness'
}

# Step 2: Define the persona mapping for final simplification
persona_mapping = {
    'ma_provider_network': 'doctor',
    'ma_drug_coverage': 'drug',
    'ma_vision': 'vision',
    'ma_dental_benefit': 'dental',
    'ma_hearing': 'hearing',  # Note: Hearing is not in the 8 personas; consider how to handle
    'ma_fitness': 'fitness'   # Note: Fitness is not in the 8 personas; consider how to handle
}

# Step 3: Function to determine initial persona based on specialneeds_option, drugs_option, and topPriority
def determine_persona(row):
    specialneeds_option = row.get('userActions.extracted_data.text.specialneeds_option', '')
    drugs_option = row.get('userActions.extracted_data.text.drugs_option', '')
    top_priority = row.get('userActions.extracted_data.text.topPriority', np.nan)
    
    # Initialize persona as an empty string to build upon
    persona = ''
    
    # Check CSNP condition (highest priority, append if topPriority is not csnp)
    if specialneeds_option and '["snp_chronic"]' in specialneeds_option:
        if pd.notna(top_priority) and top_priority != 'csnp':
            persona = f'{determine_base_persona(top_priority)},csnp'
        else:
            persona = 'csnp'
    
    # Check DSNP condition (next priority, append if topPriority is not dsnp)
    if specialneeds_option and '["snp_medicaid"]' in specialneeds_option:
        if pd.notna(top_priority) and top_priority != 'dsnp':
            if persona:
                persona += ',dsnp'
            else:
                persona = f'{determine_base_persona(top_priority)},dsnp'
        else:
            persona = 'dsnp' if not persona else f'{persona},dsnp'
    
    # Check drugs_option condition (next priority, concatenate instead of overwrite)
    if drugs_option and '["drug_yes"]' in drugs_option:
        if persona:
            persona += ',ma_drug_coverage'
        else:
            persona = 'ma_drug_coverage'
    
    # If no special conditions, use topPriority
    if not persona and pd.notna(top_priority):
        persona = determine_base_persona(top_priority)
    
    # If nothing is set, return 'unknown'
    return persona if persona else 'unknown'

# Helper function to determine base persona from topPriority
def determine_base_persona(top_priority):
    if top_priority in top_priority_mapping:
        return top_priority_mapping[top_priority]
    else:
        return top_priority

# Step 4: Function to map persona values based on substring matches
def map_persona(persona):
    if pd.isna(persona) or persona == 'unknown':
        return persona
    
    # Split the persona string into components (if comma-separated)
    persona_parts = persona.split(',')
    mapped_parts = []
    
    # Check each part against the persona_mapping
    for part in persona_parts:
        part = part.strip()
        mapped_value = part  # Default to original value if no mapping found
        for key, value in persona_mapping.items():
            if key in part:
                mapped_value = value
                break
        mapped_parts.append(mapped_value)
    
    # Join the mapped parts back together
    return ','.join(mapped_parts)

# Step 5: Apply the initial persona determination logic
output_df['persona'] = clickstream_df.apply(determine_persona, axis=1)

# Debug: Check persona distribution before mapping
print("Persona distribution before mapping:")
print(output_df['persona'].value_counts())

# Step 6: Apply the persona mapping
output_df['persona'] = output_df['persona'].apply(map_persona)

# Debug: Check persona distribution after mapping
print("Persona distribution after mapping:")
print(output_df['persona'].value_counts())

 # Define the date pattern
date_pattern = r'^[A-Za-z]{3} \d{1,2}, \d{4} @ \d{2}:\d{2}:\d{2}\.\d{3}$'

# Debug before date filter
print(f"Rows before date filter: {len(output_df)}")
output_df = output_df[output_df['start_time'].str.match(date_pattern, na=False)]
print(f"Rows after date filter: {len(output_df)}")

# Debug: Check state before final deduplication
print("Sample of state before final deduplication:")
print(output_df[['userid', 'state']].head(10))

# Deduplicate state in output_df just before saving
# Instead of aggregating only state, keep the first row per userid while preserving all columns
def deduplicate_state(df):
    # Sort by start_time to ensure consistent ordering (optional, based on your needs)
    df = df.sort_values('start_time')
    # For each user, take the first row but deduplicate state within that row if needed
    first_rows = df.groupby('userid').first().reset_index()
    # Deduplicate state values within the selected row (if state is a comma-separated string)
    first_rows['state'] = first_rows['state'].apply(
        lambda x: ', '.join(sorted(set(str(x).split(', ')))) if pd.notna(x) else np.nan
    )
    return first_rows

# Apply the deduplicate function
output_df = deduplicate_state(output_df)

# Debug: Check state deduplication just before saving
print("Sample of state after final deduplication:")
print(output_df[['userid', 'state']].head(10))

# Save the behavioral feature file
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/us_behavioral_0301_0331_2025.parquet'
print(f"Final rows in output_df: {len(output_df)}")
print("Sample of final output (all columns):")
# Ensure all columns are displayed in the sample output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Auto-detect width
print(output_df.head())

# Save the output with all columns
output_df.parquet(output_file, index=False)
print(f"Behavioral feature file saved to {output_file}")
