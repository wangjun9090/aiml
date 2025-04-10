import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# File paths
clickstream_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/us/elastic_us_0301_2025.csv'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/us_behavioral_0301_2025.csv'
print(f"Loading file: {clickstream_file}")

# Use pyarrow to read Parquet metadata
parquet_file = pq.ParquetFile(clickstream_file)
total_rows = parquet_file.metadata.num_rows
print(f"Initial rows loaded: {total_rows}")

# Define output columns
output_columns = [
    'userid', 'start_time', 'city', 'state', 'zip', 'plan_id', 'compared_plan_ids',
    'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
    'query_csnp', 'query_dsnp',
    'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug', 'filter_provider', 'filter_vision',
    'filter_csnp', 'filter_dsnp',
    'accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
    'accordion_vision', 'accordion_csnp', 'accordion_dsnp',
    'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
    'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
    'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
    'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages', 'rel_time_dsnp_pages',
    'total_session_time', 'num_pages_viewed', 'num_plans_selected', 'num_plans_compared',
    'submitted_application', 'persona'
]

# Check if file is empty
if total_rows == 0:
    print("Error: The input Parquet file is empty.")
    output_df = pd.DataFrame(columns=output_columns)
    output_df.to_csv(output_file, index=False)
    print(f"Saved empty output file to {output_file}")
    raise ValueError("Processing stopped due to empty input Parquet file.")

# Filter records where topPriority is not null and not empty, limit to 10,000
top_priority_col = 'userActions.extracted_data.text.topPriority'
filtered_data = pd.DataFrame()
for i in range(parquet_file.num_row_groups):
    row_group = parquet_file.read_row_group(i).to_pandas()
    if top_priority_col in row_group.columns:
        valid_rows = row_group[
            row_group[top_priority_col].notna() & 
            (row_group[top_priority_col] != '') & 
            (row_group[top_priority_col] != 'None')
        ]
        filtered_data = pd.concat([filtered_data, valid_rows])
        if len(filtered_data) >= 10000:
            filtered_data = filtered_data.head(10000).reset_index(drop=True)
            break
    else:
        print(f"Column {top_priority_col} not found in row group {i}")
        break

if filtered_data.empty:
    print("No records found with non-null and non-empty userActions.extracted_data.text.topPriority")
else:
    print(f"Processing data with {len(filtered_data)} rows where {top_priority_col} is non-null and non-empty")
    
    # Rename internalUserId to userid at the beginning
    if 'internalUserId' in filtered_data.columns:
        filtered_data = filtered_data.rename(columns={'internalUserId': 'userid'})
    
    print(f"Initial rows: {len(filtered_data)}")
    if not filtered_data.empty:
        print("Sample of input data:")
        print(filtered_data.head())
        print("All columns in data:")
        print(filtered_data.columns.tolist())
    
    # Filter only for non-null userid
    filtered_data = filtered_data[filtered_data['userid'].notna()].reset_index(drop=True)
    print(f"Rows after userid filter: {len(filtered_data)}")
    
    # Initial deduplication
    filtered_data = filtered_data.drop_duplicates(subset=['userid', 'userActions.targetUrl'], keep='first').reset_index(drop=True)
    print(f"Rows after initial deduplication: {len(filtered_data)}")
    
    # Initialize output_df
    output_df = pd.DataFrame(index=filtered_data.index, columns=output_columns)
    
    # Populate basic fields
    output_df['userid'] = filtered_data['userid']
    if 'startTime' in filtered_data.columns:
        output_df['start_time'] = pd.to_datetime(filtered_data['startTime'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        output_df['start_time'] = pd.Series(index=filtered_data.index, dtype='object')
    print(f"Rows after basic fields: {len(output_df)}")
    
    # URL extraction functions
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
        if pd.isna(url) or 'plan-compare' not in url or '#' not in url:
            return None
        temp_buffer = url.split('#')[-1]
        if not temp_buffer:
            return None
        plan_ids = temp_buffer.split(',')
        unique_plan_ids = list(dict.fromkeys([pid.strip() for pid in plan_ids if pid.strip().startswith('H') and any(c.isdigit() for c in pid)]))
        return ','.join(unique_plan_ids[:3]) if unique_plan_ids else None
    
    output_df['zip'] = filtered_data['userActions.targetUrl'].apply(extract_zip)
    output_df['plan_id'] = filtered_data['userActions.targetUrl'].apply(extract_plan_id)
    output_df['compared_plan_ids'] = filtered_data['userActions.targetUrl'].apply(extract_compared_plan_ids)
    output_df['city'] = filtered_data.get('city', pd.Series(index=filtered_data.index, dtype='object'))
    output_df['state'] = filtered_data.get('userActions.extracted_data.text.stateCode', pd.Series(index=filtered_data.index, dtype='object'))
    print(f"Rows after URL extraction: {len(output_df)}")
    
    # Deduplicate compared_plan_ids
    def deduplicate_compared_plan_ids(df):
        if 'start_time' in df.columns and not df['start_time'].isna().all():
            df = df.sort_values('start_time')
        df['compared_plan_ids'] = df.groupby('userid')['compared_plan_ids'].transform(
            lambda x: x.where(x != x.shift()).fillna(x)
        )
        return df
    
    output_df = deduplicate_compared_plan_ids(output_df)
    print(f"Rows after deduplicate_compared_plan_ids: {len(output_df)}")
    
    # Query and filter mappings
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
    
    filter_mappings = {
        'filter_dental': r'additionalBenefits=dental',
        'filter_transportation': r'additionalBenefits=transportation',
        'filter_otc': r'additionalBenefits=otc',
        'filter_drug': r'additionalBenefits=dsnp',
        'filter_vision': r'additionalBenefits=vision',
        'filter_csnp': r'additionalBenefits=csnp',
        'filter_dsnp': r'additionalBenefits=dsnp'
    }
    
    for col, pattern in query_mappings.items():
        output_df[col] = filtered_data['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
    for col, pattern in filter_mappings.items():
        output_df[col] = filtered_data['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
    output_df['filter_provider'] = 0
    print(f"Rows after query/filter mappings: {len(output_df)}")
    
    # Ignored fields
    for col in ['accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug',
                'accordion_provider', 'accordion_vision', 'accordion_csnp', 'accordion_dsnp',
                'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
                'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
                'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
                'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages', 'rel_time_dsnp_pages']:
        output_df[col] = 0.0
    
    # Session metrics
    if 'duration' in filtered_data.columns:
        output_df['total_session_time'] = pd.to_numeric(filtered_data['duration'], errors='coerce').fillna(0.0)
    else:
        output_df['total_session_time'] = 0.0
    
    output_df['num_pages_viewed'] = 1
    output_df['num_plans_selected'] = filtered_data['userActions.targetUrl'].str.contains(
        'https://www.uhc.com/medicare/health-plans/details.html', case=False, na=False).astype(int)
    output_df['num_plans_compared'] = filtered_data['userActions.targetUrl'].apply(
        lambda x: len(extract_compared_plan_ids(x).split(',')) if extract_compared_plan_ids(x) else 0
    )
    output_df['submitted_application'] = filtered_data['userActions.targetUrl'].str.contains(
        r'online-application.html', case=False, na=False).astype(int)
    print(f"Rows after session metrics: {len(output_df)}")
    
    # Synchronized deduplication
    def deduplicate_state(df):
        if 'startTime' in df.columns and not df['startTime'].isna().all():
            df = df.sort_values('startTime')
        return df.drop_duplicates(subset=['userid'], keep='first').reset_index(drop=True)
    
    output_df = deduplicate_state(output_df)
    filtered_data = deduplicate_state(filtered_data)
    print(f"Rows after final deduplication: {len(output_df)}")
    
    # Persona logic
    top_priority_mapping = {
        'doctorPref': 'ma_provider_network', 
        'drugPref': 'ma_drug_coverage', 
        'additional-vision': 'ma_vision',
        'additional-dental': 'ma_dental_benefit', 
        'additional-hearing': 'ma_hearing', 
        'additional-fitness': 'ma_fitness',
        'healthCarePref': 'ma_healthcare'
    }
    
    persona_mapping = {
        'ma_provider_network': 'doctor', 
        'ma_drug_coverage': 'drug', 
        'ma_vision': 'vision',
        'ma_dental_benefit': 'dental', 
        'ma_hearing': 'hearing', 
        'ma_fitness': 'fitness',
        'ma_healthcare': 'healthcare',
        'csnp': 'csnp',
        'dsnp': 'dsnp'
    }
    
    def determine_persona(row):
        specialneeds_option = row.get('userActions.extracted_data.text.specialneeds_option', '')
        drugs_option = row.get('userActions.extracted_data.text.drugs_option', '')
        top_priority = row.get('userActions.extracted_data.text.topPriority', np.nan)
        
        persona_parts = []
        
        if isinstance(specialneeds_option, str) and '["snp_chronic"]' in specialneeds_option:
            if pd.notna(top_priority) and top_priority != 'csnp':
                persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
                persona_parts.append('csnp')
            else:
                persona_parts.append('csnp')
        
        if isinstance(specialneeds_option, str) and '["snp_medicaid"]' in specialneeds_option:
            if pd.notna(top_priority) and top_priority != 'dsnp':
                if not persona_parts:
                    persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
                persona_parts.append('dsnp')
            else:
                persona_parts.append('dsnp')
        
        if isinstance(drugs_option, str) and '["drug_yes"]' in drugs_option:
            persona_parts.append('ma_drug_coverage')
        
        if not persona_parts and pd.notna(top_priority):
            persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
        
        return ','.join(persona_parts) if persona_parts else 'unknown'
    
    def map_persona(persona):
        if pd.isna(persona) or persona == 'unknown' or persona == 'none':
            return persona
        parts = persona.split(',')
        mapped = [persona_mapping.get(part.strip(), part.strip()) for part in parts]
        return ','.join(mapped)
    
    # Apply persona
    output_df['persona'] = filtered_data.apply(determine_persona, axis=1)
    output_df['persona'] = output_df['persona'].apply(map_persona)
    print(f"Rows after persona assignment: {len(output_df)}")
    
    # Write the output to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Behavioral feature file saved to {output_file}")
    print(f"Final rows in output: {len(output_df)}")
    print(f"Non-null persona row count: {output_df['persona'].notna().sum()}")
