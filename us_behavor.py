import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Load the cleaned clickstream data
clickstream_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/us/union_elastic_us_0301_0331_2025.parquet'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/us_behavioral_0301_0331_2025.parquet'
print(f"Loading file: {clickstream_file}")

# Use pyarrow to read Parquet metadata
parquet_file = pq.ParquetFile(clickstream_file)
total_rows = parquet_file.metadata.num_rows
print(f"Initial rows loaded: {total_rows}")

# Check if file is empty
if total_rows == 0:
    print("Error: The input Parquet file is empty.")
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
    output_df = pd.DataFrame(columns=output_columns)
    output_df.to_parquet(output_file, index=False, compression='snappy')
    print(f"Saved empty output file to {output_file}")
    raise ValueError("Processing stopped due to empty input Parquet file.")

# Define chunk size (adjust based on memory capacity)
chunk_size = 100000

# Process in chunks using pyarrow
def process_chunk(chunk):
    # Filter only for non-null internalUserId
    chunk = chunk[chunk['internalUserId'].notna()].reset_index(drop=True)
    
    # Initial deduplication within chunk
    chunk = chunk.drop_duplicates(subset=['internalUserId', 'userActions.targetUrl'], keep='first').reset_index(drop=True)
    
    # Initialize output_df for chunk
    output_df = pd.DataFrame(index=chunk.index, columns=output_columns)
    
    # Populate basic fields
    output_df['userid'] = chunk['internalUserId']
    output_df['start_time'] = chunk['startTime']
    
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
    
    output_df['zip'] = chunk['userActions.targetUrl'].apply(extract_zip)
    output_df['plan_id'] = chunk['userActions.targetUrl'].apply(extract_plan_id)
    output_df['compared_plan_ids'] = chunk['userActions.targetUrl'].apply(extract_compared_plan_ids)
    output_df['city'] = chunk.get('city', pd.Series(index=chunk.index, dtype='object'))
    output_df['state'] = chunk.get('userActions.extracted_data.text.stateCode', pd.Series(index=chunk.index, dtype='object'))
    
    # Deduplicate compared_plan_ids
    def deduplicate_compared_plan_ids(df):
        df = df.sort_values('start_time')
        df['compared_plan_ids'] = df.groupby('userid')['compared_plan_ids'].transform(
            lambda x: x.where(x != x.shift()).fillna(x)
        )
        return df
    
    output_df = deduplicate_compared_plan_ids(output_df)
    
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
        output_df[col] = chunk['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
    for col, pattern in filter_mappings.items():
        output_df[col] = chunk['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
    output_df['filter_provider'] = 0
    
    # Ignored fields
    for col in ['accordion_dental', 'accordion_transportation', 'accordion_otc', 'accordion_drug',
                'accordion_provider', 'accordion_vision', 'accordion_csnp', 'accordion_dsnp',
                'time_dental_pages', 'time_transportation_pages', 'time_otc_pages', 'time_drug_pages',
                'time_provider_pages', 'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages',
                'rel_time_dental_pages', 'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
                'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages', 'rel_time_dsnp_pages']:
        output_df[col] = 0.0
    
    # Session metrics
    if 'duration' in chunk.columns:
        output_df['total_session_time'] = pd.to_numeric(chunk['duration'], errors='coerce').fillna(0.0)
    else:
        output_df['total_session_time'] = 0.0
    
    output_df['num_pages_viewed'] = 1
    output_df['num_plans_selected'] = chunk['userActions.targetUrl'].str.contains(
        'https://www.uhc.com/medicare/health-plans/details.html', case=False, na=False).astype(int)
    output_df['num_plans_compared'] = chunk['userActions.targetUrl'].apply(
        lambda x: len(extract_compared_plan_ids(x).split(',')) if extract_compared_plan_ids(x) else 0
    )
    output_df['submitted_application'] = chunk['userActions.targetUrl'].str.contains(
        r'online-application.html', case=False, na=False).astype(int)
    
    # Date filter
    date_pattern = r'^[A-Za-z]{3} \d{1,2}, \d{4} @ \d{2}:\d{2}:\d{2}\.\d{3}$'
    mask = output_df['start_time'].str.match(date_pattern, na=False)
    output_df = output_df[mask]
    chunk = chunk[mask].reset_index(drop=True)
    output_df = output_df.reset_index(drop=True)
    
    # Final deduplication by userid
    def deduplicate_state(df):
        df = df.sort_values('start_time')
        return df.groupby('userid').first().reset_index()
    
    output_df = deduplicate_state(output_df)
    chunk = chunk[chunk['internalUserId'].isin(output_df['userid'])].reset_index(drop=True)
    
    # Persona logic
    top_priority_mapping = {
        'doctorPref': 'ma_provider_network', 'drugPref': 'ma_drug_coverage', 'additional-vision': 'ma_vision',
        'additional-dental': 'ma_dental_benefit', 'additional-hearing': 'ma_hearing', 'additional-fitness': 'ma_fitness'
    }
    
    persona_mapping = {
        'ma_provider_network': 'doctor', 'ma_drug_coverage': 'drug', 'ma_vision': 'vision',
        'ma_dental_benefit': 'dental', 'ma_hearing': 'hearing', 'ma_fitness': 'fitness'
    }
    
    def determine_persona(row):
        specialneeds_option = row.get('userActions.extracted_data.text.specialneeds_option', '')
        drugs_option = row.get('userActions.extracted_data.text.drugs_option', '')
        top_priority = row.get('userActions.extracted_data.text.topPriority', np.nan)
        
        persona_parts = []
        
        if specialneeds_option and '["snp_chronic"]' in specialneeds_option:
            if pd.notna(top_priority) and top_priority != 'csnp':
                persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
                persona_parts.append('csnp')
            else:
                persona_parts.append('csnp')
        
        if specialneeds_option and '["snp_medicaid"]' in specialneeds_option:
            if pd.notna(top_priority) and top_priority != 'dsnp':
                if not persona_parts:
                    persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
                persona_parts.append('dsnp')
            else:
                persona_parts.append('dsnp')
        
        if drugs_option and '["drug_yes"]' in drugs_option:
            persona_parts.append('ma_drug_coverage')
        
        if not persona_parts and pd.notna(top_priority):
            persona_parts.append(top_priority_mapping.get(top_priority, top_priority))
        
        return ','.join(persona_parts) if persona_parts else 'unknown'
    
    def map_persona(persona):
        if pd.isna(persona) or persona == 'unknown':
            return persona
        parts = persona.split(',')
        mapped = [persona_mapping.get(part.strip(), part.strip()) for part in parts]
        return ','.join(mapped)
    
    chunk['persona'] = chunk.apply(determine_persona, axis=1)
    persona_df = chunk.groupby('internalUserId')['persona'].first().reset_index()
    output_df = output_df.merge(persona_df, left_on='userid', right_on='internalUserId', how='left', suffixes=('', '_drop'))
    output_df = output_df.drop(columns=[col for col in output_df.columns if col.endswith('_drop')])
    output_df['persona'] = output_df['persona'].apply(map_persona)
    
    return output_df

# Process chunks and append to output file
first_chunk = True
for i in range(parquet_file.num_row_groups):
    # Read one row group at a time (approximate chunking)
    chunk = parquet_file.read_row_group(i).to_pandas()
    print(f"Processing chunk {i + 1} with {len(chunk)} rows")
    output_chunk = process_chunk(chunk)
    mode = 'w' if first_chunk else 'a'  # Overwrite on first chunk, append on subsequent
    output_chunk.to_parquet(output_file, index=False, compression='snappy', engine='pyarrow', mode=mode)
    first_chunk = False
    print(f"Chunk {i + 1} processed and saved. Rows in chunk output: {len(output_chunk)}")

print(f"Behavioral feature file saved to {output_file}")
