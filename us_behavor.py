import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# File paths
clickstream_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/us/union_elastic_us_0301_0331_2025.parquet'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/us_behavioral_0301_0331_2025_full.csv'
print(f"Loading file: {clickstream_file}")

# Use pyarrow to read Parquet metadata
parquet_file = pq.ParquetFile(clickstream_file)
total_rows = parquet_file.metadata.num_rows
print(f"Total rows in file: {total_rows}")

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
    output_df.to_csv(output_file, index=False)
    print(f"Saved empty output file to {output_file}")
    raise ValueError("Processing stopped due to empty input Parquet file.")

# Process a single chunk
def process_chunk(chunk):
    # Rename internalUserId to userid at the beginning
    if 'internalUserId' in chunk.columns:
        chunk = chunk.rename(columns={'internalUserId': 'userid'})
    
    print(f"Processing chunk with {len(chunk)} rows")
    
    # Filter only for non-null userid
    chunk = chunk[chunk['userid'].notna()].reset_index(drop=True)
    print(f"Rows after userid filter: {len(chunk)}")
    
    # Initial deduplication within chunk
    chunk = chunk.drop_duplicates(subset=['userid', 'userActions.targetUrl'], keep='first').reset_index(drop=True)
    print(f"Rows after initial deduplication: {len(chunk)}")
    
    # Initialize output_df for chunk
    output_df = pd.DataFrame(index=chunk.index, columns=output_columns)
    
    # Populate basic fields
    output_df['userid'] = chunk['userid']
    # Handle start_time with fallback and convert to yyyy-mm-dd
    if 'startTime' in chunk.columns:
        output_df['start_time'] = pd.to_datetime(chunk['startTime'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        output_df['start_time'] = pd.Series(index=chunk.index, dtype='object')
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
    
    output_df['zip'] = chunk['userActions.targetUrl'].apply(extract_zip)
    output_df['plan_id'] = chunk['userActions.targetUrl'].apply(extract_plan_id)
    output_df['compared_plan_ids'] = chunk['userActions.targetUrl'].apply(extract_compared_plan_ids)
    output_df['city'] = chunk.get('city', pd.Series(index=chunk.index, dtype='object'))
    output_df['state'] = chunk.get('userActions.extracted_data.text.stateCode', pd.Series(index=chunk.index, dtype='object'))
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
        output_df[col] = chunk['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
    for col, pattern in filter_mappings.items():
        output_df[col] = chunk['userActions.targetUrl'].str.contains(pattern, case=False, na=False).astype(int)
    
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
    print(f"Rows after session metrics: {len(output_df)}")
    
    # Synchronized deduplication
    def deduplicate_state(df):
        if 'startTime' in df.columns and not df['startTime'].isna().all():
            df = df.sort_values('startTime')
        return df.drop_duplicates(subset=['userid'], keep='first').reset_index(drop=True)
    
    output_df = deduplicate_state(output_df)
    chunk = deduplicate_state(chunk)
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
        
        result = ','.join(persona_parts) if persona_parts else 'unknown'
        return str(result)
    
    def map_persona(persona):
        if pd.isna(persona) or persona == 'unknown' or persona == 'none':
            return persona
        parts = persona.split(',')
        mapped = [persona_mapping.get(part.strip(), part.strip()) for part in parts]
        return ','.join(mapped)
    
    # Apply persona directly to output_df
    output_df['persona'] = chunk.apply(determine_persona, axis=1)
    output_df['persona'] = output_df['persona'].apply(map_persona)
    print(f"Chunk processed, final rows: {len(output_df)}")
    
    return output_df

# Process the entire file in chunks of 10,000
chunk_size = 10000
output_chunks = []
rows_processed = 0

for i in range(parquet_file.num_row_groups):
    row_group = parquet_file.read_row_group(i).to_pandas()
    print(f"Reading row group {i+1}/{parquet_file.num_row_groups}, rows: {len(row_group)}")
    
    # Process in chunks of 10K
    for start in range(0, len(row_group), chunk_size):
        chunk = row_group.iloc[start:start + chunk_size].reset_index(drop=True)
        if not chunk.empty:
            output_chunk = process_chunk(chunk)
            output_chunks.append(output_chunk)
            rows_processed += len(chunk)
            print(f"Processed {rows_processed} rows so far")
        if rows_processed >= total_rows:
            break
    if rows_processed >= total_rows:
        break

# Combine all chunks
final_output = pd.concat(output_chunks, ignore_index=True)
print(f"Total rows after combining chunks: {len(final_output)}")

# Write to CSV
final_output.to_csv(output_file, index=False)
print(f"Behavioral feature file saved to {output_file}")
print(f"Non-null persona row count: {final_output['persona'].notna().sum()}")
