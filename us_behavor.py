import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# File paths
clickstream_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/us/union_elastic_us_0301_0331_2025.parquet'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/us_behavioral_0301_0331_2025_test.parquet'
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

# Define schema for output Parquet file
schema = pa.schema([
    ('userid', pa.string()),
    ('start_time', pa.string()),
    ('city', pa.string()),
    ('state', pa.string()),
    ('zip', pa.string()),
    ('plan_id', pa.string()),
    ('compared_plan_ids', pa.string()),
    ('query_dental', pa.int32()),
    ('query_transportation', pa.int32()),
    ('query_otc', pa.int32()),
    ('query_drug', pa.int32()),
    ('query_provider', pa.int32()),
    ('query_vision', pa.int32()),
    ('query_csnp', pa.int32()),
    ('query_dsnp', pa.int32()),
    ('filter_dental', pa.int32()),
    ('filter_transportation', pa.int32()),
    ('filter_otc', pa.int32()),
    ('filter_drug', pa.int32()),
    ('filter_provider', pa.int32()),
    ('filter_vision', pa.int32()),
    ('filter_csnp', pa.int32()),
    ('filter_dsnp', pa.int32()),
    ('accordion_dental', pa.float64()),
    ('accordion_transportation', pa.float64()),
    ('accordion_otc', pa.float64()),
    ('accordion_drug', pa.float64()),
    ('accordion_provider', pa.float64()),
    ('accordion_vision', pa.float64()),
    ('accordion_csnp', pa.float64()),
    ('accordion_dsnp', pa.float64()),
    ('time_dental_pages', pa.float64()),
    ('time_transportation_pages', pa.float64()),
    ('time_otc_pages', pa.float64()),
    ('time_drug_pages', pa.float64()),
    ('time_provider_pages', pa.float64()),
    ('time_vision_pages', pa.float64()),
    ('time_csnp_pages', pa.float64()),
    ('time_dsnp_pages', pa.float64()),
    ('rel_time_dental_pages', pa.float64()),
    ('rel_time_transportation_pages', pa.float64()),
    ('rel_time_otc_pages', pa.float64()),
    ('rel_time_drug_pages', pa.float64()),
    ('rel_time_provider_pages', pa.float64()),
    ('rel_time_vision_pages', pa.float64()),
    ('rel_time_csnp_pages', pa.float64()),
    ('rel_time_dsnp_pages', pa.float64()),
    ('total_session_time', pa.float64()),
    ('num_pages_viewed', pa.int32()),
    ('num_plans_selected', pa.int32()),
    ('num_plans_compared', pa.int32()),
    ('submitted_application', pa.int32()),
    ('persona', pa.string())
])

# Process a single chunk with top 10000 records
def process_chunk(chunk):
    print(f"Initial chunk rows: {len(chunk)}")
    if not chunk.empty:
        print("Sample of input chunk:")
        print(chunk[['internalUserId', 'startTime', 'userActions.targetUrl']].head())
        # Check all available columns
        print("All columns in chunk:")
        print(chunk.columns.tolist())
        # Check persona-related columns with existence check
        persona_cols = ['userActions.extracted_data.text.specialneeds_option', 
                        'userActions.extracted_data.text.drugs_option', 
                        'userActions.extracted_data.text.topPriority']
        for col in persona_cols:
            if col in chunk.columns:
                print(f"Sample of {col}:")
                print(chunk[col].head())
                print(f"Non-null count in {col}: {chunk[col].notna().sum()}")
            else:
                print(f"Column {col} not found in chunk")
    
    # Filter only for non-null internalUserId
    chunk = chunk[chunk['internalUserId'].notna()].reset_index(drop=True)
    print(f"Rows after internalUserId filter: {len(chunk)}")
    
    # Initial deduplication within chunk
    chunk = chunk.drop_duplicates(subset=['internalUserId', 'userActions.targetUrl'], keep='first').reset_index(drop=True)
    print(f"Rows after initial deduplication: {len(chunk)}")
    
    # Initialize output_df for chunk
    output_df = pd.DataFrame(index=chunk.index, columns=output_columns)
    
    # Populate basic fields
    output_df['userid'] = chunk['internalUserId']
    output_df['start_time'] = chunk['startTime']
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
    
    # Final deduplication by userid
    def deduplicate_state(df):
        df = df.sort_values('start_time')
        return df.groupby('userid').first().reset_index()
    
    output_df = deduplicate_state(output_df)
    chunk = chunk[chunk['internalUserId'].isin(output_df['userid'])].reset_index(drop=True)
    print(f"Rows after final deduplication: {len(output_df)}")
    
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
        
        # Debug input values
        print(f"Row {row.name}: specialneeds_option={specialneeds_option}, drugs_option={drugs_option}, top_priority={top_priority}")
        
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
        print(f"Row {row.name}: Persona result = {result}")
        return str(result)
    
    def map_persona(persona):
        if pd.isna(persona) or persona == 'unknown':
            return persona
        parts = persona.split(',')
        mapped = [persona_mapping.get(part.strip(), part.strip()) for part in parts]
        result = ','.join(mapped)
        print(f"Mapping persona: {persona} -> {result}")
        return result
    
    if not chunk.empty:
        print("Debugging determine_persona on first 5 rows:")
        for idx, row in chunk.head(5).iterrows():
            result = determine_persona(row)
            print(f"Row {idx}: Result = {result}, Type = {type(result)}")
    
    # Apply persona and debug intermediate steps
    chunk['persona'] = chunk.apply(determine_persona, axis=1)
    print("Sample of chunk with persona:")
    print(chunk[['internalUserId', 'persona']].head())
    
    persona_df = chunk.groupby('internalUserId')['persona'].first().reset_index()
    print("Sample of persona_df:")
    print(persona_df.head())
    
    output_df = output_df.merge(persona_df, left_on='userid', right_on='internalUserId', how='left', suffixes=('', '_drop'))
    print("Sample of output_df after merge:")
    print(output_df[['userid', 'persona']].head())
    
    output_df = output_df.drop(columns=[col for col in output_df.columns if col.endswith('_drop')])
    output_df['persona'] = output_df['persona'].apply(map_persona)
    print(f"Rows after persona assignment: {len(output_df)}")
    print("Sample of final output_df:")
    print(output_df[['userid', 'persona']].head())
    
    return output_df

# Process only the top 10000 records from the first row group
row_group = parquet_file.read_row_group(0).to_pandas()  # First row group
chunk = row_group.head(10000).reset_index(drop=True)  # Top 10000 rows
print(f"Processing test chunk with {len(chunk)} rows")

# Process the chunk
output_chunk = process_chunk(chunk)

# Write the output
writer = pq.ParquetWriter(output_file, schema, compression='snappy')
table = pa.Table.from_pandas(output_chunk, schema=schema, preserve_index=False)
writer.write_table(table)
writer.close()

print(f"Test behavioral feature file saved to {output_file}")
print(f"Final rows in output: {len(output_chunk)}")
print(f"Non-null persona row count: {output_chunk['persona'].notna().sum()}")
