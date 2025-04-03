import pandas as pd
import numpy as np
import pickle
import os
import io
import json
import logging
from azure.cosmos import CosmosClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
import uvicorn
import nest_asyncio

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
model = None
behavioral_df_full = None
plan_df_full = None
app_ready = False  # For AKS health check

# File definitions
MODEL_FILE = "model-persona-0.0.1.pkl"

# FastAPI app instance
app = FastAPI()

# Input schema for the /score endpoint
class ScoreRequest(BaseModel):
    userid: str

# Function to load data from Cosmos DB into pandas DataFrame
def load_data_from_cosmos(container):
    query = "SELECT * FROM c"
    items = list(container.query_items(query, enable_cross_partition_query=True))
    return pd.DataFrame(items)

def init():
    """Initialize the model and load full datasets for the endpoint."""
    global model, behavioral_df_full, plan_df_full, app_ready
    try:
        # Load model
        model_path = MODEL_FILE
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")

        # Cosmos DB connection
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        database_name = os.getenv("COSMOS_PERSONA_DB")
        behavioral_container_name = os.getenv("COSMOS_PERSONA_CONTAINER_BEHAVIOR")
        plan_container_name = os.getenv("COSMOS_PERSONA_CONTAINER_PLAN")

        if not all([endpoint, key, database_name, behavioral_container_name, plan_container_name]):
            raise ValueError("One or more Cosmos DB environment variables are not set")

        # Initialize Cosmos DB client
        client = CosmosClient(endpoint, credential=key)
        logger.info("Successfully connected to Cosmos DB")
        
        database = client.get_database_client(database_name)
        behavioral_container = database.get_container_client(behavioral_container_name)
        plan_container = database.get_container_client(plan_container_name)

        # Load data
        behavioral_df_full = load_data_from_cosmos(behavioral_container)
        plan_df_full = load_data_from_cosmos(plan_container)

        # Convert numeric columns to appropriate types
        numeric_columns = [
            'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
            'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug',
            'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp', 'accordion_dental',
            'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
            'accordion_vision', 'accordion_csnp', 'accordion_dsnp', 'time_dental_pages',
            'time_transportation_pages', 'time_otc_pages', 'time_drug_pages', 'time_provider_pages',
            'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages', 'rel_time_dental_pages',
            'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
            'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages',
            'rel_time_dsnp_pages', 'total_session_time', 'num_pages_viewed', 'num_plans_selected',
            'num_plans_compared', 'submitted_application', 'dce_click_count', 'pro_click_count',
            'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
            'ma_provider_network', 'ma_drug_coverage'
        ]
        for df in [behavioral_df_full, plan_df_full]:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info(f"Behavioral data loaded: {len(behavioral_df_full)} rows")
        logger.info(f"Plan data loaded: {len(plan_df_full)} rows")
        
        app_ready = True  # Mark app as ready
    except Exception as e:
        logger.error(f"Error in init: {str(e)}")
        raise

# Feature preparation and scoring functions
def prepare_features(behavioral_df, plan_df):
    """Prepare features and assign quality levels for scoring, joining with plan_df."""
    df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    logger.info(f"Rows after merge with plan data: {len(df)}")

    df['state'] = df['state_beh'].fillna(df['state_plan'])
    df = df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    all_behavioral_features = [
        'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider', 'query_vision',
        'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation', 'filter_otc', 'filter_drug',
        'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp', 'accordion_dental',
        'accordion_transportation', 'accordion_otc', 'accordion_drug', 'accordion_provider',
        'accordion_vision', 'accordion_csnp', 'accordion_dsnp', 'time_dental_pages',
        'time_transportation_pages', 'time_otc_pages', 'time_drug_pages', 'time_provider_pages',
        'time_vision_pages', 'time_csnp_pages', 'time_dsnp_pages', 'rel_time_dental_pages',
        'rel_time_transportation_pages', 'rel_time_otc_pages', 'rel_time_drug_pages',
        'rel_time_provider_pages', 'rel_time_vision_pages', 'rel_time_csnp_pages',
        'rel_time_dsnp_pages', 'total_session_time', 'num_pages_viewed', 'num_plans_selected',
        'num_plans_compared', 'submitted_application', 'dce_click_count', 'pro_click_count'
    ]

    raw_plan_features = [
        'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
        'ma_provider_network', 'ma_drug_coverage'
    ]

    df['csnp_interaction'] = df['csnp'].fillna(0) * (
        df['query_csnp'].fillna(0) + df['filter_csnp'].fillna(0) + 
        df['time_csnp_pages'].fillna(0) + df['accordion_csnp'].fillna(0)
    ) * 2
    df['csnp_type_flag'] = (df['csnp_type'] == 'Y').astype(int) if 'csnp_type' in df.columns else 0
    df['csnp_signal_strength'] = (
        df['query_csnp'].fillna(0) + df['filter_csnp'].fillna(0) + 
        df['accordion_csnp'].fillna(0) + df['time_csnp_pages'].fillna(0)
    ).clip(upper=5) * 1.5

    additional_features = ['csnp_interaction', 'csnp_type_flag', 'csnp_signal_strength']

    persona_weights = {
        'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'pro_click_count'},
        'drug': {'plan_col': 'ma_drug_coverage', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'dce_click_count'},
        'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'},
        'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
        'otc': {'plan_col': 'ma_otc', 'query_col': 'query_otc', 'filter_col': 'filter_otc'},
        'transportation': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation'},
        'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
        'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
        'fitness': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation'},
        'hearing': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
    }

    k1, k3, k4, k7, k8 = 0.1, 0.5, 0.4, 0.15, 0.25
    k9, k10 = 1.0, 0.9
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 1.0, 3.0, 1.0, 1.5

    def calculate_persona_weight(row, persona_info, persona):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)

        base_weight = 0
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(float(row[plan_col]), 0.7 if persona == 'csnp' else 0.5)
            if persona == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y':
                base_weight *= W_CSNP_HIGH
            elif persona == 'csnp':
                base_weight *= W_CSNP_BASE
            elif persona == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y':
                base_weight *= W_DSNP_HIGH
            elif persona == 'dsnp':
                base_weight *= W_DSNP_BASE
        elif pd.isna(row['plan_id']) and pd.notna(row['compared_plan_ids']) and isinstance(row['compared_plan_ids'], str) and row['num_plans_compared'] > 0:
            compared_ids = row['compared_plan_ids'].split(',')
            compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
            if not compared_plans.empty:
                base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
                if persona == 'csnp':
                    csnp_type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                    base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
                elif persona == 'dsnp':
                    dsnp_type_y_ratio = (compared_plans['dsnp_type'] == 'Y').mean()
                    base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)

        pages_viewed = min(float(row['num_pages_viewed']), 3) if pd.notna(row['num_pages_viewed']) else 0
        query_value = float(row[query_col]) if pd.notna(row[query_col]) else 0
        filter_value = float(row[filter_col]) if pd.notna(row[filter_col]) else 0
        click_value = float(row[click_col]) if click_col and pd.notna(row[click_col]) else 0

        query_coeff = k9 if persona == 'csnp' else k3
        filter_coeff = k10 if persona == 'csnp' else k4
        click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0

        behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value

        if persona == 'doctor':
            if click_value >= 1.5:
                behavioral_score += 0.4
            elif click_value >= 0.5:
                behavioral_score += 0.2
        elif persona == 'drug':
            if click_value >= 5:
                behavioral_score += 0.4
            elif click_value >= 2:
                behavioral_score += 0.2
        elif persona == 'dental':
            signal_count = sum(1 for val in [query_value, filter_value, pages_viewed] if val > 0)
            if signal_count >= 2:
                behavioral_score += 0.3
            elif signal_count >= 1:
                behavioral_score += 0.15
        elif persona == 'vision':
            signal_count = sum(1 for val in [query_value, filter_value, pages_viewed] if val > 0)
            if signal_count >= 1:
                behavioral_score += 0.35
        elif persona == 'csnp':
            signal_count = sum(1 for val in [query_value, filter_value, pages_viewed] if val > 0)
            if signal_count >= 2:
                behavioral_score += 0.6
            elif signal_count >= 1:
                behavioral_score += 0.5
            if row['csnp_interaction'] > 0:
                behavioral_score += 0.3
            if row['csnp_type_flag'] == 1:
                behavioral_score += 0.2
        elif persona in ['fitness', 'hearing']:
            signal_count = sum(1 for val in [query_value, filter_value, pages_viewed] if val > 0)
            if signal_count >= 1:
                behavioral_score += 0.3

        adjusted_weight = base_weight + behavioral_score
        return min(adjusted_weight, 2.0 if persona == 'csnp' else 1.0)

    for persona, info in persona_weights.items():
        df[f'w_{persona}'] = df.apply(lambda row: calculate_persona_weight(row, info, persona), axis=1)

    weighted_features = [f'w_{persona}' for persona in persona_weights.keys() if persona != 'csnp']
    weight_sum = df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        df[wf] = df[wf] / weight_sum.where(weight_sum > 0, 1)

    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights.keys()]
    
    X = df[feature_columns].fillna(0)
    metadata = df[['userid']]

    def assign_quality_level(row):
        has_plan_id = pd.notna(row['plan_id'])
        has_clicks = (row['dce_click_count'] > 0 and pd.notna(row['dce_click_count'])) or \
                     (row['pro_click_count'] > 0 and pd.notna(row['pro_click_count']))
        has_filters = any(row[col] > 0 and pd.notna(row[col]) for col in df.columns if col.startswith('filter_'))
        has_queries = any(row[col] > 0 and pd.notna(row[col]) for col in df.columns if col.startswith('query_'))
        
        if not has_plan_id:
            return 'Low'
        elif has_plan_id and (has_clicks or has_filters):
            return 'High'
        elif has_plan_id and has_queries:
            return 'Medium'
        else:
            return 'Medium'

    df['quality_level'] = df.apply(assign_quality_level, axis=1)
    metadata['quality_level'] = df['quality_level']

    return X, metadata

def score_data(model, X, metadata):
    """Score the data and return results with quality level and top 3 persona rankings."""
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])

    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)

    def get_top_3_ranking(row):
        probs = {p: row[f'prob_{p}'] for p in personas}
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        return {persona: round(prob, 4) for persona, prob in top_3}

    output_df['prediction_scores'] = output_df.apply(get_top_3_ranking, axis=1)

    final_df = output_df[['userid', 'quality_level', 'prediction_scores']]

    return final_df

# Define the /score endpoint
@app.post("/score")
async def score(request: ScoreRequest):
    """Score endpoint to process user data and return predictions."""
    if not app_ready:
        raise HTTPException(status_code=503, detail="Service is still initializing")
    try:
        userid = request.userid
        if not userid:
            raise HTTPException(status_code=400, detail="Missing 'userid' in request payload")

        behavioral_df = behavioral_df_full[behavioral_df_full['userid'] == userid]
        if behavioral_df.empty:
            raise HTTPException(status_code=404, detail=f"No behavioral data found for userid: {userid}")

        zip_plan_pairs = behavioral_df[['zip', 'plan_id']].drop_duplicates()
        plan_df = plan_df_full[plan_df_full.set_index(['zip', 'plan_id']).index.isin(zip_plan_pairs.set_index(['zip', 'plan_id']).index)]
        
        logger.info(f"Behavioral data rows for {userid}: {len(behavioral_df)}")
        logger.info(f"Plan data rows for {userid}: {len(plan_df)}")

        X, metadata = prepare_features(behavioral_df, plan_df)
        scored_df = score_data(model, X, metadata)

        result = []
        for _, row in scored_df.iterrows():  # Fixed typo: 'scoreddf' to 'scored_df'
            output = {
                'userid': row['userid'],
                'quality_level': row['quality_level'],
                'prediction_scores': row['prediction_scores']
            }
            if row['quality_level'] == 'Low':
                output['message'] = 'prediction result might not be accurate due to low behavioral data quality'
            result.append(output)

        return {'scored_results': result}
    except Exception as e:
        logger.error(f"Error during scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during scoring: {str(e)}")

# Health check endpoint
@app.get("/actuator/health")
async def health_check():
    """Check if the service is up and running."""
    if not app_ready:
        raise HTTPException(status_code=503, detail="Service is still initializing")
    return Response(content="Healthy", status_code=200)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init()

# Driver main method compatible with Databricks and AKS
def run_app():
    """Run the FastAPI app, handling both Databricks and AKS environments."""
    try:
        # Check if running in Databricks (interactive environment)
        if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
            logger.info("Detected Databricks environment")
            nest_asyncio.apply()  # Allow nested event loops in Databricks
            uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
        else:
            # Assume AKS or standalone Python environment
            logger.info("Running in standalone/AKS environment")
            uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
    except Exception as e:
        logger.error(f"Error running app: {str(e)}")
        raise

if __name__ == "__main__":
    run_app()
