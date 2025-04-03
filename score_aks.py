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
import uvicorn  # Required for running FastAPI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
model = None
behavioral_df_full = None
plan_df_full = None

# File definitions
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona.pkl'

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


# Feature preparation and scoring functions (unchanged)
def prepare_features(behavioral_df, plan_df):
    # [Your existing prepare_features function remains unchanged]
    pass  # Replace with your actual implementation

def score_data(model, X, metadata):
    # [Your existing score_data function remains unchanged]
    pass  # Replace with your actual implementation

# Define the /score endpoint
@app.post("/score")
async def score(request: ScoreRequest):
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
        for _, row in scored_df.iterrows():
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

# Health check endpoint (fixed syntax)
from fastapi.responses import Response

@app.get("/actuator/health")
async def health_check():
    """Check if the service is up and running."""
    return Response(content="Healthy", status_code=200)

# Initialize the model and data on startup
@app.on_event("startup")
async def startup_event():
    init()

# For local testing
if __name__ == "__main__":
    init()  # Load model and data
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
