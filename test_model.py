import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_behavioral_features_0901_2024_0228_2025.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_csnp_focus.pkl'

def load_model(model_path):
    """Load the trained Random Forest model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_behavioral_data(behavioral_path):
    """Load behavioral data."""
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print("Columns available:", list(behavioral_df.columns))
        return behavioral_df
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        raise

def prepare_features(behavioral_df):
    """Prepare features from behavioral data to match training data, without plan data or weights."""
    # Define feature columns expected by the model (based on training_dataset_8.csv)
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

    # Features added during training (assuming they were derived from behavioral data)
    behavioral_df['csnp_interaction'] = 0  # Placeholder, no csnp from plan data
    behavioral_df['csnp_type_flag'] = 0    # Placeholder, no csnp_type
    behavioral_df['csnp_signal_strength'] = (behavioral_df['query_csnp'] + behavioral_df['filter_csnp'] + behavioral_df['accordion_csnp'] + behavioral_df['time_csnp_pages']).clip(upper=3)

    additional_features = ['csnp_interaction', 'csnp_type_flag', 'csnp_signal_strength']

    # Placeholder for plan features (set to 0 since no plan data)
    raw_plan_features = [
        'ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp',
        'ma_provider_network', 'ma_drug_coverage'
    ]
    for feature in raw_plan_features:
        behavioral_df[feature] = 0

    # Placeholder for persona weights (set to 0 since no weight recalculation)
    persona_weights = ['doctor', 'drug', 'vision', 'dental', 'otc', 'transportation', 'csnp', 'dsnp', 'fitness', 'hearing']
    for persona in persona_weights:
        behavioral_df[f'w_{persona}'] = 0

    # Combine all features expected by the model
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + [f'w_{persona}' for persona in persona_weights]

    # Select features and handle missing values
    X = behavioral_df[feature_columns].fillna(0)
    y = behavioral_df['persona']  # Assuming 'persona' is the ground truth column

    return X, y

def evaluate_model(model, X, y):
    """Evaluate model performance."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print(f"Total rows evaluated: {len(y)}")
    print(f"Correct predictions: {sum(y_pred == y)}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred))

def main():
    print("Evaluating Random Forest model performance on behavioral data...")

    # Load model and data
    rf_model = load_model(model_file)
    behavioral_df = load_behavioral_data(behavioral_file)

    # Prepare features and ground truth
    X, y = prepare_features(behavioral_df)

    # Evaluate model
    evaluate_model(rf_model, X, y)

if __name__ == "__main__":
    main()
