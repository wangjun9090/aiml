import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

# Hardcoded file paths for Databricks
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-0.0.3.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder.pkl'  # New
OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_results_0401_2025_0420_2025.csv'
SUMMARY_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_summary_0401_2025_0420_2025.csv'

def load_model_and_encoder(model_path, encoder_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        print(f"Model classes: {model.classes_}")
        
        try:
            with open(encoder_path, 'rb') as f:
                le = pickle.load(f)
            print(f"Label encoder loaded from {encoder_path}")
            print(f"Label encoder classes: {le.classes_}")
        except FileNotFoundError:
            print(f"Warning: Label encoder file {encoder_path} not found. Using default persona mapping.")
            # Default mapping based on expected personas
            le = LabelEncoder()
            le.classes_ = np.array(['csnp', 'dental', 'doctor', 'drug', 'dsnp', 'vision'])
            print(f"Default label encoder classes: {le.classes_}")
        
        return model, le
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
        raise

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Plan data loaded: {len(plan_df)} rows")
        print(f"Behavioral_df columns: {behavioral_df.columns.tolist()}")
        print(f"Plan_df columns: {plan_df.columns.tolist()}")
        print(f"Unique personas in behavioral_df: {behavioral_df['persona'].unique().tolist()}")
        
        # Ensure correct data types for merge columns
        behavioral_df['userid'] = behavioral_df['userid'].astype(str)
        behavioral_df['zip'] = behavioral_df['zip'].astype(str)
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str)
        plan_df['zip'] = plan_df['zip'].astype(str)
        plan_df['plan_id'] = plan_df['plan_id'].astype(str)
        
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    """Normalize personas for ground truth evaluation (split multiple personas into separate rows)."""
    new_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row['persona']):
            print(f"Warning: NaN persona at index {idx}")
            continue
        personas = [p.strip().lower() for p in str(row['persona']).split(',')]
        if not personas or personas[0] == '':
            print(f"Warning: Empty persona at index {idx}")
            continue
        if 'dsnp' in personas or 'csnp' in personas:
            first_row = row.copy()
            first_persona = personas[0]
            if first_persona in ['unknown', 'none', 'healthcare', '']:
                first_persona = 'dsnp' if 'dsnp' in personas else 'csnp'
            first_row['persona'] = first_persona
            new_rows.append(first_row)
            second_row = row.copy()
            second_row['persona'] = 'dsnp' if 'dsnp' in personas else 'csnp'
            new_rows.append(second_row)
        else:
            row_copy = row.copy()
            row_copy['persona'] = personas[0]
            new_rows.append(row_copy)
    
    if not new_rows:
        print("Warning: No valid rows after persona normalization")
        return pd.DataFrame(columns=df.columns)
        
    normalized_df = pd.DataFrame(new_rows).reset_index(drop=True)
    print(f"Rows after persona normalization: {len(normalized_df)}")
    print(f"Unique personas after normalization: {normalized_df['persona'].unique().tolist()}")
    return normalized_df

def prepare_evaluation_features(behavioral_df, plan_df, model, le):
    # Ensure correct data types for all relevant columns
    behavioral_df['userid'] = behavioral_df['userid'].astype(str)
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')
    
    feature_df = behavioral_df.copy()
    
    training_df = feature_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    ).reset_index(drop=True)
    print(f"Rows after merge: {len(training_df)}")
    print("Columns after merge:", training_df.columns.tolist())

    if len(training_df) == 0:
        print("Error: Merge resulted in empty DataFrame. Check zip/plan_id compatibility.")
        print(f"Behavioral_df zip/plan_id sample: {feature_df[['zip', 'plan_id']].head().to_string()}")
        print(f"Plan_df zip/plan_id sample: {plan_df[['zip', 'plan_id']].head().to_string()}")
        return None, None, None, None

    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore').reset_index(drop=True)

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

    for col in raw_plan_features + ['csnp_type']:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    # Fill missing feature columns with 0
    for col in all_behavioral_features:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]

    def assign_quality_level(row):
        has_plan_id = pd.notna(row['plan_id']) and row['plan_id'] != ''
        has_clicks = (row.get('dce_click_count', 0) > 0 and pd.notna(row.get('dce_click_count'))) or \
                     (row.get('pro_click_count', 0) > 0 and pd.notna(row.get('pro_click_count')))
        has_filters = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in filter_cols)
        has_queries = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in query_cols)
        
        if has_plan_id and (has_clicks or has_filters):
            return 'High'
        elif has_plan_id and not has_clicks and not has_filters and has_queries:
            return 'Medium'
        elif not has_plan_id and not has_clicks and not has_filters and not has_queries:
            return 'Low'
        else:
            return 'Medium'

    training_df['quality_level'] = training_df.apply(assign_quality_level, axis=1)
    print(f"Columns after adding quality_level: {training_df.columns.tolist()}")
    training_df = training_df.reset_index(drop=True)

    additional_features = []
    training_df['csnp_interaction'] = training_df['csnp'] * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 2.5
    additional_features.append('csnp_interaction')

    training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 2.5
    additional_features.append('csnp_signal_strength')

    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * training_df['ma_dental_benefit'] * 1.5
    additional_features.append('dental_interaction')

    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * training_df['ma_vision'] * 1.5
    additional_features.append('vision_interaction')

    training_df['csnp_drug_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.0 - training_df['ma_drug_coverage'] * (
            training_df.get('query_drug', 0).fillna(0) + training_df.get('filter_drug', 0).fillna(0) + 
            training_df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 2.5
    additional_features.append('csnp_drug_interaction')

    training_df['csnp_doctor_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0)
        ) * 1.5 - training_df['ma_provider_network'] * (
            training_df.get('query_provider', 0).fillna(0) + training_df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 1.5
    additional_features.append('csnp_doctor_interaction')

    training_df['vision_signal'] = (
        training_df['query_vision'].fillna(0) +
        training_df['filter_vision'].fillna(0) +
        training_df['time_vision_pages'].fillna(0).clip(upper=5)
    ) * 2.0
    additional_features.append('vision_signal')

    training_df['dental_signal'] = (
        training_df['query_dental'].fillna(0) +
        training_df['filter_dental'].fillna(0) +
        training_df['time_dental_pages'].fillna(0).clip(upper=5)
    ) * 2.0
    additional_features.append('dental_signal')

    training_df['csnp_specific_signal'] = (
        training_df['query_csnp'].fillna(0) +
        training_df['filter_csnp'].fillna(0) +
        training_df['csnp_drug_interaction'].fillna(0) +
        training_df['csnp_doctor_interaction'].fillna(0)
    ).clip(upper=5) * 3.0
    additional_features.append('csnp_specific_signal')

    training_df = training_df.reset_index(drop=True)

    feature_weights = {
        'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'pro_click_count', 'interaction_col': None},
        'drug': {'plan_col': 'ma_drug_coverage', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'dce_click_count', 'interaction_col': None},
        'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision', 'interaction_col': 'vision_interaction'},
        'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental', 'interaction_col': 'dental_interaction'},
        'otc': {'plan_col': 'ma_otc', 'query_col': 'query_otc', 'filter_col': 'filter_otc', 'interaction_col': None},
        'transportation': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation', 'interaction_col': None},
        'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp', 'interaction_col': 'csnp_interaction'},
        'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp', 'interaction_col': None}
    }

    k1, k3, k4, k7, k8 = 0.15, 0.8, 0.7, 0.3, 0.4
    k9, k10 = 2.5, 2.3
    W_BASE, W_HIGH = 1.5, 3.0

    def calculate_feature_weight(row, feature_info, plan_df):
        plan_col = feature_info['plan_col']
        query_col = feature_info['query_col']
        filter_col = feature_info['filter_col']
        click_col = feature_info.get('click_col', None)
        interaction_col = feature_info.get('interaction_col', None)
        
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(row[plan_col], 0.5)
            if plan_col in ['csnp', 'dsnp'] and row.get('csnp_type', 'N') == 'Y':
                base_weight *= W_HIGH
            else:
                base_weight *= W_BASE
        elif pd.notna(row.get('compared_plan_ids')) and isinstance(row['compared_plan_ids'], str) and row.get('num_plans_compared', 0) > 0:
            compared_ids = row['compared_plan_ids'].split(',')
            compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
            if not compared_plans.empty and plan_col in compared_plans.columns:
                base_weight = min(compared_plans[plan_col].mean(), 0.5)
                if plan_col in ['csnp', 'dsnp'] and 'csnp_type' in compared_plans.columns:
                    type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                    base_weight *= (W_BASE + (W_HIGH - W_BASE) * type_y_ratio)
                else:
                    base_weight *= W_BASE
            else:
                base_weight = 0
        else:
            base_weight = 0
        
        pages_viewed = min(row.get('num_pages_viewed', 0), 3) if pd.notna(row.get('num_pages_viewed')) else 0
        query_value = row.get(query_col, 0) if pd.notna(row.get(query_col)) else 0
        filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col)) else 0
        click_value = row.get(click_col, 0) if click_col and click_col in row and pd.notna(row.get(click_col)) else 0
        interaction_value = row.get(interaction_col, 0) if interaction_col and pd.notna(row.get(interaction_col)) else 0
        
        query_coeff = k9 if 'csnp' in query_col else k3
        filter_coeff = k10 if 'csnp' in filter_col else k4
        click_coefficient = k8 if click_col == 'pro_click_count' else k7 if click_col == 'dce_click_count' else 0
        interaction_coeff = 1.5 if interaction_col and 'interaction' in interaction_col else 1.0
        
        behavioral_score = (
            query_coeff * query_value +
            filter_coeff * filter_value +
            k1 * pages_viewed +
            click_coefficient * click_value +
            interaction_coeff * interaction_value
        )
        
        has_filters = any(row.get(col, 0) > 0 and pd.notna(row.get(col)) for col in filter_cols)
        has_clicks = (row.get('dce_click_count', 0) > 0 and pd.notna(row.get('dce_click_count'))) or \
                     (row.get('pro_click_count', 0) > 0 and pd.notna(row.get('pro_click_count')))
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed, interaction_value] if val > 0])
        if signal_count >= 3:
            behavioral_score += 0.8
        elif signal_count >= 2:
            behavioral_score += 0.5
        
        if has_filters and has_clicks:
            behavioral_score += 0.8
        elif has_filters or has_clicks:
            behavioral_score += 0.4
        
        if row['quality_level'] == 'High':
            behavioral_score += 0.5
        elif row['quality_level'] == 'Medium':
            behavioral_score += 0.2
        
        adjusted_weight = base_weight + behavioral_score
        return min(adjusted_weight, 1.2)

    print("Calculating feature weights...")
    for feature, info in feature_weights.items():
        click_col = info.get('click_col', 'click_dummy')
        if click_col not in training_df.columns:
            training_df[click_col] = 0
        training_df[f'w_{feature}'] = training_df.apply(
            lambda row: calculate_feature_weight(row, info, plan_df), axis=1
        )

    weighted_features = [f'w_{feature}' for feature in feature_weights.keys()]
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)
        training_df[wf] = training_df[wf].clip(upper=1.0)

    print(f"Weighted features added: {weighted_features}")
    print("Sample weights:")
    print(training_df[weighted_features].head())

    all_weighted_features = [f'w_{feature}' for feature in [
        'doctor', 'drug', 'vision', 'dental', 'otc', 'transportation', 'csnp', 'dsnp'
    ]]
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + all_weighted_features

    print(f"Feature columns expected: {feature_columns}")

    missing_features = [col for col in feature_columns if col not in training_df.columns]
    if missing_features:
        print(f"Warning: Missing features in training_df: {missing_features}")
        for col in missing_features:
            training_df[col] = 0

    print(f"Columns in training_df after filling: {training_df.columns.tolist()}")

    # Prepare ground truth using persona column
    ground_truth_df = normalize_persona(behavioral_df)
    if len(ground_truth_df) == 0:
        print("Error: Normalized ground truth is empty")
        return None, None, None, None

    # Ensure correct data types for join columns
    ground_truth_df['userid'] = ground_truth_df['userid'].astype(str)
    ground_truth_df['zip'] = ground_truth_df['zip'].astype(str)
    ground_truth_df['plan_id'] = ground_truth_df['plan_id'].astype(str)
        
    # Encode ground truth personas
    valid_personas = [p for p in ground_truth_df['persona'].unique() if p in le.classes_]
    print(f"Valid personas for encoding: {valid_personas}")
    valid_ground_truth = ground_truth_df[ground_truth_df['persona'].isin(valid_personas)].reset_index(drop=True)
    print(f"Rows in valid ground truth: {len(valid_ground_truth)}")
    print(f"Unique personas in ground truth: {valid_ground_truth['persona'].unique().tolist()}")

    if len(valid_ground_truth) == 0:
        print("Error: No valid ground truth personas match label encoder classes.")
        print(f"Label encoder classes: {le.classes_}")
        return None, None, None, None

    # Encode y_true
    y_true = le.transform(valid_ground_truth['persona'])
    valid_ground_truth['persona_encoded'] = y_true
    print(f"Encoded ground truth personas: {valid_ground_truth['persona_encoded'].unique().tolist()}")

    # Debug join columns
    print("Training dataframe dtypes for join columns:")
    print(training_df[['userid', 'zip', 'plan_id']].dtypes)
    print("Ground truth dataframe dtypes for join columns:")
    print(valid_ground_truth[['userid', 'zip', 'plan_id']].dtypes)
    
    # Check for missing or NaN values in join columns
    print(f"Training dataframe join columns NaN counts: {training_df[['userid', 'zip', 'plan_id']].isna().sum()}")
    print(f"Ground truth join columns NaN counts: {valid_ground_truth[['userid', 'zip', 'plan_id']].isna().sum()}")
    
    # Align feature_df with ground_truth_df using userid, zip, plan_id
    try:
        merged_df = training_df.merge(
            valid_ground_truth[['userid', 'zip', 'plan_id', 'persona', 'persona_encoded']],
            how='inner',
            on=['userid', 'zip', 'plan_id']
        ).reset_index(drop=True)
        print(f"Rows after aligning with ground truth: {len(merged_df)}")
    except Exception as e:
        print(f"Error during merge with ground truth: {e}")
        # Try to identify problematic records
        print("Sample values from training_df:")
        print(training_df[['userid', 'zip', 'plan_id']].head())
        print("Sample values from valid_ground_truth:")
        print(valid_ground_truth[['userid', 'zip', 'plan_id']].head())
        return None, None, None, None

    if len(merged_df) == 0:
        print("Error: No data remains after aligning with ground truth. Check userid/zip/plan_id compatibility.")
        return None, None, None, None

    # Prepare features and metadata
    metadata_columns = ['userid', 'zip', 'plan_id', 'persona', 'persona_encoded', 'quality_level'] + feature_columns
    available_columns = [col for col in metadata_columns if col in merged_df.columns]
    metadata = merged_df[available_columns]
    print(f"Metadata columns: {metadata.columns.tolist()}")

    X = merged_df[feature_columns].fillna(0)
    y_true = merged_df['persona_encoded']

    print(f"Shape of X: {X.shape}")
    print(f"Unique encoded personas in y_true: {y_true.unique().tolist()}")

    return X, y_true, metadata, le

def evaluate_model(model, X, y_true, metadata, le):
    if X is None or y_true is None or len(X) == 0:
        print("Cannot evaluate model: No valid data provided.")
        return None

    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)

    # Decode predictions and ground truth back to string personas
    personas = le.inverse_transform(model.classes_)
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred_str = le.inverse_transform(y_pred)
    y_true_str = le.inverse_transform(y_true)

    # Combine metadata with predictions
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred_str
    output_df['persona'] = y_true_str  # Update persona to string labels

    # Add probability ranking and confidence score
    output_df['probability_ranking'] = ''
    output_df['confidence_score'] = 0.0
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])
        predicted_persona = output_df.loc[i, 'predicted_persona']
        output_df.loc[i, 'confidence_score'] = probs[predicted_persona]

    # Compute per-record accuracy_rate
    output_df['accuracy_rate'] = (output_df['predicted_persona'] == output_df['persona']).astype(int)

    # Compute overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Compute per-persona metrics
    print("\nPer-Persona Metrics:")
    persona_metrics = []
