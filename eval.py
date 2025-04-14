import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File paths
behavioral_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/processed_us_dce_pro_behavioral_features_092024_032025_v7.csv'
plan_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
model_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/rf_model_persona_092024_032025_v7.pkl'
output_file = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/032025/eval_results_092024_032025_v7.csv'

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        print(f"Behavioral data loaded: {len(behavioral_df)} rows")
        print(f"Behavioral_df columns: {behavioral_df.columns.tolist()}")
        print(f"Plan data loaded: {len(plan_df)} rows")
        print(f"Plan_df columns: {plan_df.columns.tolist()}")
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    new_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row['persona']):
            new_rows.append(row)
            continue
        personas = [p.strip().lower() for p in str(row['persona']).split(',')]
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
            first_persona = personas[0]
            row_copy['persona'] = first_persona
            new_rows.append(row_copy)
    return pd.DataFrame(new_rows)

def assign_quality_level(row, filter_cols, query_cols):
    has_plan_id = pd.notna(row['plan_id'])
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

def prepare_evaluation_features(behavioral_df, plan_df):
    behavioral_df = normalize_persona(behavioral_df)
    print(f"Rows after persona normalization: {len(behavioral_df)}")

    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    )
    print(f"Rows after merge: {len(training_df)}")
    print("Columns after merge:", training_df.columns.tolist())

    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore')

    # Compute or ensure quality_level
    filter_cols = [col for col in training_df.columns if col.startswith('filter_')]
    query_cols = [col for col in training_df.columns if col.startswith('query_')]
    if 'quality_level' not in training_df.columns:
        print("Warning: 'quality_level' not found in training_df. Computing it.")
        training_df['quality_level'] = training_df.apply(
            lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
        )
    else:
        null_quality = training_df['quality_level'].isna().sum()
        if null_quality > 0:
            print(f"Warning: {null_quality} rows with null 'quality_level'. Recomputing.")
            training_df['quality_level'] = training_df.apply(
                lambda row: assign_quality_level(row, filter_cols, query_cols), axis=1
            )

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

    # Ensure all raw_plan_features and csnp_type exist
    for col in raw_plan_features + ['csnp_type']:
        if col not in training_df.columns:
            print(f"Warning: '{col}' not found in training_df. Filling with 0.")
            training_df[col] = 0
        else:
            training_df[col] = training_df[col].fillna(0)

    additional_features = []
    training_df['csnp_interaction'] = training_df['csnp'] * (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('time_csnp_pages', 0).fillna(0) + training_df.get('accordion_csnp', 0).fillna(0)
    ) * 3.0
    additional_features.append('csnp_interaction')

    training_df['csnp_type_flag'] = training_df['csnp_type'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    additional_features.append('csnp_type_flag')

    training_df['csnp_signal_strength'] = (
        training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
        training_df.get('accordion_csnp', 0).fillna(0) + training_df.get('time_csnp_pages', 0).fillna(0)
    ).clip(upper=5) * 3.0
    additional_features.append('csnp_signal_strength')

    training_df['dental_interaction'] = (
        training_df.get('query_dental', 0).fillna(0) + training_df.get('filter_dental', 0).fillna(0)
    ) * training_df['ma_dental_benefit'] * 2.0
    additional_features.append('dental_interaction')

    training_df['vision_interaction'] = (
        training_df.get('query_vision', 0).fillna(0) + training_df.get('filter_vision', 0).fillna(0)
    ) * training_df['ma_vision'] * 2.0
    additional_features.append('vision_interaction')

    training_df['csnp_drug_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0) + 
            training_df.get('time_csnp_pages', 0).fillna(0)
        ) * 2.5 - training_df['ma_drug_coverage'] * (
            training_df.get('query_drug', 0).fillna(0) + training_df.get('filter_drug', 0).fillna(0) + 
            training_df.get('time_drug_pages', 0).fillna(0)
        )
    ).clip(lower=0) * 3.0
    additional_features.append('csnp_drug_interaction')

    training_df['csnp_doctor_interaction'] = (
        training_df['csnp'] * (
            training_df.get('query_csnp', 0).fillna(0) + training_df.get('filter_csnp', 0).fillna(0)
        ) * 2.0 - training_df['ma_provider_network'] * (
            training_df.get('query_provider', 0).fillna(0) + training_df.get('filter_provider', 0).fillna(0)
        )
    ).clip(lower=0) * 2.0
    additional_features.append('csnp_doctor_interaction')

    # Debug: Check csnp features
    high_quality_csnp = training_df[(training_df['quality_level'] == 'High') & (training_df['persona'] == 'csnp')]
    print(f"Eval: High-quality csnp samples: {len(high_quality_csnp)}")
    print(f"Eval: Non-zero csnp_interaction: {sum(high_quality_csnp['csnp_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_drug_interaction: {sum(high_quality_csnp['csnp_drug_interaction'] > 0)}")
    print(f"Eval: Non-zero csnp_doctor_interaction: {sum(high_quality_csnp['csnp_doctor_interaction'] > 0)}")

    feature_columns = all_behavioral_features + raw_plan_features + additional_features

    print(f"Feature columns expected: {feature_columns}")

    missing_features = [col for col in feature_columns if col not in training_df.columns]
    if missing_features:
        print(f"Warning: Missing features in training_df: {missing_features}")
        for col in missing_features:
            training_df[col] = 0

    print(f"Columns in training_df after filling: {training_df.columns.tolist()}")

    # Ensure metadata includes quality_level
    metadata_columns = ['userid', 'zip', 'plan_id', 'persona', 'quality_level'] + feature_columns
    missing_metadata_cols = [col for col in metadata_columns if col not in training_df.columns]
    if missing_metadata_cols:
        print(f"Warning: Missing metadata columns: {missing_metadata_cols}")
        for col in missing_metadata_cols:
            training_df[col] = 0

    metadata = training_df[metadata_columns]

    # Filter out fitness and hearing
    valid_mask = (
        training_df['persona'].notna() & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    training_df = training_df[valid_mask]
    metadata = metadata[valid_mask]
    print(f"Rows after filtering fitness/hearing: {len(training_df)}")

    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona'] if 'persona' in training_df.columns else None

    if y_true is not None:
        print(f"Unique personas before filtering: {y_true.unique().tolist()}")

    return X, y_true, metadata

def evaluate_predictions(model, X, y_true, metadata):
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X)
    
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred
    
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])
    
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    if y_true is not None:
        y_true_lower = y_true.str.lower()
        invalid_personas = ['', 'unknown', 'none', 'healthcare', 'fitness', 'hearing']
        valid_mask = y_true.notna() & (~y_true_lower.isin(invalid_personas))
        if not valid_mask.all():
            excluded_count = (~valid_mask).sum()
            print(f"Warning: Excluding {excluded_count} rows with invalid 'persona' values.")
            print(f"Sample of excluded personas: {y_true[~valid_mask].head().tolist()}")
        
        X_valid = X[valid_mask]
        y_true_valid = y_true[valid_mask].reset_index(drop=True)
        y_pred_valid = y_pred[valid_mask.values]
        metadata_valid = metadata[valid_mask].reset_index(drop=True)

        if len(y_true_valid) > 0:
            accuracy = accuracy_score(y_true_valid, y_pred_valid)
            print(f"\nOverall Accuracy (valid personas only): {accuracy * 100:.2f}%")
            print(f"Rows evaluated: {len(y_true_valid)}")
            print(f"Correct predictions: {sum(y_pred_valid == y_true_valid)}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_true_valid, y_pred_valid))
            
            print("\nIndividual Persona Accuracy Rates (Overall):")
            for persona in sorted(set(y_true_valid)):
                persona_true = y_true_valid == persona
                persona_count = sum(persona_true)
                if persona_count > 0:
                    persona_accuracy = accuracy_score(y_true_valid[persona_true], y_pred_valid[persona_true]) * 100
                    print(f"Accuracy for '{persona}': {persona_accuracy:.2f}% (Count: {persona_count})")

            quality_levels = ['High', 'Medium', 'Low']
            for level in quality_levels:
                level_mask = metadata_valid['quality_level'] == level
                level_true = y_true_valid[level_mask]
                level_pred = y_pred_valid[level_mask.values]
                print(f"\n{level} Quality Data Results:")
                print(f"Rows: {len(level_true)}")
                if len(level_true) > 0:
                    level_accuracy = accuracy_score(level_true, level_pred) * 100
                    print(f"Accuracy: {level_accuracy:.2f}%")
                    print(f"Individual Persona Accuracy Rates ({level}):")
                    for persona in sorted(set(level_true)):
                        persona_true = level_true == persona
                        persona_count = sum(persona_true)
                        if persona_count > 0:
                            persona_accuracy = accuracy_score(level_true[persona_true], level_pred[persona_true]) * 100
                            print(f"Accuracy for '{persona}': {persona_accuracy:.2f}% (Count: {persona_count})")
                else:
                    print("No records in this quality level.")

            print("\nConfusion Matrix (Overall, valid personas only):")
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(sorted(set(y_true_valid))))
            print(pd.DataFrame(cm, index=list(sorted(set(y_true_valid))), columns=list(sorted(set(y_true_valid)))))
        else:
            print("ERROR: No valid persona values remain for evaluation.")

def main():
    print("Evaluating data with pre-trained Random Forest model...")
    rf_model = load_model(model_file)
    behavioral_df, plan_df = load_data(behavioral_file, plan_file)
    X, y_true, metadata = prepare_evaluation_features(behavioral_df, plan_df)
    evaluate_predictions(rf_model, X, y_true, metadata)

if __name__ == "__main__":
    main()
