import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score

# Hardcoded file paths for Databricks
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/normalized_us_dce_pro_behavioral_features_0401_2025_0420_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-0.0.3.pkl'  # Updated model
OUTPUT_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_results_0401_2025_0420_2025.csv'
SUMMARY_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/eval/042025/eval_summary_0401_2025_0420_2025.csv'

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
        print(f"Plan data loaded: {len(plan_df)} rows")
        print(f"Behavioral_df columns: {behavioral_df.columns.tolist()}")
        print(f"Plan_df columns: {plan_df.columns.tolist()}")
        return behavioral_df, plan_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    """Normalize personas for evaluation (split multiple personas into separate rows)."""
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
    return pd.DataFrame(new_rows).reset_index(drop=True)

def prepare_evaluation_features(behavioral_df, plan_df, model):
    # Normalize personas for evaluation (to prepare y_true)
    behavioral_df = normalize_persona(behavioral_df)
    print(f"Rows after persona normalization: {len(behavioral_df)}")

    # Ensure 'zip' and 'plan_id' columns have the same data type (string)
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')

    # Merge behavioral and plan data
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left',
        on=['zip', 'plan_id'],
        suffixes=('_beh', '_plan')
    ).reset_index(drop=True)
    print(f"Rows after merge: {len(training_df)}")
    print("Columns after merge:", training_df.columns.tolist())

    training_df['state'] = training_df['state_beh'].fillna(training_df['state_plan'])
    training_df = training_df.drop(columns=['state_beh', 'state_plan'], errors='ignore').reset_index(drop=True)

    # Define feature lists
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

    # Compute quality level
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

    # Compute additional features
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

    # New persona-specific signal features
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

    # Define persona weights for weighted features
    persona_weights = {
        'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'pro_click_count', 'interaction_col': None},
        'drug': {'plan_col': 'ma_drug_coverage', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'dce_click_count', 'interaction_col': None},
        'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision', 'interaction_col': 'vision_interaction'},
        'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental', 'interaction_col': 'dental_interaction'},
        'otc': {'plan_col': 'ma_otc', 'query_col': 'query_otc', 'filter_col': 'filter_otc', 'interaction_col': None},
        'transportation': {'plan_col': 'ma_transportation', 'query_col': 'query_transportation', 'filter_col': 'filter_transportation', 'interaction_col': None},
        'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp', 'interaction_col': 'csnp_interaction'},
        'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp', 'interaction_col': None}
    }

    # Constants
    k1, k3, k4, k7, k8 = 0.15, 0.8, 0.7, 0.3, 0.4
    k9, k10 = 2.5, 2.3
    W_CSNP_BASE, W_CSNP_HIGH, W_DSNP_BASE, W_DSNP_HIGH = 3.0, 7.0, 1.2, 1.8

    def calculate_persona_weight(row, persona_info, persona, plan_df):
        plan_col = persona_info['plan_col']
        query_col = persona_info['query_col']
        filter_col = persona_info['filter_col']
        click_col = persona_info.get('click_col', None)
        interaction_col = persona_info.get('interaction_col', None)
        
        # Compute base weight from plan features
        if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
            base_weight = min(row[plan_col], 0.7 if persona == 'csnp' else 0.5)
            if persona == 'csnp' and row.get('csnp_type', 'N') == 'Y':
                base_weight *= W_CSNP_HIGH
            elif persona == 'csnp':
                base_weight *= W_CSNP_BASE
            elif persona == 'dsnp' and row.get('dsnp_type', 'N') == 'Y':
                base_weight *= W_DSNP_HIGH
            elif persona == 'dsnp':
                base_weight *= W_DSNP_BASE
        elif pd.notna(row.get('compared_plan_ids')) and isinstance(row['compared_plan_ids'], str) and row.get('num_plans_compared', 0) > 0:
            compared_ids = row['compared_plan_ids'].split(',')
            compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
            if not compared_plans.empty and plan_col in compared_plans.columns:
                base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
                if persona == 'csnp' and 'csnp_type' in compared_plans.columns:
                    csnp_type_y_ratio = (compared_plans['csnp_type'] == 'Y').mean()
                    base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
                elif persona == 'dsnp' and 'dsnp_type' in compared_plans.columns:
                    dsnp_type_y_ratio = (compared_plans['dsnp_type'] == 'Y').mean()
                    base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)
            else:
                base_weight = 0
        else:
            base_weight = 0
        
        # Compute generic behavioral score
        pages_viewed = min(row.get('num_pages_viewed', 0), 3) if pd.notna(row.get('num_pages_viewed')) else 0
        query_value = row.get(query_col, 0) if pd.notna(row.get(query_col)) else 0
        filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col)) else 0
        click_value = row.get(click_col, 0) if click_col and click_col in row and pd.notna(row.get(click_col)) else 0
        interaction_value = row.get(interaction_col, 0) if interaction_col and pd.notna(row.get(interaction_col)) else 0
        
        query_coeff = k9 if persona == 'csnp' else k3
        filter_coeff = k10 if persona == 'csnp' else k4
        click_coefficient = k8 if click_col == 'pro_click_count' else k7 if click_col == 'dce_click_count' else 0
        interaction_coeff = 1.5 if interaction_col in ['vision_interaction', 'dental_interaction', 'csnp_interaction'] else 1.0
        
        behavioral_score = (
            query_coeff * query_value +
            filter_coeff * filter_value +
            k1 * pages_viewed +
            click_coefficient * click_value +
            interaction_coeff * interaction_value
        )
        
        # Generic behavioral adjustments
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
        
        # Quality-level adjustment
        if row['quality_level'] == 'High':
            behavioral_score += 0.5
        elif row['quality_level'] == 'Medium':
            behavioral_score += 0.2
        
        adjusted_weight = base_weight + behavioral_score
        return min(adjusted_weight, 3.5 if persona == 'csnp' else 1.2)

    # Compute weighted features
    print("Calculating persona weights...")
    for persona, info in persona_weights.items():
        click_col = info.get('click_col', 'click_dummy')
        if click_col not in training_df.columns:
            training_df[click_col] = 0
        training_df[f'w_{persona}'] = training_df.apply(
            lambda row: calculate_persona_weight(row, info, persona, plan_df), axis=1
        )

    # Normalize weighted features
    weighted_features = [f'w_{persona}' for persona in persona_weights.keys()]
    weight_sum = training_df[weighted_features].sum(axis=1)
    for wf in weighted_features:
        training_df[wf] = training_df[wf] / weight_sum.where(weight_sum > 0, 1)
        training_df[wf] = training_df[wf].clip(upper=1.0)

    print(f"Weighted features added: {weighted_features}")
    print("Sample weights:")
    print(training_df[weighted_features].head())

    # Define all feature columns expected by the model
    all_weighted_features = [f'w_{persona}' for persona in [
        'doctor', 'drug', 'vision', 'dental', 'otc', 'transportation', 'csnp', 'dsnp'
    ]]
    feature_columns = all_behavioral_features + raw_plan_features + additional_features + all_weighted_features

    print(f"Feature columns expected: {feature_columns}")

    # Check for missing features and fill with 0
    missing_features = [col for col in feature_columns if col not in training_df.columns]
    if missing_features:
        print(f"Warning: Missing features in training_df: {missing_features}")
        for col in missing_features:
            training_df[col] = 0

    print(f"Columns in training_df after filling: {training_df.columns.tolist()}")

    # Filter valid personas
    training_df = training_df.reset_index(drop=True)
    valid_mask = (
        training_df['persona'].notna() & 
        (~training_df['persona'].str.lower().isin(['unknown', 'none', 'healthcare', 'fitness', 'hearing']))
    )
    print(f"Number of valid rows before filtering: {len(training_df)}")
    print(f"Number of True values in valid_mask: {valid_mask.sum()}")
    training_df = training_df.loc[valid_mask].reset_index(drop=True)
    print(f"Rows after filtering invalid personas: {len(training_df)}")

    # Filter to include only personas in model.classes_
    model_classes = set(model.classes_)
    valid_persona_mask = training_df['persona'].isin(model_classes)
    training_df = training_df.loc[valid_persona_mask].reset_index(drop=True)
    print(f"Rows after filtering for model classes: {len(training_df)}")

    # Prepare features and metadata
    metadata_columns = ['userid', 'zip', 'plan_id', 'persona', 'quality_level'] + feature_columns
    available_columns = [col for col in metadata_columns if col in training_df.columns]
    metadata = training_df[available_columns]
    print(f"Metadata columns: {metadata.columns.tolist()}")

    X = training_df[feature_columns].fillna(0)
    y_true = training_df['persona']

    print(f"Unique personas after filtering: {y_true.unique().tolist()}")

    return X, y_true, metadata

def evaluate_model(model, X, y_true, metadata):
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X)
    personas = model.classes_
    proba_df = pd.DataFrame(y_pred_proba, columns=[f'prob_{p}' for p in personas])
    y_pred = model.predict(X)

    # Combine metadata with predictions
    output_df = pd.concat([metadata.reset_index(drop=True), proba_df], axis=1)
    output_df['predicted_persona'] = y_pred

    # Add probability ranking and confidence score
    output_df['probability_ranking'] = ''
    output_df['confidence_score'] = 0.0
    for i in range(len(output_df)):
        probs = {persona: output_df.loc[i, f'prob_{persona}'] for persona in personas}
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        output_df.loc[i, 'probability_ranking'] = '; '.join([f"{p}: {prob:.4f}" for p, prob in ranked])
        predicted_persona = output_df.loc[i, 'predicted_persona']
        output_df.loc[i, 'confidence_score'] = probs[predicted_persona]

    # Compute per-record accuracy_rate (binary)
    output_df['accuracy_rate'] = (output_df['predicted_persona'] == output_df['persona']).astype(int)

    # Compute overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Compute per-persona metrics
    print("\nPer-Persona Metrics:")
    persona_metrics = []
    for persona in sorted(personas):
        mask = output_df['persona'] == persona
        count = mask.sum()
        matches = output_df['accuracy_rate'][mask].sum()
        accuracy = accuracy_score(output_df['persona'][mask], output_df['predicted_persona'][mask]) if count > 0 else 0.0
        avg_confidence = output_df['confidence_score'][mask].mean() if count > 0 else 0.0
        low_conf_correct = ((output_df['accuracy_rate'] == 1) & (output_df['confidence_score'] < 0.7) & mask).sum()
        very_low_conf_correct = ((output_df['accuracy_rate'] == 1) & (output_df['confidence_score'] < 0.5) & mask).sum()
        persona_metrics.append({
            'persona': persona,
            'total_records': count,
            'matches': matches,
            'accuracy_rate': round(accuracy, 2),
            'avg_confidence': round(avg_confidence, 2)
        })
        print(f"Persona '{persona}':")
        print(f"  Total Records: {count}")
        print(f"  Matches: {matches}")
        print(f"  Accuracy Rate: {accuracy * 100:.2f}%")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Correct Predictions with Low Confidence (< 0.7): {low_conf_correct} ({low_conf_correct/matches*100:.2f}% of matches)" if matches > 0 else "  Correct Predictions with Low Confidence (< 0.7): N/A")
        print(f"  Correct Predictions with Very Low Confidence (< 0.5): {very_low_conf_correct} ({very_low_conf_correct/matches*100:.2f}% of matches)" if matches > 0 else "  Correct Predictions with Very Low Confidence (< 0.5): N/A")
        # Diagnostics for vision, csnp, dental
        if persona in ['vision', 'csnp', 'dental']:
            print(f"\nDetailed {persona.capitalize()} Records:")
            cols = ['userid', 'persona', 'predicted_persona', 'confidence_score', 'accuracy_rate', 'probability_ranking', f'w_{persona}', f'query_{persona}', f'filter_{persona}']
            if persona == 'vision':
                cols.append('vision_interaction')
                cols.append('vision_signal')
            elif persona == 'dental':
                cols.append('dental_interaction')
                cols.append('dental_signal')
            elif persona == 'csnp':
                cols.append('csnp_interaction')
                cols.append('csnp_specific_signal')
            persona_df = output_df[mask][cols]
            print(persona_df.to_string(index=False))

    # Overall metrics
    overall_metrics = {
        'persona': 'Overall',
        'total_records': len(output_df),
        'matches': output_df['accuracy_rate'].sum(),
        'accuracy_rate': round(overall_accuracy, 2),
        'avg_confidence': round(output_df['confidence_score'].mean(), 2)
    }
    low_conf_correct_overall = ((output_df['accuracy_rate'] == 1) & (output_df['confidence_score'] < 0.7)).sum()
    very_low_conf_correct_overall = ((output_df['accuracy_rate'] == 1) & (output_df['confidence_score'] < 0.5)).sum()
    print("\nOverall Metrics:")
    print(f"  Total Records: {overall_metrics['total_records']}")
    print(f"  Matches: {overall_metrics['matches']}")
    print(f"  Accuracy Rate: {overall_metrics['accuracy_rate'] * 100:.2f}%")
    print(f"  Average Confidence: {overall_metrics['avg_confidence']:.2f}")
    print(f"  Correct Predictions with Low Confidence (< 0.7): {low_conf_correct_overall} ({low_conf_correct_overall/overall_metrics['matches']*100:.2f}% of matches)")
    print(f"  Correct Predictions with Very Low Confidence (< 0.5): {very_low_conf_correct_overall} ({very_low_conf_correct_overall/overall_metrics['matches']*100:.2f}% of matches)")

    # Confidence analysis by correctness
    print("\nConfidence Analysis by Correctness:")
    for persona in sorted(personas):
        mask = output_df['persona'] == persona
        correct_mask = mask & (output_df['accuracy_rate'] == 1)
        incorrect_mask = mask & (output_df['accuracy_rate'] == 0)
        correct_conf = output_df['confidence_score'][correct_mask].mean() if correct_mask.sum() > 0 else 0.0
        incorrect_conf = output_df['confidence_score'][incorrect_mask].mean() if incorrect_mask.sum() > 0 else 0.0
        print(f"Persona '{persona}':")
        print(f"  Avg Confidence (Correct): {correct_conf:.2f} (Count: {correct_mask.sum()})")
        print(f"  Avg Confidence (Incorrect): {incorrect_conf:.2f} (Count: {incorrect_mask.sum()})")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))

    # Misclassification analysis
    print("\nMisclassification Analysis:")
    misclassified = output_df[output_df['accuracy_rate'] == 0]
    for persona in ['vision', 'csnp', 'dental']:
        persona_mis = misclassified[misclassified['persona'] == persona]
        print(f"\nMisclassified {persona.capitalize()} Records:")
        cols = ['userid', 'persona', 'predicted_persona', 'confidence_score', f'w_{persona}', f'query_{persona}', f'filter_{persona}']
        if persona == 'vision':
            cols.append('vision_interaction')
            cols.append('vision_signal')
        elif persona == 'dental':
            cols.append('dental_interaction')
            cols.append('dental_signal')
        elif persona == 'csnp':
            cols.append('csnp_interaction')
            cols.append('csnp_specific_signal')
        print(persona_mis[cols].head().to_string(index=False))

    # Top-2 accuracy
    top_2_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=2, labels=personas)
    print(f"\nTop-2 Accuracy: {top_2_accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=personas)
    cm_df = pd.DataFrame(cm, index=personas, columns=personas)
    print("\nConfusion Matrix:")
    print(cm_df)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=personas, target_names=personas))

    # Create summary DataFrame
    summary_data = persona_metrics + [overall_metrics]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary evaluation results saved to {SUMMARY_FILE}")

    # Add overall metrics to output_df
    output_df['overall_accuracy'] = overall_accuracy
    output_df['top_2_accuracy'] = top_2_accuracy

    # Save detailed results
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed evaluation results saved to {OUTPUT_FILE}")

    return output_df

def main():
    print("Evaluating Random Forest model...")
    model = load_model(MODEL_FILE)
    behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    X, y_true, metadata = prepare_evaluation_features(behavioral_df, plan_df, model)
    if y_true is not None:
        evaluate_model(model, X, y_true, metadata)
    else:
        print("No ground truth labels available for evaluation.")

if __name__ == "__main__":
    main()
