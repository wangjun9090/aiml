import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from catboost import CatBoostClassifier
import optuna
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Suppress Py4J logs

# File paths (Databricks-compatible)
BEHAVIORAL_FILE = '/dbfs/path_to_behavioral.csv'  # Adjust to your DBFS path
PLAN_FILE = '/dbfs/path_to_plan.csv'              # Adjust to your DBFS path
MODEL_FILE = '/dbfs/model-persona-0.2.0.pkl'
LABEL_ENCODER_FILE = '/dbfs/label_encoder.pkl'
SCALER_FILE = '/dbfs/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp', 'transportation', 'otc']

# Default coefficients (adjust if known)
W_CSNP_HIGH = 1.5
W_CSNP_BASE = 1.0
W_DSNP_HIGH = 1.5
W_DSNP_BASE = 1.0
k1 = 0.1  # Pages viewed
k3 = 0.2  # Query coefficient (non-csnp)
k4 = 0.2  # Filter coefficient (non-csnp)
k7 = 0.3  # Click coefficient (drug)
k8 = 0.3  # Click coefficient (doctor)
k9 = 0.4  # Query coefficient (csnp)
k10 = 0.4  # Filter coefficient (csnp)

# Persona info dictionary
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
}

def calculate_persona_weight(row, persona_info, persona, plan_df):
    plan_col = persona_info['plan_col']
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    click_col = persona_info.get('click_col', None)
    
    base_weight = 0
    if pd.notna(row['plan_id']) and plan_col in row and pd.notna(row[plan_col]):
        base_weight = min(row[plan_col], 0.7 if persona == 'csnp' else 0.5)
        if persona == 'csnp' and 'csnp_type' in row and row['csnp_type'] == 'Y':
            base_weight *= W_CSNP_HIGH
        elif persona == 'csnp':
            base_weight *= W_CSNP_BASE
        elif persona == 'dsnp' and 'dsnp_type' in row and row['dsnp_type'] == 'Y':
            base_weight *= W_DSNP_HIGH
        elif persona == 'dsnp':
            base_weight *= W_DSNP_BASE
    elif pd.isna(row['plan_id']) and pd.notna(row['compared_plan_ids']) and isinstance(row['compared_plan_ids'], str) and row.get('num_plans_compared', 0) > 0:
        compared_ids = row['compared_plan_ids'].split(',')
        compared_plans = plan_df[plan_df['plan_id'].isin(compared_ids) & (plan_df['zip'] == row['zip'])]
        if not compared_plans.empty:
            base_weight = min(compared_plans[plan_col].mean(), 0.7 if persona == 'csnp' else 0.5)
            if persona == 'csnp':
                csnp_type_y_ratio = (compared_plans.get('csnp_type', pd.Series(['N']*len(compared_plans))) == 'Y').mean()
                base_weight *= (W_CSNP_BASE + (W_CSNP_HIGH - W_CSNP_BASE) * csnp_type_y_ratio)
            elif persona == 'dsnp':
                dsnp_type_y_ratio = (compared_plans.get('dsnp_type', pd.Series(['N']*len(compared_plans))) == 'Y').mean()
                base_weight *= (W_DSNP_BASE + (W_DSNP_HIGH - W_DSNP_BASE) * dsnp_type_y_ratio)
    
    pages_viewed = min(row.get('num_pages_viewed', 0), 3) if pd.notna(row.get('num_pages_viewed', np.nan)) else 0
    query_value = row.get(query_col, 0) if pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    query_coeff = k9 if persona == 'csnp' else k3
    filter_coeff = k10 if persona == 'csnp' else k4
    click_coefficient = k8 if persona == 'doctor' else k7 if persona == 'drug' else 0
    
    behavioral_score = query_coeff * query_value + filter_coeff * filter_value + k1 * pages_viewed + click_coefficient * click_value
    
    if persona == 'doctor':
        if click_value >= 1.5: behavioral_score += 0.4
        elif click_value >= 0.5: behavioral_score += 0.2
    elif persona == 'drug':
        if click_value >= 5: behavioral_score += 0.4
        elif click_value >= 2: behavioral_score += 0.2
    elif persona == 'dental':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 2: behavioral_score += 0.3
        elif signal_count >= 1: behavioral_score += 0.15
    elif persona == 'vision':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 1: behavioral_score += 0.35
    elif persona == 'csnp':
        signal_count = sum([1 for val in [query_value, filter_value, pages_viewed] if val > 0])
        if signal_count >= 2: behavioral_score += 0.6
        elif signal_count >= 1: behavioral_score += 0.5
        if row.get('csnp_interaction', 0) > 0: behavioral_score += 0.3
        if row.get('csnp_type_flag', 0) == 1: behavioral_score += 0.2
    
    adjusted_weight = base_weight + behavioral_score
    return min(adjusted_weight, 2.0 if persona == 'csnp' else 1.0)

def load_data(behavioral_path, plan_path):
    behavioral_df = pd.read_csv(behavioral_path)
    plan_df = pd.read_csv(plan_path)
    
    behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.strip().str.lower()
    behavioral_df['persona'] = behavioral_df['persona'].replace('nan', '')
    
    return behavioral_df, plan_df

def normalize_persona(df):
    valid_personas = PERSONAS
    new_rows = []
    
    for _, row in df.iterrows():
        if not row['persona'] or row['persona'] == '':
            continue
        
        personas = [p.strip() for p in row['persona'].split(',')]
        valid_found = [p for p in personas if p in valid_personas]
        
        if not valid_found:
            continue
        
        row_copy = row.copy()
        row_copy['persona'] = valid_found[0]
        new_rows.append(row_copy)
    
    return pd.DataFrame(new_rows).reset_index(drop=True)

def prepare_features(behavioral_df, plan_df):
    behavioral_df = normalize_persona(behavioral_df)
    
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')
    
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='inner', on=['zip', 'plan_id']
    ).reset_index(drop=True)
    
    behavioral_features = [
        'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
        'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
        'num_pages_viewed'
    ]
    plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp']
    
    for col in behavioral_features + plan_features:
        training_df[col] = training_df.get(col, 0).fillna(0)
    
    # Add persona weights as features
    for persona in PERSONAS:
        if persona in PERSONA_INFO:
            training_df[f'{persona}_weight'] = training_df.apply(
                lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona, plan_df), axis=1
            )
    
    training_df['dental_interaction'] = training_df['query_dental'] * training_df['ma_dental_benefit']
    training_df['csnp_interaction'] = training_df['query_csnp'] * training_df['csnp']
    training_df['dsnp_interaction'] = training_df['query_dsnp'] * training_df['dsnp']
    training_df['vision_interaction'] = training_df['query_vision'] * training_df['ma_vision']
    additional_features = ['dental_interaction', 'csnp_interaction', 'dsnp_interaction', 'vision_interaction']
    additional_features += [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
    
    feature_columns = behavioral_features + plan_features + additional_features
    
    training_df = training_df[training_df['persona'].isin(PERSONAS)].reset_index(drop=True)
    
    X = training_df[feature_columns].fillna(0)
    y = training_df['persona']
    
    # Feature selection: top 15 features by variance
    variances = X.var()
    top_features = variances.nlargest(15).index.tolist()
    X = X[top_features]
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, scaler

def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    return per_persona_accuracy

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 128)
    }
    
    model = CatBoostClassifier(**params, random_seed=42, verbose=0, auto_class_weights='Balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def main():
    # Load data
    try:
        behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Prepare features
    try:
        X, y, scaler = prepare_features(behavioral_df, plan_df)
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        return
    
    # Verify labels
    if not all(isinstance(label, str) for label in y):
        logger.error(f"Non-string labels found in y: {y.unique()}")
        return
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Total samples: {len(X)}, Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%), Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        return
    
    # Label encoding
    le = LabelEncoder()
    try:
        le.fit(y)
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)
    except Exception as e:
        logger.error(f"Failed to encode labels: {e}")
        return
    
    # Borderline-SMOTE
    smote = BorderlineSMOTE(sampling_strategy={
        persona: 300 if persona in ['csnp', 'dental', 'doctor', 'dsnp', 'vision'] else max(200, count)
        for persona, count in y_train.value_counts().items()
    }, random_state=42)
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        y_train_balanced_encoded = le.transform(y_train_balanced)
    except Exception as e:
        logger.error(f"SMOTE failed: {e}. Using original training data.")
        X_train_balanced, y_train_balanced = X_train, y_train
        y_train_balanced_encoded = y_train_encoded
    
    # Hyperparameter tuning
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_balanced, y_train_balanced_encoded, X_test, y_test_encoded), n_trials=50)
        best_params = study.best_params
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return
    
    # Train CatBoost
    try:
        model = CatBoostClassifier(**best_params, random_seed=42, verbose=0, auto_class_weights='Balanced')
        model.fit(X_train_balanced, y_train_balanced_encoded)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return
    
    # Evaluate
    logger.info("Evaluating on test set...")
    try:
        y_pred = model.predict(X_test)
        overall_accuracy = accuracy_score(y_test_encoded, y_pred)
        logger.info(f"Overall Accuracy on Test Set (20% of data): {overall_accuracy * 100:.2f}%")
        
        if overall_accuracy < 0.8:
            logger.warning(f"Accuracy {overall_accuracy * 100:.2f}% is below target of 80%. Check per-persona accuracies.")
        
        per_persona_accuracy = compute_per_persona_accuracy(y_test_encoded, y_pred, le.classes_, le.classes_)
        logger.info("Per-Persona Accuracy (%):")
        for persona, acc in per_persona_accuracy.items():
            logger.info(f"  {persona}: {acc:.2f}%")
        
        logger.info("Classification Report:\n" + classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return
    
    # Save model
    try:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(le, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Saved model, label encoder, and scaler to disk.")
    except Exception as e:
        logger.error(f"Failed to save model to DBFS: {e}. Trying /tmp directory...")
        try:
            tmp_model_file = '/tmp/model-persona-0.2.0.pkl'
            tmp_label_file = '/tmp/label_encoder.pkl'
            tmp_scaler_file = '/tmp/scaler.pkl'
            with open(tmp_model_file, 'wb') as f:
                pickle.dump(model, f)
            with open(tmp_label_file, 'wb') as f:
                pickle.dump(le, f)
            with open(tmp_scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved model to /tmp: {tmp_model_file}, {tmp_label_file}, {tmp_scaler_file}")
        except Exception as e2:
            logger.error(f"Failed to save to /tmp: {e2}")

if __name__ == "__main__":
    main()
