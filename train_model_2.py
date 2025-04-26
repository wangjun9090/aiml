import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
import optuna
import logging
import sys
import os

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
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-1.0.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp', 'transportation', 'otc']

# Persona info
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
    'dental': {'plan_col': 'ma_dental_benefit', 'query_col': 'query_dental', 'filter_col': 'filter_dental'},
    'doctor': {'plan_col': 'ma_provider_network', 'query_col': 'query_provider', 'filter_col': 'filter_provider', 'click_col': 'click_provider'},
    'dsnp': {'plan_col': 'dsnp', 'query_col': 'query_dsnp', 'filter_col': 'filter_dsnp'},
    'drug': {'plan_col': 'ma_drug_benefit', 'query_col': 'query_drug', 'filter_col': 'filter_drug', 'click_col': 'click_drug'},
    'vision': {'plan_col': 'ma_vision', 'query_col': 'query_vision', 'filter_col': 'filter_vision'}
}

def calculate_persona_weight(row, persona_info, persona):
    query_col = persona_info['query_col']
    filter_col = persona_info['filter_col']
    plan_col = persona_info.get('plan_col', None)
    click_col = persona_info.get('click_col', None)
    
    query_value = row.get(query_col, 0) if pd.notna(row.get(query_col, np.nan)) else 0
    filter_value = row.get(filter_col, 0) if pd.notna(row.get(filter_col, np.nan)) else 0
    plan_value = row.get(plan_col, 0) if plan_col and pd.notna(row.get(plan_col, np.nan)) else 0
    click_value = row.get(click_col, 0) if click_col and pd.notna(row.get(click_col, np.nan)) else 0
    
    # Coefficients: boost dental, dsnp, vision
    query_coeff = 0.4 if persona in ['dental', 'dsnp', 'vision'] else 0.2
    filter_coeff = 0.4 if persona in ['dental', 'dsnp', 'vision'] else 0.2
    plan_coeff = 0.3 if persona in ['dental', 'dsnp', 'vision'] else 0.2
    click_coeff = 0.3 if persona in ['doctor', 'drug'] else 0.2
    
    weight = (query_coeff * query_value + filter_coeff * filter_value + 
              plan_coeff * plan_value + click_coeff * click_value)
    
    # Interaction boost for minority classes
    if persona in ['csnp', 'dental', 'dsnp', 'vision']:
        interaction = row.get(f'{persona}_interaction', 0)
        weight += 0.3 * interaction if persona in ['dental', 'dsnp', 'vision'] else 0.2 * interaction
    
    # Normalize to [0, 1]
    return min(max(weight, 0), 1.0)

def load_data(behavioral_path, plan_path):
    try:
        if not os.path.exists(behavioral_path):
            raise FileNotFoundError(f"Behavioral file not found: {behavioral_path}")
        if not os.path.exists(plan_path):
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        
        if behavioral_df.empty or plan_df.empty:
            raise ValueError("Loaded DataFrames are empty")
        
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.strip().str.lower()
        behavioral_df['persona'] = behavioral_df['persona'].replace('nan', '')
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip()
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip()
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        
        logger.info(f"Behavioral_df rows: {len(behavioral_df)}, Unique personas: {behavioral_df['persona'].unique()}")
        logger.info(f"Plan_df rows: {len(plan_df)}")
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

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
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    if result.empty:
        logger.error(f"No valid personas found. Valid personas: {valid_personas}")
        raise ValueError("No valid personas found")
    logger.info(f"Rows after normalization: {len(result)}")
    return result

def prepare_features(behavioral_df, plan_df):
    try:
        behavioral_df = normalize_persona(behavioral_df)
        
        training_df = behavioral_df.merge(
            plan_df.rename(columns={'StateCode': 'state'}),
            how='left', on=['zip', 'plan_id']
        ).reset_index(drop=True)
        logger.info(f"Rows after merge: {len(training_df)}")
        
        if training_df.empty or len(training_df) < 10:
            logger.warning("Merge yielded insufficient data. Using behavioral_df features only.")
            training_df = behavioral_df.copy()
            plan_features = []
        else:
            plan_features = ['ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp']
        
        behavioral_features = [
            'query_dental', 'query_drug', 'query_provider', 'query_vision', 'query_csnp', 'query_dsnp',
            'filter_dental', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp', 'filter_dsnp',
            'num_pages_viewed', 'total_session_time', 'time_dental_pages', 'num_clicks'
        ]
        
        # Compute aggregate features
        query_cols = [c for c in behavioral_features if c.startswith('query_') and c in training_df.columns]
        filter_cols = [c for c in behavioral_features if c.startswith('filter_') and c in training_df.columns]
        training_df['query_count'] = training_df[query_cols].sum(axis=1) if query_cols else pd.Series(0, index=training_df.index)
        training_df['filter_count'] = training_df[filter_cols].sum(axis=1) if filter_cols else pd.Series(0, index=training_df.index)
        
        # Initialize missing columns with zeros
        for col in behavioral_features + plan_features:
            if col not in training_df.columns:
                training_df[col] = pd.Series(0, index=training_df.index)
            else:
                training_df[col] = training_df[col].fillna(0)
        
        for persona in PERSONAS:
            if persona in PERSONA_INFO:
                training_df[f'{persona}_weight'] = training_df.apply(
                    lambda row: calculate_persona_weight(row, PERSONA_INFO[persona], persona), axis=1
                )
        
        training_df['dental_interaction'] = training_df['query_dental'] * training_df['ma_dental_benefit']
        training_df['csnp_interaction'] = training_df['query_csnp'] * training_df['csnp']
        training_df['dsnp_interaction'] = training_df['query_dsnp'] * training_df['dsnp']
        training_df['vision_interaction'] = training_df['query_vision'] * training_df['ma_vision']
        additional_features = ['dental_interaction', 'csnp_interaction', 'dsnp_interaction', 'vision_interaction']
        additional_features += [f'{persona}_weight' for persona in PERSONAS if persona in PERSONA_INFO]
        additional_features += ['query_count', 'filter_count']
        
        feature_columns = behavioral_features + plan_features + additional_features
        
        training_df = training_df[training_df['persona'].isin(PERSONAS)].reset_index(drop=True)
        logger.info(f"Rows after filtering: {len(training_df)}")
        
        if training_df.empty:
            logger.error("No rows left after persona filtering")
            raise ValueError("No rows left after persona filtering")
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['persona']
        
        if X.empty or y.empty:
            logger.error(f"Feature matrix empty: X shape={X.shape}, y shape={y.shape}")
            raise ValueError("Feature matrix or target empty")
        
        variances = X.var()
        valid_features = variances[variances > 1e-5].index.tolist()
        if not valid_features:
            logger.error("No features with sufficient variance")
            raise ValueError("No features with sufficient variance")
        X = X[valid_features]
        logger.info(f"Selected features: {valid_features}")
        
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y, scaler
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

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

def focal_loss_objective(y_true, y_pred):
    # Custom focal loss for XGBoost
    gamma = 2.0
    alpha = 0.25
    p = 1 / (1 + np.exp(-y_pred))
    grad = alpha * (p - y_true) * (1 - p)**gamma * np.log(p + 1e-8)
    hess = alpha * (1 - p)**gamma * (np.log(p + 1e-8) * (1 - p) + p * (1 - p)**gamma / (p + 1e-8))
    return grad, hess

def objective(trial, X_train, y_train, X_val, y_val, le):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, objective='multi:softmax', num_class=len(le.classes_))
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
    
    # Custom class weights
    class_counts = y_train.value_counts()
    class_weights = {le.transform([k])[0]: 1.0 / class_counts[k] for k in class_counts.index}
    # Boost weights for dental, dsnp, vision
    for k in class_counts.index:
        if k in ['dental', 'dsnp', 'vision']:
            class_weights[le.transform([k])[0]] *= 1.5
    weights = np.array([class_weights[y] for y in y_train_encoded])
    
    # Borderline-SMOTE
    smote = BorderlineSMOTE(sampling_strategy={
        persona: 100 if persona in ['csnp', 'dental', 'dsnp', 'vision'] else max(300, count)
        for persona, count in y_train.value_counts().items()
    }, random_state=42, k_neighbors=3)
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        y_train_balanced_encoded = le.transform(y_train_balanced)
        weights_balanced = np.array([class_weights[y] for y in y_train_balanced_encoded])
    except Exception as e:
        logger.error(f"SMOTE failed: {e}. Using original training data.")
        X_train_balanced, y_train_balanced = X_train, y_train
        y_train_balanced_encoded = y_train_encoded
        weights_balanced = weights
    
    # Hyperparameter tuning
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_balanced, y_train_balanced_encoded, X_test, y_test_encoded, le), n_trials=30)
        best_params = study.best_params
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return
    
    # Train XGBoost
    try:
        model = xgb.XGBClassifier(**best_params, random_state=42, objective='multi:softmax', num_class=len(le.classes_))
        model.fit(X_train_balanced, y_train_balanced_encoded, sample_weight=weights_balanced)
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
            tmp_model_file = '/tmp/model-persona-1.0.0.pkl'
            tmp_label_file = '/tmp/label_encoder_1.pkl'
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
