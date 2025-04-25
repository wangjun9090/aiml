import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
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

# Hardcoded file paths (replace with your paths)
BEHAVIORAL_FILE = 'path_to_behavioral.csv'
PLAN_FILE = 'path_to_plan.csv'
MODEL_FILE = 'model-persona-0.2.0.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

# Persona coefficients for weighting
PERSONA_COEFFICIENTS = {
    'dental': 3.5, 'doctor': 3.0, 'dsnp': 2.5, 'drug': 1.0,
    'vision': 1.8, 'csnp': 2.0, 'transportation': 1.0, 'otc': 1.0
}

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.strip().str.lower()
        behavioral_df['persona'] = behavioral_df['persona'].replace('nan', '')
        
        logger.info(f"Behavioral data: {len(behavioral_df)} rows, Plan data: {len(plan_df)} rows")
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    """Normalize personas, ensuring all are valid strings."""
    valid_personas = list(PERSONA_COEFFICIENTS.keys())
    original_len = len(df)
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
    skipped = original_len - len(result)
    if skipped > 0:
        logger.info(f"Skipped {skipped} rows due to empty or invalid personas")
    logger.info(f"Normalized to {len(result)} rows")
    
    if not all(isinstance(p, str) for p in result['persona']):
        logger.error(f"Non-string personas found: {result['persona'].unique()}")
        raise ValueError("Persona column contains non-string values after normalization")
    
    return result

def prepare_features(behavioral_df, plan_df):
    behavioral_df = normalize_persona(behavioral_df)
    
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')
    
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left', on=['zip', 'plan_id']
    ).reset_index(drop=True)
    logger.info(f"Rows after merge: {len(training_df)}")
    
    behavioral_features = [
        'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider',
        'query_vision', 'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation',
        'filter_otc', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp',
        'filter_dsnp', 'total_session_time', 'num_pages_viewed'
    ]
    plan_features = ['ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp']
    
    for col in behavioral_features + plan_features:
        training_df[col] = training_df.get(col, 0).fillna(0)
    
    # Enhanced features for underperforming personas
    training_df['dental_interaction'] = (training_df['query_dental'] + training_df['filter_dental']) * training_df['ma_dental_benefit'] * 2.0
    training_df['doctor_interaction'] = (training_df['query_provider'] + training_df['filter_provider']) * training_df.get('ma_provider_network', 0) * 2.0
    training_df['csnp_interaction'] = (training_df['query_csnp'] + training_df['filter_csnp']) * training_df['csnp'] * 2.0
    training_df['dsnp_interaction'] = (training_df['query_dsnp'] + training_df['filter_dsnp']) * training_df['dsnp'] * 2.0
    training_df['vision_signal'] = (training_df['query_vision'] + training_df['filter_vision']) * training_df['ma_vision']
    additional_features = ['dental_interaction', 'doctor_interaction', 'csnp_interaction', 'dsnp_interaction', 'vision_signal']
    
    feature_columns = behavioral_features + plan_features + additional_features
    
    training_df = training_df[training_df['persona'].isin(PERSONA_COEFFICIENTS.keys())].reset_index(drop=True)
    logger.info(f"Rows after filtering: {len(training_df)}")
    
    X = training_df[feature_columns].fillna(0)
    y = training_df['persona']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y, scaler

def compute_per_persona_accuracy(y_true, y_pred, classes, class_names):
    """Calculate accuracy for each persona."""
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
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def main():
    logger.info("Starting model training...")
    
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
    
    # Split data (80% train, 20% test)
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
    
    # SMOTE for minority classes
    sampling_strategy = {
        persona: 1000 if persona in ['csnp', 'dental', 'doctor', 'dsnp', 'vision'] else max(500, count)
        for persona, count in y_train.value_counts().items()
    }
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        y_train_balanced_encoded = le.transform(y_train_balanced)
        logger.info(f"Balanced training set: {X_train_balanced.shape[0]} samples")
    except Exception as e:
        logger.error(f"SMOTE failed: {e}. Using original training data.")
        X_train_balanced, y_train_balanced = X_train, y_train
        y_train_balanced_encoded = y_train_encoded
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=le.classes_, y=y_train)
    class_weight_dict = {i: class_weights[i] * PERSONA_COEFFICIENTS[le.classes_[i]] for i in range(len(le.classes_))}
    
    # Hyperparameter tuning
    logger.info("Starting hyperparameter tuning...")
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_balanced, y_train_balanced_encoded, X_test, y_test_encoded), n_trials=50)
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return
    
    # Train final model
    try:
        model = lgb.LGBMClassifier(**best_params, random_state=42, class_weight=class_weight_dict, verbose=-1)
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
    
    # Save model and artifacts
    try:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(le, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved model, label encoder, and scaler to disk.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()
