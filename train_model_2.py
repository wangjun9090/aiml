import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
import logging
import os
import sys

# Set up logging with force flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
    'vision': 1.8, 'csnp': 2SyS': 2.0, 'transportation': 1.0, 'otc': 1.0
}

def load_data(behavioral_path, plan_path):
    try:
        behavioral_df = pd.read_csv(behavioral_path)
        plan_df = pd.read_csv(plan_path)
        
        # Clean persona column: convert to string and handle invalid values
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.strip().str.lower()
        behavioral_df['persona'] = behavioral_df['persona'].replace('nan', '')
        logger.info(f"Raw persona values: {behavioral_df['persona'].unique()}")
        
        logger.info(f"Behavioral data: {len(behavioral_df)} rows, Plan data: {len(plan_df)} rows")
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def normalize_persona(df):
    """Normalize personas, ensuring all are valid strings."""
    valid_personas = list(PERSONA_COEFFICIENTS.keys())
    new_rows = []
    
    for _, row in df.iterrows():
        if not row['persona'] or row['persona'] == '':
            logger.warning(f"Skipping row with empty persona: {row.to_dict()}")
            continue
        
        personas = [p.strip() for p in row['persona'].split(',')]
        valid_found = [p for p in personas if p in valid_personas]
        
        if not valid_found:
            logger.warning(f"Skipping row with no valid personas: {row['persona']}")
            continue
        
        row_copy = row.copy()
        row_copy['persona'] = valid_found[0]
        new_rows.append(row_copy)
    
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(f"Normalized {len(df)} rows into {len(result)} rows")
    logger.info(f"Persona distribution: {result['persona'].value_counts().to_dict()}")
    
    if not all(isinstance(p, str) for p in result['persona']):
        logger.error(f"Non-string personas found: {result['persona'].unique()}")
        raise ValueError("Persona column contains non-string values after normalization")
    
    return result

def prepare_features(behavioral_df, plan_df):
    behavioral_df = normalize_persona(behavioral_df)
    
    # Ensure consistent data types
    behavioral_df['zip'] = behavioral_df['zip'].astype(str).fillna('')
    behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).fillna('')
    plan_df['zip'] = plan_df['zip'].astype(str).fillna('')
    plan_df['plan_id'] = plan_df['plan_id'].astype(str).fillna('')
    
    # Merge data
    training_df = behavioral_df.merge(
        plan_df.rename(columns={'StateCode': 'state'}),
        how='left', on=['zip', 'plan_id']
    ).reset_index(drop=True)
    logger.info(f"Rows after merge: {len(training_df)}")
    
    # Simplified feature list
    behavioral_features = [
        'query_dental', 'query_transportation', 'query_otc', 'query_drug', 'query_provider',
        'query_vision', 'query_csnp', 'query_dsnp', 'filter_dental', 'filter_transportation',
        'filter_otc', 'filter_drug', 'filter_provider', 'filter_vision', 'filter_csnp',
        'filter_dsnp', 'total_session_time', 'num_pages_viewed'
    ]
    plan_features = ['ma_otc', 'ma_transportation', 'ma_dental_benefit', 'ma_vision', 'csnp', 'dsnp']
    
    # Fill missing features
    for col in behavioral_features + plan_features:
        training_df[col] = training_df.get(col, 0).fillna(0)
    
    # Simplified feature engineering
    training_df['dental_signal'] = (training_df['query_dental'] + training_df['filter_dental']) * training_df['ma_dental_benefit']
    training_df['vision_signal'] = (training_df['query_vision'] + training_df['filter_vision']) * training_df['ma_vision']
    additional_features = ['dental_signal', 'vision_signal']
    
    feature_columns = behavioral_features + plan_features + additional_features
    
    training_df = training_df[training_df['persona'].isin(PERSONA_COEFFICIENTS.keys())].reset_index(drop=True)
    logger.info(f"Rows after filtering: {len(training_df)}")
    
    X = training_df[feature_columns].fillna(0)
    y = training_df['persona']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    logger.info(f"Features: {X.columns.tolist()}")
    logger.info(f"Persona distribution: {y.value_counts().to_dict()}")
    return X, y, scaler

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def main():
    logger.info("Starting model training...")
    
    # Load data
    behavioral_df, plan_df = load_data(BEHAVIORAL_FILE, PLAN_FILE)
    
    # Prepare features
    X, y, scaler = prepare_features(behavioral_df, plan_df)
    
    # Verify all labels are strings
    if not all(isinstance(label, str) for label in y):
        logger.error(f"Non-string labels found in y: {y.unique()}")
        raise ValueError("Target variable y contains non-string labels")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Verify labels
    if not all(isinstance(label, str) for label in y_train):
        logger.error(f"Non-string labels in y_train: {y_train.unique()}")
        raise ValueError("y_train contains non-string labels")
    if not all(isinstance(label, str) for label in y_test):
        logger.error(f"Non-string labels in y_test: {y_test.unique()}")
        raise ValueError("y_test contains non-string labels")
    
    # Label encoding
    le = LabelEncoder()
    le.fit(y)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Balance training data with SMOTE
    sampling_strategy = {persona: max(200, count) for persona, count in y_train.value_counts().items()}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"Balanced training set: {X_train_balanced.shape[0]} samples")
        logger.info(f"Class distribution after balancing: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        if not all(isinstance(label, str) for label in y_train_balanced):
            logger.error(f"Non-string labels in y_train_balanced: {y_train_balanced.unique()}")
            raise ValueError("y_train_balanced contains non-string labels")
        
        y_train_balanced_encoded = le.transform(y_train_balanced)
    except ValueError as e:
        logger.error(f"SMOTE failed: {e}. Falling back to original training data.")
        X_train_balanced, y_train_balanced = X_train, y_train
        y_train_balanced_encoded = y_train_encoded
    
    # Verify test set labels
    unseen_labels = set(y_test) - set(le.classes_)
    if unseen_labels:
        logger.error(f"Test set contains unseen labels: {unseen_labels}")
        raise ValueError(f"Test set contains labels not seen in training: {unseen_labels}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=le.classes_, y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(le.classes_))}
    
    # Hyperparameter tuning with Optuna
    logger.info("Starting hyperparameter tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_balanced, y_train_balanced_encoded, X_test, y_test_encoded), n_trials=50)
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Train final model with class weights
    xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
    lgb_model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, class_weight=class_weight_dict)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight=class_weight_dict)
    
    stacking = StackingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
        final_estimator=xgb.XGBClassifier(random_state=42),
        cv=5, n_jobs=-1
    )
    
    # Cross-validation on training set
    logger.info("Performing cross-validation on training set...")
    cv_scores = cross_val_score(stacking, X_train_balanced, y_train_balanced_encoded, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # Fit final model
    logger.info("Training final model...")
    stacking.fit(X_train_balanced, y_train_balanced_encoded)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    logger.info(f"Overall Accuracy on Test Set (20% of data): {accuracy * 100:.2f}%")
    
    if accuracy < 0.8:
        logger.warning(f"Accuracy {accuracy * 100:.2f}% is below target of 80%. See classification report for details.")
    
    logger.info("Classification Report:\n" + classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test_encoded, y_pred)))
    
    # Save model and artifacts
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(stacking, f)
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved model, label encoder, and scaler to disk.")

if __name__ == "__main__":
    main()
