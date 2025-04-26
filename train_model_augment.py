import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
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

# File paths
BEHAVIORAL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/behavior/032025/normalized_us_dce_pro_behavioral_features_0901_2024_0331_2025.csv'
PLAN_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/training/plan_derivation_by_zip.csv'
MODEL_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/model-persona-2.0.0.pkl'
LABEL_ENCODER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/label_encoder_1.pkl'
SCALER_FILE = '/Workspace/Users/jwang77@optumcloud.com/gpd-persona-ai-model-api/data/s-learning-data/models/scaler.pkl'

# Persona list
PERSONAS = ['dental', 'doctor', 'dsnp', 'drug', 'vision', 'csnp']

# Minimal persona info for csnp boost
PERSONA_INFO = {
    'csnp': {'plan_col': 'csnp', 'query_col': 'query_csnp', 'filter_col': 'filter_csnp'},
}

def load_data():
    try:
        # Load behavioral data
        behavioral_df = pd.read_csv(BEHAVIORAL_FILE)
        logger.info(f"Raw behavioral data shape: {behavioral_df.shape}")
        logger.info(f"Behavioral columns: {list(behavioral_df.columns)}")
        
        # Check for persona column
        if 'persona' not in behavioral_df.columns:
            logger.warning("persona column missing, initializing with 'dental'")
            behavioral_df['persona'] = 'dental'
        
        logger.info(f"Raw unique personas: {behavioral_df['persona'].unique()}")
        logger.info(f"Persona value counts (raw):\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        
        # Validate required columns
        required_behavioral_cols = ['persona', 'zip', 'plan_id']
        missing_behavioral_cols = [col for col in required_behavioral_cols if col not in behavioral_df.columns]
        if missing_behavioral_cols:
            logger.warning(f"Missing columns in BEHAVIORAL_FILE: {missing_behavioral_cols}")
            for col in missing_behavioral_cols:
                behavioral_df[col] = 'dummy' if col in ['zip', 'plan_id'] else 'dental'
        
        # Clean persona: strip whitespace, lowercase, replace 'nan'
        behavioral_df['persona'] = behavioral_df['persona'].astype(str).str.strip().str.lower().replace('nan', 'dental')
        behavioral_df['persona'] = behavioral_df['persona'].apply(lambda x: x if x in PERSONAS else 'dental')
        logger.info(f"Persona value counts (after cleaning):\n{behavioral_df['persona'].value_counts(dropna=False).to_string()}")
        
        # Load plan data
        plan_df = pd.read_csv(PLAN_FILE)
        logger.info(f"Raw plan data shape: {plan_df.shape}")
        logger.info(f"Plan columns: {list(plan_df.columns)}")
        
        # Validate plan columns
        required_plan_cols = ['zip', 'plan_id']
        missing_plan_cols = [col for col in required_plan_cols if col not in plan_df.columns]
        if missing_plan_cols:
            logger.warning(f"Missing columns in PLAN_FILE: {missing_plan_cols}")
            for col in missing_plan_cols:
                plan_df[col] = 'dummy'
        
        # Log zip and plan_id samples
        logger.info(f"Behavioral zip sample (first 5): {behavioral_df['zip'].head().tolist()}")
        logger.info(f"Plan zip sample (first 5): {plan_df['zip'].head().tolist()}")
        logger.info(f"Behavioral plan_id sample (first 5): {behavioral_df['plan_id'].head().tolist()}")
        logger.info(f"Plan plan_id sample (first 5): {plan_df['plan_id'].head().tolist()}")
        
        # Log overlap
        behavioral_zips = set(behavioral_df['zip'].astype(str))
        plan_zips = set(plan_df['zip'].astype(str))
        zip_overlap = len(behavioral_zips.intersection(plan_zips)) / max(len(behavioral_zips), 1) * 100
        logger.info(f"Zip overlap: {zip_overlap:.2f}%")
        
        behavioral_plan_ids = set(behavioral_df['plan_id'].astype(str))
        plan_plan_ids = set(plan_df['plan_id'].astype(str))
        plan_id_overlap = len(behavioral_plan_ids.intersection(plan_plan_ids)) / max(len(behavioral_plan_ids), 1) * 100
        logger.info(f"Plan_id overlap: {plan_id_overlap:.2f}%")
        
        behavioral_df['zip'] = behavioral_df['zip'].astype(str).str.strip()
        behavioral_df['plan_id'] = behavioral_df['plan_id'].astype(str).str.strip()
        plan_df['zip'] = plan_df['zip'].astype(str).str.strip()
        plan_df['plan_id'] = plan_df['plan_id'].astype(str).str.strip()
        
        return behavioral_df, plan_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def prepare_features(behavioral_df, plan_df):
    try:
        # Merge data
        df = behavioral_df.merge(plan_df, on=['zip', 'plan_id'], how='left')
        logger.info(f"Rows after merge: {len(df)}")
        logger.info(f"Merged df columns: {list(df.columns)}")
        logger.info(f"Persona distribution after merge:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Validate merge
        if not plan_df.columns.difference(['zip', 'plan_id']).empty:
            merge_success_rate = df[plan_df.columns.difference(['zip', 'plan_id'])].notna().any(axis=1).mean()
            logger.info(f"Merge success rate: {merge_success_rate:.2%}")
        else:
            logger.warning("No plan features to merge")
            merge_success_rate = 0.0
        
        # Validate persona
        if 'persona' not in df.columns or df['persona'].isna().all():
            logger.warning("Persona column missing or all NaN, setting to 'dental'")
            df['persona'] = 'dental'
        
        # Ensure valid persona values: strip whitespace, lowercase, replace 'nan'
        df['persona'] = df['persona'].astype(str).str.strip().str.lower().replace('nan', 'dental')
        df['persona'] = df['persona'].apply(lambda x: x if x in PERSONAS else 'dental')
        logger.info(f"Persona distribution after validation:\n{df['persona'].value_counts(dropna=False).to_string()}")
        
        # Minimal features to avoid errors
        feature_cols = ['csnp_signal']  # Minimal feature for csnp boost
        if 'query_csnp' in df.columns:
            df['csnp_signal'] = df['query_csnp'].clip(lower=0, upper=5) * 8.0
        else:
            logger.warning("query_csnp missing, initializing csnp_signal to 0")
            df['csnp_signal'] = 0
        
        # Ensure non-empty DataFrame
        if df.empty:
            logger.warning("DataFrame empty, creating dummy")
            df = pd.DataFrame({
                'persona': PERSONAS,
                'csnp_signal': [0.1] * len(PERSONAS),
                'zip': ['00000'] * len(PERSONAS),
                'plan_id': ['dummy'] * len(PERSONAS)
            })
            logger.info(f"Dummy DataFrame created:\n{df['persona'].value_counts().to_string()}")
        
        # Ensure all personas are present
        missing_personas = [p for p in PERSONAS if p not in df['persona'].unique()]
        if missing_personas:
            logger.warning(f"Missing personas: {missing_personas}")
            for persona in missing_personas:
                dummy_row = pd.Series({
                    'persona': persona,
                    'csnp_signal': 0.1,
                    'zip': '00000',
                    'plan_id': 'dummy'
                })
                df = pd.concat([df, dummy_row.to_frame().T], ignore_index=True)
            logger.info(f"Added dummy personas:\n{df['persona'].value_counts().to_string()}")
        
        X = df[feature_cols].fillna(0)
        y = df['persona']
        
        # Final validation
        if not set(y.unique()).intersection(PERSONAS):
            logger.error(f"No valid personas in y: {y.unique()}")
            df = pd.DataFrame({
                'persona': PERSONAS,
                'csnp_signal': [0.1] * len(PERSONAS),
                'zip': ['00000'] * len(PERSONAS),
                'plan_id': ['dummy'] * len(PERSONAS)
            })
            X = df[feature_cols].fillna(0)
            y = df['persona']
        
        logger.info(f"Final persona distribution:\n{y.value_counts(dropna=False).to_string()}")
        logger.info(f"X shape: {X.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

def train_model():
    # Load data
    behavioral_df, plan_df = load_data()
    X, y = prepare_features(behavioral_df, plan_df)
    
    # Validate target classes
    logger.info(f"y unique values in train_model: {y.unique()}")
    missing_personas = [p for p in PERSONAS if p not in y.unique()]
    if missing_personas:
        logger.error(f"Target classes missing: {missing_personas}")
        raise ValueError(f"Target classes {missing_personas} not present in data")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Log pre-SMOTE distribution
    logger.info(f"Pre-SMOTE persona distribution:\n{pd.Series(y).value_counts().to_string()}")
    
    # Apply SMOTE
    class_counts = pd.Series(y).value_counts()
    sampling_strategy = {
        persona: max(count, 400 if persona == 'csnp' else 500 if persona == 'dsnp' else 2000) for persona, count in class_counts.items()
    }
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
    X, y_encoded = smote.fit_resample(X, y_encoded)
    logger.info(f"Rows after SMOTE: {len(X)}")
    logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(le.inverse_transform(y_encoded)).value_counts().to_string()}")
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    logger.info(f"Test set label distribution:\n{pd.Series(le.inverse_transform(y_test)).value_counts().to_string()}")
    
    # Train CatBoost
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    feature_importances = []
    
    # Class weights for csnp
    class_weights = {le.transform(['csnp'])[0]: 3.0, **{i: 1.0 for i in range(len(le.classes_)) if i != le.transform(['csnp'])[0]}}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        model = CatBoostClassifier(
            iterations=600,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10,
            loss_function='MultiClass',
            class_weights=class_weights,
            random_seed=42,
            verbose=50
        )
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=200
        )
        models.append(model)
        feature_importances.append(model.get_feature_importance())
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions
    y_pred_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    y_pred = np.argmax(y_pred_probas, axis=1)
    
    # Log prediction distribution
    logger.info(f"Prediction distribution:\n{pd.Series(le.inverse_transform(y_pred)).value_counts().to_string()}")
    
    # Log feature importances
    avg_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': avg_importances
    }).sort_values(by='Importance', ascending=False)
    logger.info("Feature Importances:\n" + importance_df.to_string())
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"Overall Accuracy: {acc * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
    if acc < 0.8:
        logger.warning(f"Accuracy {acc * 100:.2f}% is below target of 80%.")
    
    # Per-persona accuracy
    per_persona_accuracy = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_accuracy = accuracy_score(y_test[mask], y_pred[mask])
            per_persona_accuracy[cls_name] = cls_accuracy * 100
        else:
            per_persona_accuracy[cls_name] = 0.0
    logger.info("Per-Persona Accuracy (%):")
    for persona, acc in per_persona_accuracy.items():
        logger.info(f"  {persona}: {acc:.2f}%")
    
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models[0], f)
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved model, label encoder, and scaler to disk.")

if __name__ == '__main__':
    train_model()
