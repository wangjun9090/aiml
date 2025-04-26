import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
PERSONAS = ['doctor', 'csnp', 'drug', 'dental', 'dsnp', 'vision']
MODEL_FILE = 'model.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
SCALER_FILE = 'scaler.pkl'

def load_data():
    # Placeholder: Replace with your actual data loading logic
    # Returns behavioral_df and plan_df
    pass

def prepare_features(behavioral_df, plan_df):
    # Placeholder: Replace with your feature preparation logic
    # Returns X (features) and y (target)
    pass

def train_model():
    # Load data
    behavioral_df, plan_df = load_data()
    X, y = prepare_features(behavioral_df, plan_df)
    
    # Convert y to a pandas Series and clean it
    y = pd.Series(y).astype(str).str.strip()
    
    # Log unique values in y
    logger.info(f"Unique values in y: {y.unique().tolist()}")
    
    # Check for missing personas
    missing_personas = [p for p in PERSONAS if p not in y.unique()]
    if missing_personas:
        logger.error(f"Target classes missing: {missing_personas}")
        raise ValueError(f"Target classes {missing_personas} not present in data")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Log pre-SMOTE distribution
    logger.info(f"Pre-SMOTE persona distribution:\n{y.value_counts().to_string()}")
    
    # Apply SMOTE to balance classes, emphasizing csnp, dsnp, and dental
    class_counts = y.value_counts()
    sampling_strategy = {
        persona: max(count, 500 if persona in ['csnp', 'dsnp', 'dental'] else 2000)
        for persona, count in class_counts.items()
    }
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
    X, y_encoded = smote.fit_resample(X, y_encoded)
    logger.info(f"Rows after SMOTE: {len(X)}")
    logger.info(f"Post-SMOTE persona distribution:\n{pd.Series(le.inverse_transform(y_encoded)).value_counts().to_string()}")
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Test set label distribution:\n{pd.Series(le.inverse_transform(y_test)).value_counts().to_string()}")
    
    # Train CatBoost with class weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    
    # Assign higher weight to csnp, dsnp, and dental
    class_weights = {
        i: 3.0 if le.classes_[i] in ['csnp', 'dsnp', 'dental'] else 1.0
        for i in range(len(le.classes_))
    }
    
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
        logger.info(f"Fold {fold+1} training completed")
    
    # Ensemble predictions
    y_pred_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)
    y_pred = np.argmax(y_pred_probas, axis=1)
    
    # Log prediction distribution
    logger.info(f"Prediction distribution:\n{pd.Series(le.inverse_transform(y_pred)).value_counts().to_string()}")
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"Overall Accuracy: {acc * 100:.2f}%")
    logger.info(f"Macro F1 Score: {macro_f1:.2f}")
    
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
