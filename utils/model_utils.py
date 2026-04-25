import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import shap
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# Feature Configuration
# -----------------------------
FEATURE_NAMES = [
    'age', 'gender', 'blood_pressure', 'cholesterol',
    'bmi', 'glucose_level', 'smoking', 'alcohol_intake', 'physical_activity'
]

FEATURE_DISPLAY = {
    'age': 'Age',
    'gender': 'Gender',
    'blood_pressure': 'High Blood Pressure',
    'cholesterol': 'Cholesterol Level',
    'bmi': 'BMI',
    'glucose_level': 'Glucose Level',
    'smoking': 'Smoking',
    'alcohol_intake': 'Alcohol Intake',
    'physical_activity': 'Physical Activity'
}

# -----------------------------
# Model Definitions
# -----------------------------
MODEL_CLASSES = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=500,
        random_state=42
    )
}

# -----------------------------
# Load Data
# -----------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = df['heart_disease']
    return X, y, df


# -----------------------------
# Train Models
# -----------------------------
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in MODEL_CLASSES.items():

        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'scaler': scaler if name == 'Logistic Regression' else None
        }

    return results, X_test, y_test


# -----------------------------
# SHAP Values
# -----------------------------
def get_shap_values(model, model_name, X_input, background_data=None):
    """Compute SHAP values for a single prediction"""

    try:
        # Choose explainer
        if model_name in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(model)
        else:
            if background_data is None:
                background_data = X_input
            explainer = shap.LinearExplainer(model, background_data)

        shap_vals = explainer.shap_values(X_input)

        # Handle multi-class output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Handle batch vs single row
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        return shap_vals

    except Exception:
        # Fallback to feature importance approximation
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            pred_prob = model.predict_proba(X_input)[0][1]
            return importance * (pred_prob - 0.5) * 2

        return np.zeros(len(FEATURE_NAMES))


# -----------------------------
# Prediction
# -----------------------------
def predict_risk(model, model_name, patient_data, scaler=None):
    """Predict heart disease risk for a single patient"""

    X = pd.DataFrame([patient_data], columns=FEATURE_NAMES)

    if scaler and model_name == 'Logistic Regression':
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]
    else:
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

    return pred, prob, X