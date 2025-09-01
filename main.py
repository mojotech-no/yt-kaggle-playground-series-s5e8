"""
Kaggle Playground Series S5E8: Bank Term Deposit Prediction

Your Goal: Predict whether a client will subscribe to a bank term deposit.

Dataset Description:
The competition dataset (train and test) was generated using a deep learning model
trained on the Bank Marketing Dataset. Feature distributions are similar, but not
identical, to the original.

Files:
- data/train.csv - the training dataset; y is the binary target
- data/test.csv - the test dataset; predict probability y for each row
- data/sample_submission.csv - sample submission file in correct format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_explore_data():
    """Load and perform initial exploration of the data"""
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nTarget distribution in training data:")
    print(train_df['y'].value_counts(normalize=True))
    
    # Basic info about the datasets
    print("\nTraining data info:")
    print(train_df.info())
    
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    print("\nMissing values in test data:")
    print(test_df.isnull().sum())
    
    return train_df, test_df

def visualize_data(train_df):
    """Create visualizations for data exploration"""
    plt.figure(figsize=(15, 12))
    
    # Target distribution
    plt.subplot(2, 3, 1)
    train_df['y'].value_counts().plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Subscription (y)')
    plt.ylabel('Count')
    
    # Age distribution by target
    plt.subplot(2, 3, 2)
    sns.boxplot(data=train_df, x='y', y='age')
    plt.title('Age Distribution by Target')
    
    # Balance distribution by target
    plt.subplot(2, 3, 3)
    sns.boxplot(data=train_df, x='y', y='balance')
    plt.title('Balance Distribution by Target')
    
    # Duration distribution by target
    plt.subplot(2, 3, 4)
    sns.boxplot(data=train_df, x='y', y='duration')
    plt.title('Duration Distribution by Target')
    
    # Campaign distribution by target
    plt.subplot(2, 3, 5)
    sns.boxplot(data=train_df, x='y', y='campaign')
    plt.title('Campaign Distribution by Target')
    
    # Job distribution by target
    plt.subplot(2, 3, 6)
    job_target = pd.crosstab(train_df['job'], train_df['y'], normalize='index')
    job_target.plot(kind='bar', stacked=True)
    plt.title('Job Distribution by Target')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def preprocess_data(train_df, test_df):
    """Preprocess the data for machine learning"""
    print("Preprocessing data...")
    
    # Separate features and target
    X_train = train_df.drop(['y'], axis=1)
    y_train = train_df['y']
    X_test = test_df.copy()
    
    # Store test IDs for submission
    test_ids = X_test['id']
    
    # Drop ID column from features
    X_train = X_train.drop(['id'], axis=1)
    X_test = X_test.drop(['id'], axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to ensure consistent encoding
        combined_col = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined_col)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    # Feature engineering
    print("Engineering features...")
    
    # Age groups (convert to int to avoid categorical issues with XGBoost)
    X_train['age_group'] = pd.cut(X_train['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(int)
    X_test['age_group'] = pd.cut(X_test['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # Balance categories
    X_train['balance_positive'] = (X_train['balance'] > 0).astype(int)
    X_test['balance_positive'] = (X_test['balance'] > 0).astype(int)
    
    # Duration categories
    X_train['duration_long'] = (X_train['duration'] > X_train['duration'].median()).astype(int)
    X_test['duration_long'] = (X_test['duration'] > X_train['duration'].median()).astype(int)
    
    # Previous contact success rate
    X_train['has_previous'] = (X_train['previous'] > 0).astype(int)
    X_test['has_previous'] = (X_test['previous'] > 0).astype(int)
    
    return X_train, X_test, y_train, test_ids, label_encoders

def train_models(X_train, y_train):
    """Train multiple models and compare their performance"""
    print("Training models...")
    
    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    models = {}
    scores = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_tr_scaled, y_tr)
    lr_pred = lr.predict_proba(X_val_scaled)[:, 1]
    lr_score = roc_auc_score(y_val, lr_pred)
    
    models['logistic_regression'] = (lr, scaler)
    scores['logistic_regression'] = lr_score
    print(f"Logistic Regression AUC: {lr_score:.4f}")
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict_proba(X_val)[:, 1]
    rf_score = roc_auc_score(y_val, rf_pred)
    
    models['random_forest'] = rf
    scores['random_forest'] = rf_score
    print(f"Random Forest AUC: {rf_score:.4f}")
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_score = roc_auc_score(y_val, xgb_pred)
    
    models['xgboost'] = xgb_model
    scores['xgboost'] = xgb_score
    print(f"XGBoost AUC: {xgb_score:.4f}")
    
    # 4. LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_score = roc_auc_score(y_val, lgb_pred)
    
    models['lightgbm'] = lgb_model
    scores['lightgbm'] = lgb_score
    print(f"LightGBM AUC: {lgb_score:.4f}")
    
    # Find best model
    best_model_name = max(scores, key=scores.get)
    best_score = scores[best_model_name]
    
    print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
    
    return models, scores, best_model_name

def create_ensemble(models, X_train, y_train, X_test, test_ids):
    """Create ensemble predictions"""
    print("Creating ensemble predictions...")
    
    # Get predictions from all models
    test_predictions = {}
    
    for name, model in models.items():
        if name == 'logistic_regression':
            model_obj, scaler = model
            X_test_scaled = scaler.transform(X_test)
            pred = model_obj.predict_proba(X_test_scaled)[:, 1]
        else:
            pred = model.predict_proba(X_test)[:, 1]
        
        test_predictions[name] = pred
    
    # Simple average ensemble
    ensemble_pred = np.mean(list(test_predictions.values()), axis=0)
    
    # Weighted ensemble (giving more weight to better performing models)
    weights = {
        'logistic_regression': 0.2,
        'random_forest': 0.25,
        'xgboost': 0.3,
        'lightgbm': 0.25
    }
    
    weighted_pred = sum(weights[name] * test_predictions[name] for name in weights.keys())
    
    return ensemble_pred, weighted_pred, test_predictions

def create_submission(test_ids, predictions, filename='submission.csv'):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_ids,
        'y': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return submission

def main():
    """Main execution function"""
    print("Starting Kaggle Playground Series S5E8 Solution")
    print("=" * 50)
    
    # Load and explore data
    train_df, test_df = load_and_explore_data()
    
    # Create visualizations
    visualize_data(train_df)
    
    # Preprocess data
    X_train, X_test, y_train, test_ids, label_encoders = preprocess_data(train_df, test_df)
    
    # Train models
    models, scores, best_model_name = train_models(X_train, y_train)
    
    # Create ensemble predictions
    ensemble_pred, weighted_pred, individual_preds = create_ensemble(
        models, X_train, y_train, X_test, test_ids
    )
    
    # Create submissions
    create_submission(test_ids, ensemble_pred, 'ensemble_submission.csv')
    create_submission(test_ids, weighted_pred, 'weighted_ensemble_submission.csv')
    
    # Create submission with best individual model
    best_model = models[best_model_name]
    if best_model_name == 'logistic_regression':
        model_obj, scaler = best_model
        X_test_scaled = scaler.transform(X_test)
        best_pred = model_obj.predict_proba(X_test_scaled)[:, 1]
    else:
        best_pred = best_model.predict_proba(X_test)[:, 1]
    
    create_submission(test_ids, best_pred, f'{best_model_name}_submission.csv')
    
    print("\n" + "=" * 50)
    print("Solution completed successfully!")
    print(f"Best individual model: {best_model_name} (AUC: {scores[best_model_name]:.4f})")
    print("Created submissions:")
    print("- ensemble_submission.csv (simple average)")
    print("- weighted_ensemble_submission.csv (weighted average)")
    print(f"- {best_model_name}_submission.csv (best individual model)")

if __name__ == "__main__":
    main()
