#!/usr/bin/env python3
"""
Kaggle Playground Series S5E8: Binary Classification with Bank Dataset
Goal: Predict whether a client will subscribe to a bank term deposit

This script implements a comprehensive machine learning pipeline including:
- Data exploration and preprocessing
- Feature engineering
- Multiple model training (Random Forest, XGBoost, LightGBM, CatBoost)
- Hyperparameter optimization with Optuna
- Model ensembling
- Submission generation
"""

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from optuna.samplers import TPESampler
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna

import logging
import warnings

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BankMarketingPredictor:
    """Main class for the bank marketing prediction pipeline."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.models = {}
        self.ensemble_weights = {}

    def load_data(self):
        """Load training and test datasets."""
        logger.info("Loading datasets...")

        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            logger.error(f"Data files not found in {self.data_dir}")
            logger.info(
                "Please ensure train.csv and test.csv are in the data directory"
            )
            return False

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # Use a sample for faster processing on weak CPU
        if len(self.train_df) > 100000:
            logger.info("Using sample of training data for faster processing...")
            self.train_df = self.train_df.sample(n=100000, random_state=42).reset_index(
                drop=True
            )

        logger.info(f"Training data shape: {self.train_df.shape}")
        logger.info(f"Test data shape: {self.test_df.shape}")

        return True

    def explore_data(self):
        """Perform exploratory data analysis."""
        logger.info("Performing exploratory data analysis...")

        # Basic info
        print("\n=== DATASET OVERVIEW ===")
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")

        print("\n=== TRAINING DATA INFO ===")
        print(self.train_df.info())

        print("\n=== MISSING VALUES ===")
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()

        print("Training data missing values:")
        print(missing_train[missing_train > 0])
        print("\nTest data missing values:")
        print(missing_test[missing_test > 0])

        # Target distribution
        if "y" in self.train_df.columns:
            print("\n=== TARGET DISTRIBUTION ===")
            target_dist = self.train_df["y"].value_counts()
            print(target_dist)
            print(f"Target balance: {target_dist[1] / len(self.train_df):.3f}")

        # Data types
        print("\n=== DATA TYPES ===")
        print("Categorical columns:")
        cat_cols = self.train_df.select_dtypes(include=["object"]).columns.tolist()
        if "y" in cat_cols:
            cat_cols.remove("y")
        print(cat_cols)

        print("\nNumerical columns:")
        num_cols = self.train_df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        if "id" in num_cols:
            num_cols.remove("id")
        if "y" in num_cols:
            num_cols.remove("y")
        print(num_cols)

        return cat_cols, num_cols

    def preprocess_data(self):
        """Preprocess the data including encoding and scaling."""
        logger.info("Preprocessing data...")

        # Combine train and test for consistent preprocessing
        train_size = len(self.train_df)

        # Separate target and features
        if "y" in self.train_df.columns:
            self.y_train = self.train_df["y"].copy()
            train_features = self.train_df.drop(["y"], axis=1)
        else:
            train_features = self.train_df.copy()

        test_features = self.test_df.copy()

        # Remove ID columns if present
        id_cols = ["id"]
        for col in id_cols:
            if col in train_features.columns:
                train_features = train_features.drop(col, axis=1)
            if col in test_features.columns:
                test_features = test_features.drop(col, axis=1)

        # Combine for preprocessing
        all_features = pd.concat(
            [train_features, test_features], axis=0, ignore_index=True
        )

        # Handle missing values
        for col in all_features.columns:
            if all_features[col].dtype == "object":
                # Fill categorical missing values with mode
                mode_val = (
                    all_features[col].mode()[0]
                    if len(all_features[col].mode()) > 0
                    else "unknown"
                )
                all_features[col] = all_features[col].fillna(mode_val)
            else:
                # Fill numerical missing values with median
                median_val = all_features[col].median()
                all_features[col] = all_features[col].fillna(median_val)

        # Encode categorical variables
        categorical_cols = all_features.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            all_features[col] = le.fit_transform(all_features[col].astype(str))
            self.label_encoders[col] = le

        # Feature engineering
        all_features = self.engineer_features(all_features)

        # Split back to train and test
        self.X_train = all_features[:train_size].copy()
        self.X_test = all_features[train_size:].copy()

        # Store feature names
        self.feature_names = self.X_train.columns.tolist()

        logger.info(f"Preprocessed feature shape: {self.X_train.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")

        return True

    def engineer_features(self, df):
        """Create additional features - simplified for weak CPU."""
        logger.info("Engineering features...")

        # Create age groups if age column exists
        age_cols = [col for col in df.columns if "age" in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            df[f"{age_col}_group"] = pd.cut(
                df[age_col],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=[0, 1, 2, 3, 4, 5],
            )
            df[f"{age_col}_group"] = df[f"{age_col}_group"].astype(int)

        # Only create a few essential interaction features
        if "balance" in df.columns:
            df["balance_positive"] = (df["balance"] > 0).astype(int)
            df["balance_log"] = np.log1p(np.abs(df["balance"]) + 1)

        if "duration" in df.columns:
            df["duration_log"] = np.log1p(df["duration"])

        return df

    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model."""
        logger.info("Training Random Forest...")

        rf_params = {
            "n_estimators": 50,  # Reduced from 200
            "max_depth": 8,  # Reduced from 15
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": 2,  # Reduced from -1
        }

        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_pred = rf_model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            logger.info(f"Random Forest validation AUC: {val_score:.4f}")

        return rf_model

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        logger.info("Training XGBoost...")

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 4,  # Reduced from 6
            "learning_rate": 0.1,
            "n_estimators": 100,  # Reduced from 300
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 2,  # Reduced from -1
            "early_stopping_rounds": 50,
        }

        xgb_model = xgb.XGBClassifier(**xgb_params)

        if X_val is not None and y_val is not None:
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = xgb_model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            logger.info(f"XGBoost validation AUC: {val_score:.4f}")
        else:
            xgb_model.fit(X_train, y_train)

        return xgb_model

    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model."""
        logger.info("Training LightGBM...")

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 15,  # Reduced from 31
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42,
            "n_jobs": 2,  # Reduced from -1
            "verbose": -1,
        }

        lgb_model = lgb.LGBMClassifier(
            **lgb_params, n_estimators=100
        )  # Reduced from 300

        if X_val is not None and y_val is not None:
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            val_pred = lgb_model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            logger.info(f"LightGBM validation AUC: {val_score:.4f}")
        else:
            lgb_model.fit(X_train, y_train)

        return lgb_model

    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model."""
        logger.info("Training CatBoost...")

        cat_params = {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "iterations": 100,  # Reduced from 300
            "learning_rate": 0.1,
            "depth": 4,  # Reduced from 6
            "random_seed": 42,
            "verbose": False,
            "thread_count": 2,  # Limited threads
        }

        cat_model = cat.CatBoostClassifier(**cat_params)

        if X_val is not None and y_val is not None:
            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False,
            )
            val_pred = cat_model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            logger.info(f"CatBoost validation AUC: {val_score:.4f}")
        else:
            cat_model.fit(X_train, y_train)

        return cat_model

    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters using Optuna."""
        logger.info("Optimizing hyperparameters with Optuna...")

        def objective(trial):
            # XGBoost hyperparameters
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
            }

            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBClassifier(**xgb_params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )

                val_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_pred)
                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        logger.info(f"Best validation score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return study.best_params

    def train_models(self):
        """Train lightweight models for weak CPU."""
        logger.info("Training lightweight models...")

        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train,
        )

        # Train only two lightweight models
        self.models["xgb"] = self.train_xgboost(X_tr, y_tr, X_val, y_val)
        self.models["lgb"] = self.train_lightgbm(X_tr, y_tr, X_val, y_val)

        # Calculate ensemble weights based on validation performance
        val_scores = {}
        for name, model in self.models.items():
            val_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_pred)
            val_scores[name] = score
            logger.info(f"{name.upper()} validation AUC: {score:.4f}")

        # Normalize weights
        total_score = sum(val_scores.values())
        self.ensemble_weights = {
            name: score / total_score for name, score in val_scores.items()
        }

        logger.info("Ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            logger.info(f"  {name.upper()}: {weight:.3f}")

    def make_predictions(self):
        """Generate predictions on test set."""
        logger.info("Making predictions on test set...")

        # Individual model predictions
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict_proba(self.X_test)[:, 1]
            predictions[name] = pred
            logger.info(
                f"{name.upper()} predictions range: [{pred.min():.4f}, {pred.max():.4f}]"
            )

        # Ensemble prediction (weighted average)
        ensemble_pred = np.zeros(len(self.X_test))
        for name, pred in predictions.items():
            ensemble_pred += self.ensemble_weights[name] * pred

        logger.info(
            f"Ensemble predictions range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]"
        )

        return ensemble_pred, predictions

    def create_submission(self, predictions):
        """Create submission file."""
        logger.info("Creating submission file...")

        # Load sample submission to get the correct format
        sample_sub_path = self.data_dir / "sample_submission.csv"
        if sample_sub_path.exists():
            sample_sub = pd.read_csv(sample_sub_path)
            submission = sample_sub.copy()
            submission["y"] = predictions
        else:
            # Create submission from test data IDs
            submission = pd.DataFrame(
                {
                    "id": self.test_df["id"]
                    if "id" in self.test_df.columns
                    else range(len(predictions)),
                    "y": predictions,
                }
            )

        # Save submission
        submission_path = "submission.csv"
        submission.to_csv(submission_path, index=False)

        logger.info(f"Submission saved to {submission_path}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info("Prediction statistics:")
        logger.info(f"  Mean: {predictions.mean():.4f}")
        logger.info(f"  Std: {predictions.std():.4f}")
        logger.info(f"  Min: {predictions.min():.4f}")
        logger.info(f"  Max: {predictions.max():.4f}")

        return submission_path

    def run_full_pipeline(self):
        """Run the complete machine learning pipeline."""
        logger.info("Starting full ML pipeline...")

        # Load data
        if not self.load_data():
            return None

        # Explore data
        cat_cols, num_cols = self.explore_data()

        # Preprocess data
        if not self.preprocess_data():
            return None

        # Train models
        self.train_models()

        # Make predictions
        ensemble_pred, individual_preds = self.make_predictions()

        # Create submission
        submission_path = self.create_submission(ensemble_pred)

        logger.info("Pipeline completed successfully!")
        return submission_path


def main():
    """Main function to run the prediction pipeline."""
    print("🏦 Kaggle Playground Series S5E8: Bank Marketing Prediction")
    print("=" * 60)

    # Initialize predictor
    predictor = BankMarketingPredictor()

    # Run pipeline
    submission_path = predictor.run_full_pipeline()

    if submission_path:
        print(f"\n✅ Success! Submission file created: {submission_path}")
        print("📊 Ready for submission to Kaggle!")
    else:
        print("\n❌ Pipeline failed. Please check the logs and data files.")


if __name__ == "__main__":
    main()
