"""
Kaggle Playground Series S5E8: Binary Classification with Bank Dataset
Predicts whether a client will subscribe to a bank term deposit.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    """Load training and test data if available."""
    data_dir = Path("data")

    # Check if data files exist
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Please download the Kaggle dataset and place files in the data/ directory."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Please download the Kaggle dataset and place files in the data/ directory."
        )

    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    print(f"Training data shape: {train_df.shape}")

    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")

    # Load sample submission to understand format
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        print(f"Sample submission shape: {sample_df.shape}")
        print(f"Sample submission columns: {sample_df.columns.tolist()}")
    else:
        print("Sample submission file not found, will infer format from test data")
        sample_df = None

    return train_df, test_df, sample_df


def preprocess_data(train_df, test_df):
    """Preprocess training and test data."""
    print("Preprocessing data...")

    # Separate features and target
    X_train = train_df.drop("y", axis=1)
    y_train = train_df["y"]
    X_test = test_df.copy()

    # Get feature columns (excluding id if present)
    feature_cols = [col for col in X_train.columns if col != "id"]

    # Keep only feature columns for training
    X_train_features = X_train[feature_cols].copy()
    X_test_features = X_test[feature_cols].copy()

    # Handle categorical variables
    categorical_cols = X_train_features.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {categorical_cols.tolist()}")

        # Combine train and test for consistent encoding
        combined_features = pd.concat(
            [X_train_features, X_test_features], axis=0, ignore_index=True
        )

        # Label encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            combined_features[col] = le.fit_transform(
                combined_features[col].astype(str)
            )
            label_encoders[col] = le

        # Split back to train and test
        n_train = len(X_train_features)
        X_train_processed = combined_features[:n_train].copy()
        X_test_processed = combined_features[n_train:].copy()
    else:
        X_train_processed = X_train_features.copy()
        X_test_processed = X_test_features.copy()

    print(f"Processed training features shape: {X_train_processed.shape}")
    print(f"Processed test features shape: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train


def train_model(X_train, y_train):
    """Train a simple but effective model."""
    print("Training model...")

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Try both LogisticRegression and RandomForest (lightweight models)
    models = {
        "logistic": LogisticRegression(random_state=42, max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,  # Limit depth for speed
        ),
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_tr, y_tr)

        # Validate
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_pred_proba)

        print(f"{name} validation AUC: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_name = name

    print(f"Best model: {best_name} with AUC: {best_score:.4f}")

    # Retrain on full training data
    print("Retraining best model on full training data...")
    best_model.fit(X_train, y_train)

    return best_model


def create_submission(model, X_test, test_df, sample_df=None):
    """Create submission file in the correct format."""
    print("Creating predictions...")

    # Make predictions
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Create submission dataframe
    if sample_df is not None:
        # Use sample submission format
        submission = sample_df.copy()
        # Assume first column is ID and second is prediction
        id_col = submission.columns[0]
        pred_col = submission.columns[1]

        if "id" in test_df.columns:
            submission[id_col] = test_df["id"]
        else:
            submission[id_col] = range(len(test_pred_proba))

        submission[pred_col] = test_pred_proba
    else:
        # Infer format - assume id column exists or create index
        if "id" in test_df.columns:
            submission = pd.DataFrame({"id": test_df["id"], "y": test_pred_proba})
        else:
            submission = pd.DataFrame(
                {"id": range(len(test_pred_proba)), "y": test_pred_proba}
            )

    # Save submission
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Submission columns: {submission.columns.tolist()}")
    print("First few predictions:")
    print(submission.head())

    return submission


def main():
    """Main function to run the entire pipeline."""
    try:
        print("=== Kaggle Playground Series S5E8: Bank Term Deposit Prediction ===")

        # Load data
        train_df, test_df, sample_df = load_data()

        # Preprocess data
        X_train, X_test, y_train = preprocess_data(train_df, test_df)

        # Train model
        model = train_model(X_train, y_train)

        # Create submission
        create_submission(model, X_test, test_df, sample_df)

        print("\n=== Pipeline completed successfully! ===")
        print("Next steps:")
        print("1. Review the submission.csv file")
        print("2. Submit to Kaggle competition")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo run this script:")
        print("1. Download the competition data from Kaggle")
        print(
            "2. Place train.csv, test.csv, and sample_submission.csv in the data/ directory"
        )
        print("3. Run this script again")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
