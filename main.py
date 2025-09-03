import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train columns:", train_df.columns.tolist())
print("Train head:")
print(train_df.head())
print("Train info:")
print(train_df.info())
print("Missing values in train:")
print(train_df.isnull().sum())
print("Target distribution:")
print(train_df["y"].value_counts())

# Preprocess
# Encode categorical variables
categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
encoders = {}
for col in categorical_cols:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le

# Fill missing values if any (though probably none)
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# Features and target
X = train_df.drop(["id", "y"], axis=1)
y = train_df["y"]
X_test = test_df.drop(["id"], axis=1)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on val
y_val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC: {auc}")

# Predict on test
test_pred = model.predict_proba(X_test)[:, 1]

# Submission
submission = sample_submission.copy()
submission["y"] = test_pred
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
