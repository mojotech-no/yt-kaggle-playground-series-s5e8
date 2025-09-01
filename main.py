"""
Train a baseline model and generate a Kaggle submission for Playground S5E8.

What it does
- Loads training and test CSVs from ./data
- Preprocesses numeric/categorical features via a robust sklearn Pipeline
- Trains a baseline classifier
- Reports CV ROC-AUC (Stratified KFold)
- Fits on full training data and writes submission.csv with probabilities

Usage examples
- Default full run (may take a few minutes depending on hardware):
	python -m main

- Quick smoke test on a subset of rows:
	python -m main --nrows 50000 --cv 3 --out submission_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"


def load_data(nrows: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if not TRAIN_PATH.exists() or not TEST_PATH.exists():
		raise FileNotFoundError(
			f"Expected files not found. Ensure {TRAIN_PATH} and {TEST_PATH} exist."
		)
	# Low-memory False to keep types consistent; dtype inference is ok here.
	train = pd.read_csv(TRAIN_PATH, nrows=nrows, low_memory=False)
	test = pd.read_csv(TEST_PATH, nrows=nrows, low_memory=False)
	return train, test


def get_feature_splits(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
	features = [c for c in df.columns if c != target]
	cat_cols = [c for c in features if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
	num_cols = [c for c in features if c not in cat_cols]
	return num_cols, cat_cols


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
	numeric_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)
	categorical_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			# Avoid sparse_output kw to keep compatibility across sklearn versions
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	pre = ColumnTransformer(
		transformers=[
			("num", numeric_pipe, num_cols),
			("cat", categorical_pipe, cat_cols),
		]
	)

	model = LogisticRegression(
		solver="lbfgs",
		max_iter=1000,
		# class_weight can help if target is imbalanced
		class_weight=None,
	)

	pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
	return pipe


def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int, seed: int) -> float:
	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
	scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=None)
	return float(np.mean(scores))


def fit_full_and_predict(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
	pipe.fit(X, y)
	proba = pipe.predict_proba(X_test)[:, 1]
	return proba


def make_submission(ids: pd.Series, preds: np.ndarray, out_path: Path) -> None:
	sub = pd.DataFrame({"id": ids, "y": preds})
	sub.to_csv(out_path, index=False)


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Baseline trainer for Playground S5E8")
	parser.add_argument("--nrows", type=int, default=None, help="Limit rows for both train/test for a quick run")
	parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for ROC-AUC evaluation")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for CV shuffling")
	parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output submission CSV path")
	args = parser.parse_args(argv)

	print("Loading data…")
	train, test = load_data(nrows=args.nrows)

	target_col = "y"
	if target_col not in train.columns:
		raise KeyError(f"Target column '{target_col}' not found in training data.")

	# Extract features/target
	y = train[target_col]
	X = train.drop(columns=[target_col])
	X_test = test.copy()

	# Safety: coerce binary target to 0/1 if it's strings like 'yes'/'no'
	if y.dtype == "object":
		y = y.map({"no": 0, "yes": 1}).astype(int)

	# Identify columns by type
	num_cols, cat_cols = get_feature_splits(pd.concat([X, X_test], axis=0, ignore_index=True), target_col)
	print(f"Detected {len(num_cols)} numeric and {len(cat_cols)} categorical features.")

	pipe = build_pipeline(num_cols, cat_cols)

	# CV evaluation
	if args.cv and args.cv > 1:
		print(f"Running {args.cv}-fold CV (ROC-AUC)…")
		cv_auc = evaluate_cv(pipe, X, y, cv=args.cv, seed=args.seed)
		print(f"CV ROC-AUC: {cv_auc:.5f}")

	# Fit on full data and predict test
	print("Training on full data and generating predictions…")
	preds = fit_full_and_predict(pipe, X, y, X_test)

	# Save submission
	ids_col = "id"
	if ids_col not in test.columns:
		raise KeyError(f"ID column '{ids_col}' not found in test data.")
	make_submission(test[ids_col], preds, args.out)
	print(f"Wrote submission to {args.out.resolve()}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
