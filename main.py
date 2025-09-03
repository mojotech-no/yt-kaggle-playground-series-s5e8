from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_SUB_PATH = os.path.join(DATA_DIR, "submission.csv")


@dataclass
class DataInfo:
	num_features: List[str]
	cat_features: List[str]
	id_col: str
	target_col: str


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
	train = pd.read_csv(TRAIN_PATH)
	test = pd.read_csv(TEST_PATH)
	return train, test


def infer_schema(df: pd.DataFrame) -> DataInfo:
	# Assumptions from Kaggle playground: id column exists and target is 'y'
	id_col = "id" if "id" in df.columns else df.columns[0]
	target_col = "y" if "y" in df.columns else df.columns[-1]
	feature_cols = [c for c in df.columns if c not in {id_col, target_col}]

	# Infer categorical vs numeric
	num_features: List[str] = []
	cat_features: List[str] = []
	for c in feature_cols:
		if pd.api.types.is_numeric_dtype(df[c]):
			num_features.append(c)
		else:
			cat_features.append(c)

	return DataInfo(num_features=num_features, cat_features=cat_features, id_col=id_col, target_col=target_col)


def make_model() -> LGBMClassifier:
	threads = min(4, os.cpu_count() or 2)
	clf = LGBMClassifier(
		n_estimators=1000,
		learning_rate=0.05,
		objective="binary",
		subsample=0.8,
		colsample_bytree=0.8,
		reg_lambda=1.0,
		num_leaves=31,
		max_bin=127,
		bagging_freq=1,
		random_state=42,
		n_jobs=threads,
		class_weight="balanced",
		verbose=-1,
	)
	return clf


def _prepare_features(train: pd.DataFrame, test: pd.DataFrame, info: DataInfo) -> Tuple[pd.DataFrame, pd.DataFrame]:
	X_train = train.drop(columns=[info.target_col])
	X_test = test.copy()

	# Numeric: fill with median from train
	for c in info.num_features:
		med = X_train[c].median()
		X_train[c] = X_train[c].fillna(med)
		X_test[c] = X_test[c].fillna(med)

	# Categorical: fill missing and align categories across train+test
	for c in info.cat_features:
		tr = X_train[c].astype("string").fillna("__MISSING__")
		te = X_test[c].astype("string").fillna("__MISSING__")
		both = pd.concat([tr, te], axis=0)
		cats = pd.Categorical(both).categories
		X_train[c] = pd.Categorical(tr, categories=cats)
		X_test[c] = pd.Categorical(te, categories=cats)

	return X_train, X_test


def train_and_eval(train: pd.DataFrame, test: pd.DataFrame, info: DataInfo) -> Tuple[LGBMClassifier, pd.DataFrame, pd.DataFrame]:
	X_train_all, X_test_prepared = _prepare_features(train, test, info)
	y = train[info.target_col]

	X_tr, X_va, y_tr, y_va = train_test_split(
		X_train_all, y, test_size=0.1, random_state=42, stratify=y
	)

	clf = make_model()
	clf.fit(
		X_tr,
		y_tr,
		eval_set=[(X_va, y_va)],
		eval_metric="auc",
		callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
		categorical_feature=info.cat_features,
	)

	va_pred = clf.predict_proba(X_va)[:, 1]
	auc = roc_auc_score(y_va, va_pred)
	print(f"Validation ROC AUC: {auc:.5f}")
	return clf, X_train_all, X_test_prepared


def cross_val_score_auc(*_args, **_kwargs) -> float:  # kept for compatibility
	raise RuntimeError("cross_val_score_auc disabled for speed. Use single holdout.")


def train_full_and_predict(train: pd.DataFrame, test: pd.DataFrame, info: DataInfo) -> pd.DataFrame:
	clf, X_train_all, X_test_prepared = train_and_eval(train, test, info)
	# Use the model trained with early stopping on 90% data for fast inference.
	proba = clf.predict_proba(X_test_prepared)[:, 1]
	sub = pd.DataFrame({info.id_col: test[info.id_col], "y": proba})
	return sub


def main() -> None:
	print("Loading data…")
	train, test = load_data()
	info = infer_schema(train)
	print(f"Detected {len(info.num_features)} numeric and {len(info.cat_features)} categorical features.")

	print("Training full model and predicting…")
	sub = train_full_and_predict(train, test, info)

	# Align with sample submission if available (order and columns)
	if os.path.exists(SAMPLE_SUB_PATH):
		sample = pd.read_csv(SAMPLE_SUB_PATH)
		sub = sample[["id"]].merge(sub, on="id", how="left")
		# Fill any missing with median probability for safety
		if sub["y"].isna().any():
			sub["y"] = sub["y"].fillna(sub["y"].median())

	sub.to_csv(OUT_SUB_PATH, index=False)
	print(f"Wrote submission to {OUT_SUB_PATH}")


if __name__ == "__main__":
	main()

