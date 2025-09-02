from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	"""
	Detect categorical (object/category/bool) and numeric columns.

	Returns (categorical_cols, numeric_cols)
	"""
	cat_cols = [
		c
		for c in df.columns
		if str(df[c].dtype) in ("object", "category", "bool")
	]
	num_cols = [c for c in df.columns if c not in cat_cols]
	return cat_cols, num_cols


def _prep_categoricals(train: pd.DataFrame, test: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
	"""
	Convert categorical columns to pandas 'category' dtype, aligning categories across train/test.
	"""
	aligned_cols: List[str] = []
	for col in cat_cols:
		# Ensure string dtype for consistent categories (avoid mixed types)
		tr = train[col].astype("string")
		te = test[col].astype("string")

		# Union categories across train/test
		cats = pd.Categorical(pd.concat([tr, te], axis=0, ignore_index=True))
		categories = cats.categories

		train[col] = pd.Categorical(tr, categories=categories)
		test[col] = pd.Categorical(te, categories=categories)
		aligned_cols.append(col)
	return train, test, aligned_cols


@dataclass
class Config:
	data_dir: str = "data"
	train_file: str = "train.csv"
	test_file: str = "test.csv"
	target: str = "y"
	id_col: str = "id"
	n_splits: int = 3  # keep it light by default
	random_state: int = 42
	output_file: str = "submission.csv"


def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_path = os.path.join(cfg.data_dir, cfg.train_file)
	test_path = os.path.join(cfg.data_dir, cfg.test_file)

	if not os.path.exists(train_path):
		raise FileNotFoundError(f"Missing train file: {train_path}")
	if not os.path.exists(test_path):
		raise FileNotFoundError(f"Missing test file: {test_path}")

	# Low memory off to avoid mixed dtypes in wide CSVs
	train = pd.read_csv(train_path, low_memory=False)
	test = pd.read_csv(test_path, low_memory=False)

	# Validate expected columns
	if cfg.target not in train.columns:
		raise KeyError(f"Target column '{cfg.target}' not found in train data")
	if cfg.id_col not in test.columns:
		raise KeyError(f"ID column '{cfg.id_col}' not found in test data")

	return train, test


def train_and_predict(cfg: Config) -> pd.DataFrame:
	import lightgbm as lgb

	train, test = load_data(cfg)

	y = train[cfg.target].astype(int).values
	features = [c for c in train.columns if c not in {cfg.target}]

	# Keep id if present in train, but do not train on it
	if cfg.id_col in features:
		features.remove(cfg.id_col)

	X = train[features].copy()
	X_test = test[features].copy()

	cat_cols, _ = _detect_columns(X)
	if cat_cols:
		X, X_test, cat_cols = _prep_categoricals(X, X_test, cat_cols)

	# LightGBM supports categorical features when dtype is category
	cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

	# Fast mode toggle: set FAST=1 in env to do a single stratified holdout with fewer trees
	fast_mode = os.environ.get("FAST", "0") == "1"
	n_splits = cfg.n_splits
	if not fast_mode:
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

	oof = np.zeros(len(X), dtype=float)
	preds = np.zeros(len(X_test), dtype=float)

	# Model configuration: conservative defaults, early stopping enabled
	if fast_mode:
		lrate = 0.05
		n_estimators = 600
		early_rounds = 50
		num_leaves = 31
	else:
		lrate = 0.03
		n_estimators = 1500
		early_rounds = 100
		num_leaves = 48

	base_params = dict(
		objective="binary",
		metric="auc",
		learning_rate=lrate,
		n_estimators=n_estimators,
		num_leaves=num_leaves,
		max_depth=-1,
		subsample=0.8,
		colsample_bytree=0.8,
		reg_alpha=0.1,
		reg_lambda=0.1,
		random_state=cfg.random_state,
		n_jobs=max(1, (os.cpu_count() or 2) - 1),  # leave a core
		verbose=-1,
	)

	if fast_mode:
		# Single holdout split
		X_tr, X_va, y_tr, y_va = train_test_split(
			X, y, test_size=0.1, random_state=cfg.random_state, stratify=y
		)
		model = lgb.LGBMClassifier(**base_params)
		model.fit(
			X_tr,
			y_tr,
			eval_set=[(X_va, y_va)],
			eval_metric="auc",
			categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
			callbacks=[
				lgb.early_stopping(stopping_rounds=early_rounds, verbose=False),
				lgb.log_evaluation(period=0 if fast_mode else 200),
			],
		)
		va_pred = model.predict_proba(X_va)[:, 1]
		fold_auc = roc_auc_score(y_va, va_pred)
		print(f"Holdout AUC: {fold_auc:.5f}  |  Best iter: {getattr(model, 'best_iteration_', None)}")
		preds += model.predict_proba(X_test)[:, 1]
	else:
		for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
			X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
			y_tr, y_va = y[tr_idx], y[va_idx]

			model = lgb.LGBMClassifier(**base_params)

			model.fit(
				X_tr,
				y_tr,
				eval_set=[(X_va, y_va)],
				eval_metric="auc",
				categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
				callbacks=[
					lgb.early_stopping(stopping_rounds=early_rounds, verbose=False),
					lgb.log_evaluation(period=0 if fast_mode else 200),
				],
			)

			va_pred = model.predict_proba(X_va)[:, 1]
			oof[va_idx] = va_pred
			preds += model.predict_proba(X_test)[:, 1] / n_splits

			fold_auc = roc_auc_score(y_va, va_pred)
			print(f"Fold {fold}/{n_splits} AUC: {fold_auc:.5f}  |  Best iter: {getattr(model, 'best_iteration_', None)}")

	if not fast_mode and n_splits > 1:
		oof_auc = roc_auc_score(y, oof)
		print(f"OOF AUC: {oof_auc:.6f}")

	submission = pd.DataFrame({
		cfg.id_col: test[cfg.id_col].values,
		cfg.target: np.clip(preds, 1e-6, 1 - 1e-6),
	})
	return submission


def main() -> None:
	cfg = Config()
	submission = train_and_predict(cfg)
	out_path = cfg.output_file
	submission.to_csv(out_path, index=False)
	print(f"Wrote submission to: {os.path.abspath(out_path)} | shape={submission.shape}")


if __name__ == "__main__":
	main()

