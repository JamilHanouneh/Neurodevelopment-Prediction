import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_metrics
from src.utils.helpers import (
    ensure_dir,
    get_logger,
    load_config,
    load_json,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.master_path = Path(self.config["data"]["master_features"])
        self.catalog_path = Path(self.config["data"]["feature_catalog"])
        self.models_dir = ensure_dir(Path(self.config["outputs"]["models_dir"]))
        self.results_dir = ensure_dir(Path(self.config["outputs"]["results_dir"]))
        self.seed = self.config["project"]["seed"]

    def run(
        self,
        targets: Optional[List[str]] = None,
        feature_sets: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
    ) -> None:
        df = pd.read_csv(self.master_path)
        catalog = load_json(self.catalog_path)

        targets = targets or self.config["features"]["target_variables"]
        feature_sets = feature_sets or self.config["models"]["feature_sets"]
        available_algorithms = self.config["models"]["algorithms"]
        algo_keys = algorithms or list(available_algorithms.keys())

        results = []
        importance_rows = []

        for target in targets:
            if target not in df.columns:
                self.logger.warning("Target %s missing in dataset. Skipping.", target)
                continue

            for feature_set in feature_sets:
                feature_cols = self._select_features(catalog, feature_set, df.columns)
                if not feature_cols:
                    self.logger.warning(
                        "No features found for set %s. Skipping.", feature_set
                    )
                    continue

                X, subject_ids = self._prepare_features(df, feature_cols)
                y = df[target].astype(float)
                split_ratio = self.config["models"]["test_size"]
                if len(df) * split_ratio < 1:
                    split_ratio = max(1 / len(df), split_ratio)

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    id_train,
                    id_test,
                ) = train_test_split(
                    X,
                    y,
                    subject_ids,
                    test_size=split_ratio,
                    random_state=self.seed,
                    shuffle=True,
                )

                for algo_name in algo_keys:
                    algo_cfg = available_algorithms.get(algo_name)
                    if algo_cfg is None:
                        self.logger.warning("Algorithm %s not configured.", algo_name)
                        continue
                    if algo_cfg["type"] == "xgboost" and not XGBOOST_AVAILABLE:
                        self.logger.warning("xgboost not available. Skipping.")
                        continue

                    try:
                        (
                            best_model,
                            best_params,
                            metrics_dict,
                            predictions,
                            feature_importance,
                        ) = self._train_single(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            feature_cols,
                            algo_name,
                            algo_cfg,
                        )
                    except ValueError as err:
                        self.logger.warning(
                            "Skipping %s/%s/%s due to: %s",
                            target,
                            feature_set,
                            algo_name,
                            err,
                        )
                        continue

                    model_path = (
                        self.models_dir
                        / f"{target}_{feature_set}_{algo_name}_model.pkl"
                    )
                    artifact = {
                        "model": best_model,
                        "features": feature_cols,
                        "target": target,
                        "feature_set": feature_set,
                        "params": best_params,
                        "algorithm": algo_name,
                    }
                    joblib.dump(artifact, model_path)

                    metrics_record = {
                        "target": target,
                        "feature_set": feature_set,
                        "algorithm": algo_name,
                        "n_features": len(feature_cols),
                        **metrics_dict,
                        "model_path": str(model_path),
                    }
                    results.append(metrics_record)

                    pred_df = pd.DataFrame(
                        {
                            "subject_id": id_test,
                            "y_true": y_test,
                            "y_pred": predictions,
                            "target": target,
                            "feature_set": feature_set,
                            "algorithm": algo_name,
                        }
                    )
                    pred_path = (
                        self.results_dir
                        / f"predictions_{target}_{feature_set}_{algo_name}.csv"
                    )
                    pred_df.to_csv(pred_path, index=False)

                    for feat, importance in feature_importance:
                        importance_rows.append(
                            {
                                "target": target,
                                "feature_set": feature_set,
                                "algorithm": algo_name,
                                "feature": feat,
                                "importance": importance,
                            }
                        )

                    self.logger.info(
                        "%s | %s | %s | R2=%.3f RMSE=%.2f",
                        target,
                        feature_set,
                        algo_name,
                        metrics_dict.get("r2", float("nan")),
                        metrics_dict.get("rmse", float("nan")),
                    )

        if results:
            results_df = pd.DataFrame(results)
            results_path = self.results_dir / "model_performance.csv"
            results_df.to_csv(results_path, index=False)
            self.logger.info("Saved model performance summary to %s", results_path)

        if importance_rows:
            importance_df = pd.DataFrame(importance_rows)
            importance_path = self.results_dir / "feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            self.logger.info("Saved feature importances to %s", importance_path)

    # ------------------------------------------------------------------ #

    def _select_features(
        self, catalog: Dict[str, List[str]], feature_set: str, available_columns
    ) -> List[str]:
        cols = catalog.get(feature_set, [])
        filtered = [col for col in cols if col in available_columns]
        return filtered

    def _prepare_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        subject_ids = df["subject_id"]
        return X, subject_ids

    def _build_pipeline(self, algo_cfg: Dict) -> Tuple[Pipeline, Dict]:
        algo_type = algo_cfg["type"]
        steps = [("imputer", SimpleImputer(strategy="median"))]

        if algo_type == "ridge":
            estimator = Ridge(random_state=self.seed)
            steps.append(("scaler", StandardScaler()))
        elif algo_type == "random_forest":
            estimator = RandomForestRegressor(
                random_state=self.seed, n_jobs=self.config["models"]["n_jobs"]
            )
        elif algo_type == "gradient_boosting":
            estimator = GradientBoostingRegressor(random_state=self.seed)
        elif algo_type == "xgboost":
            estimator = XGBRegressor(
                random_state=self.seed,
                objective="reg:squarederror",
                tree_method="hist",
                n_estimators=200,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo_type}")

        steps.append(("model", estimator))
        pipeline = Pipeline(steps)

        param_grid = {f"model__{k}": v for k, v in algo_cfg["param_grid"].items()}
        return pipeline, param_grid

    def _train_single(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols: List[str],
        algo_name: str,
        algo_cfg: Dict,
    ):
        pipeline, param_grid = self._build_pipeline(algo_cfg)
        cv_folds = min(self.config["models"]["cv_folds"], len(X_train))
        if cv_folds < 2:
            raise ValueError("Insufficient samples for cross-validation.")

        cv = KFold(
            n_splits=cv_folds, shuffle=True, random_state=self.config["project"]["seed"]
        )

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=self.config["models"]["n_jobs"],
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        predictions = best_model.predict(X_test)

        metrics_dict = compute_metrics(
            y_test, predictions, self.config["evaluation"]["metrics"]
        )
        feature_importance = self._extract_feature_importance(best_model, feature_cols)

        return best_model, grid.best_params_, metrics_dict, predictions, feature_importance

    def _extract_feature_importance(
        self, pipeline: Pipeline, feature_cols: List[str]
    ) -> List[Tuple[str, float]]:
        estimator = pipeline.named_steps["model"]
        importance: List[Tuple[str, float]] = []

        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_)
        else:
            return []

        for feature, value in zip(feature_cols, values):
            importance.append((feature, float(value)))

        importance.sort(key=lambda x: x[1], reverse=True)
        return importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument("--config", default="config.yaml", help="Config path.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Subset of targets to train.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        help="Subset of feature sets to use.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Subset of algorithms to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ModelTrainer(args.config)
    trainer.run(targets=args.targets, feature_sets=args.feature_sets, algorithms=args.algorithms)


if __name__ == "__main__":
    main()
