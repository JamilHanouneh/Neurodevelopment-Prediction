import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, get_logger, load_config, save_json


class FeatureBuilder:
    """Combine imaging features with clinical metadata and generate proxy targets."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.processed_dir = Path(self.config["data"]["processed_dir"])
        self.aggregated_path = Path(self.config["data"]["aggregated_features"])
        self.master_path = Path(self.config["data"]["master_features"])
        self.catalog_path = Path(self.config["data"]["feature_catalog"])
        self.meta_filename = self.config["data"]["meta_filename"]
        self.rng = np.random.default_rng(self.config["project"]["seed"])

    def run(self) -> pd.DataFrame:
        imaging_df = self._load_imaging_features()
        meta_df = self._load_metadata()
        merged = meta_df.merge(imaging_df, on="subject_id", how="inner")
        if merged.empty:
            raise RuntimeError("No overlap between metadata and imaging features.")

        merged = self._encode_clinical_features(merged)
        merged = self._generate_proxy_scores(merged)
        merged = merged.sort_values("subject_id").reset_index(drop=True)

        ensure_dir(self.master_path.parent)
        merged.to_csv(self.master_path, index=False)
        self.logger.info(
            "Master feature table with %d subjects and %d features saved to %s",
            len(merged),
            merged.shape[1],
            self.master_path,
        )

        catalog = self._build_feature_catalog(merged)
        save_json(catalog, self.catalog_path)
        self.logger.info("Feature catalog saved to %s", self.catalog_path)

        return merged

    # ------------------------------------------------------------------ #

    def _load_imaging_features(self) -> pd.DataFrame:
        if not self.aggregated_path.exists():
            raise FileNotFoundError(
                "Preprocessed feature file not found. Run preprocess.py first."
            )
        df = pd.read_csv(self.aggregated_path)
        if "subject_id" not in df.columns:
            raise ValueError("subject_id column missing from imaging features.")
        return df

    def _load_metadata(self) -> pd.DataFrame:
        meta_path = self.raw_dir / self.meta_filename
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta = pd.read_csv(meta_path, sep=";")
        meta.rename(columns={"image_id": "subject_id"}, inplace=True)
        meta["subject_id"] = meta["subject_id"].str.strip()
        meta["myelinisation"] = meta["myelinisation"].str.lower().str.strip()
        return meta

    def _encode_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["myelinisation_encoded"] = df["myelinisation"].map(
            {"delayed": 0, "normal": 1, "accelerated": 2}
        )

        for col in self.config["features"]["clinical"]["numerical"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        categorical_cols = [
            col
            for col in self.config["features"]["clinical"]["categorical"]
            if col in df.columns
        ]
        if categorical_cols:
            encoded = pd.get_dummies(
                df[categorical_cols], prefix=categorical_cols, dummy_na=False
            )
            df = pd.concat([df, encoded], axis=1)

        df["diagnosis_encoded"] = (
            pd.Categorical(df.get("diagnosis"))
            .codes.astype(float)
        )
        df["group_encoded"] = (
            pd.Categorical(df.get("group"))
            .codes.astype(float)
        )
        return df

    def _generate_proxy_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        proxy_cfg = self.config["features"]["proxy_scores"]
        base = df["myelinisation"].map(proxy_cfg["baseline"]).astype(float)
        base = base.fillna(proxy_cfg["baseline"]["normal"])
        age_column = "age_corrected" if "age_corrected" in df.columns else "age"
        age_series = df[age_column].fillna(df[age_column].median())
        age_adjustment = (age_series - age_series.mean()) * proxy_cfg["age_scale"]

        def _compose(offset: float) -> np.ndarray:
            noise = self.rng.normal(0, proxy_cfg["noise_std"], size=len(df))
            score = base + age_adjustment + noise + offset
            return np.clip(score, 55, 135)

        df["cognitive_score"] = _compose(offset=0.0)
        df["motor_score"] = _compose(offset=-2.0)
        df["language_score"] = _compose(offset=1.5)
        return df

    def _build_feature_catalog(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        exclude = {"subject_id", "myelinisation"}
        exclude.update(self.config["features"]["target_variables"])
        catalog: Dict[str, List[str]] = {
            "volumetric": [],
            "morphometric": [],
            "clinical": [],
        }

        volumetric_keywords = ["volume", "hemisphere", "gm_", "wm_", "csf_"]
        volumetric_ratio_names = {
            "gm_wm_ratio",
            "csf_brain_ratio",
            "anterior_posterior_ratio",
            "superior_inferior_ratio",
        }

        for col in df.columns:
            if col in exclude:
                continue
            if any(keyword in col for keyword in volumetric_keywords) or col in volumetric_ratio_names:
                catalog["volumetric"].append(col)
            elif any(
                key in col
                for key in [
                    "surface",
                    "thickness",
                    "region_",
                    "compactness",
                    "intensity",
                    "asymmetry",
                ]
            ):
                catalog["morphometric"].append(col)
            elif (
                col in self.config["features"]["clinical"]["numerical"]
                or col.endswith("_encoded")
                or col.startswith("group_")
                or col.startswith("diagnosis_")
            ):
                catalog["clinical"].append(col)
            elif col.startswith("t1_") or col.startswith("t2_"):
                catalog["morphometric"].append(col)

        for key in catalog:
            catalog[key] = sorted(set(catalog[key]))

        catalog["combined"] = sorted(
            set().union(*[set(vals) for vals in catalog.values()])
        )

        return catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine preprocessing features with clinical metadata."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file."
    )
    args = parser.parse_args()
    builder = FeatureBuilder(args.config)
    builder.run()


if __name__ == "__main__":
    main()
