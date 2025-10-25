import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import pandas as pd

from src.utils.helpers import get_logger, load_config


class Predictor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.default_input = Path(self.config["data"]["master_features"])

    def run(
        self,
        model_path: Path,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        artifact = joblib.load(model_path)
        features = artifact["features"]

        data_path = input_path or self.default_input
        df = pd.read_csv(data_path)
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise KeyError(
                f"Input data missing required features: {', '.join(missing)}"
            )

        X = df[features].apply(pd.to_numeric, errors="coerce")
        predictions = artifact["model"].predict(X)

        result_df = pd.DataFrame(
            {
                "subject_id": df["subject_id"],
                "prediction": predictions,
                "target": artifact.get("target"),
                "feature_set": artifact.get("feature_set"),
                "algorithm": artifact.get("algorithm"),
            }
        )

        if output_path is None:
            target = artifact.get("target")
            feature_set = artifact.get("feature_set")
            algo = artifact.get("algorithm")
            output_path = (
                Path(self.config["outputs"]["results_dir"])
                / f"predictions_{target}_{feature_set}_{algo}_full.csv"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        self.logger.info("Predictions saved to %s", output_path)
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions.")
    parser.add_argument("--config", default="config.yaml", help="Config file.")
    parser.add_argument(
        "--model-path", required=True, help="Path to saved model artifact (.pkl)."
    )
    parser.add_argument(
        "--input-path", default=None, help="Optional override for feature table."
    )
    parser.add_argument(
        "--output-path", default=None, help="Optional override for predictions CSV."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(args.config)
    predictor.run(
        model_path=Path(args.model_path),
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()
