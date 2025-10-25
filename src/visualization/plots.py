import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.helpers import ensure_dir, get_logger, load_config


class VisualizationBuilder:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.results_dir = Path(self.config["outputs"]["results_dir"])
        self.figures_dir = ensure_dir(Path(self.config["outputs"]["figures_dir"]))
        self.palette = self.config["visualization"]["color_palette"]
        self.dpi = self.config["visualization"]["figure_dpi"]
        self.top_k = self.config["visualization"]["top_features"]

    def run(self) -> None:
        performance_path = self.results_dir / "model_performance.csv"
        importance_path = self.results_dir / "feature_importances.csv"
        if not performance_path.exists() or not importance_path.exists():
            raise FileNotFoundError(
                "Model performance / feature importance results not found. Run training first."
            )

        performance_df = pd.read_csv(performance_path)
        importance_df = pd.read_csv(importance_path)

        self._plot_model_performance(performance_df)
        self._plot_feature_importance(performance_df, importance_df)
        self._plot_predictions(performance_df)
        self._plot_brain_heatmap(performance_df, importance_df)

    # ------------------------------------------------------------------ #

    def _plot_model_performance(self, df: pd.DataFrame) -> None:
        df = df.sort_values("r2", ascending=False)
        labels = df.apply(
            lambda row: f"{row['target']} | {row['feature_set']} | {row['algorithm']}",
            axis=1,
        )
        colors = [self.palette[i % len(self.palette)] for i in range(len(df))]

        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
        ax.barh(labels, df["r2"], color=colors)
        ax.set_xlabel("R²")
        ax.set_title("Model Performance (R²)")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = self.figures_dir / "model_performance_r2.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.logger.info("Saved model performance plot to %s", path)

    def _best_combinations(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        best = {}
        for target, group in df.groupby("target"):
            best[target] = group.sort_values("r2", ascending=False).iloc[0]
        return best

    def _plot_feature_importance(
        self, performance_df: pd.DataFrame, importance_df: pd.DataFrame
    ) -> None:
        best = self._best_combinations(performance_df)
        for target, row in best.items():
            subset = importance_df[
                (importance_df["target"] == target)
                & (importance_df["feature_set"] == row["feature_set"])
                & (importance_df["algorithm"] == row["algorithm"])
            ].sort_values("importance", ascending=False)

            if subset.empty:
                continue

            top = subset.head(self.top_k)
            fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.3)))
            ax.barh(top["feature"], top["importance"], color=self.palette[0])
            ax.set_title(
                f"Top {len(top)} Feature Importances ({target} | {row['feature_set']} | {row['algorithm']})"
            )
            ax.set_xlabel("Importance (normalized units)")
            ax.invert_yaxis()
            fig.tight_layout()

            path = self.figures_dir / f"feature_importance_{target}.png"
            fig.savefig(path, dpi=self.dpi)
            plt.close(fig)
            self.logger.info("Saved feature importance plot for %s to %s", target, path)

    def _plot_predictions(self, performance_df: pd.DataFrame) -> None:
        best = self._best_combinations(performance_df)
        for target, row in best.items():
            pred_path = (
                self.results_dir
                / f"predictions_{target}_{row['feature_set']}_{row['algorithm']}.csv"
            )
            if not pred_path.exists():
                continue
            pred_df = pd.read_csv(pred_path)
            if pred_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                pred_df["y_true"],
                pred_df["y_pred"],
                alpha=self.config["visualization"]["scatter_alpha"],
                color=self.palette[1],
                edgecolor="k",
            )
            lims = [
                min(pred_df["y_true"].min(), pred_df["y_pred"].min()),
                max(pred_df["y_true"].max(), pred_df["y_pred"].max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_xlabel("Actual score")
            ax.set_ylabel("Predicted score")
            ax.set_title(
                f"Actual vs Predicted ({target} | {row['feature_set']} | {row['algorithm']})"
            )
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.grid(alpha=0.3)
            fig.tight_layout()

            path = self.figures_dir / f"actual_vs_pred_{target}.png"
            fig.savefig(path, dpi=self.dpi)
            plt.close(fig)
            self.logger.info("Saved prediction scatter plot for %s to %s", target, path)

    def _plot_brain_heatmap(
        self, performance_df: pd.DataFrame, importance_df: pd.DataFrame
    ) -> None:
        best = self._best_combinations(performance_df)
        heatmap_rows = []
        for target, row in best.items():
            subset = importance_df[
                (importance_df["target"] == target)
                & (importance_df["feature_set"] == row["feature_set"])
                & (importance_df["algorithm"] == row["algorithm"])
                & (importance_df["feature"].str.contains("region_"))
            ]
            if subset.empty:
                continue
            subset = subset.groupby("feature")["importance"].max().reset_index()
            subset["target"] = target
            heatmap_rows.append(subset)

        if not heatmap_rows:
            self.logger.warning("No regional features available for heatmap.")
            return

        combined = pd.concat(heatmap_rows)
        pivot = combined.pivot(index="target", columns="feature", values="importance").fillna(0)

        fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.4), 4))
        cax = ax.imshow(pivot.values, aspect="auto", cmap="magma")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=90)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Regional Feature Importance Heatmap")
        fig.colorbar(cax, ax=ax, label="Relative importance")
        fig.tight_layout()

        path = self.figures_dir / "regional_importance_heatmap.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.logger.info("Saved heatmap to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualizations.")
    parser.add_argument("--config", default="config.yaml", help="Config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = VisualizationBuilder(args.config)
    builder.run()


if __name__ == "__main__":
    main()
