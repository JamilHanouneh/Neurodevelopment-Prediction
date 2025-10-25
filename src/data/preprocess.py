import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import measure

from src.utils.helpers import ensure_dir, get_logger, list_subjects, load_config, save_json


class InfantMRIPreprocessor:
    """Pipeline that loads subject MRIs and computes volumetric + morphometric features."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.features_dir = ensure_dir(Path(self.config["data"]["features_dir"]))
        self.aggregated_path = Path(self.config["data"]["aggregated_features"])
        self.meta_filename = self.config["data"]["meta_filename"]
        self.subject_prefix = self.config["data"]["subject_prefix"]
        self.allowed_ext = tuple(self.config["data"]["allowed_extensions"])
        self.random_state = self.config["project"]["seed"]
        self.rng = np.random.default_rng(self.random_state)

    def run(self, limit_subjects: Optional[int] = None) -> pd.DataFrame:
        """Process subjects and return dataframe with extracted features."""
        subject_dirs = list_subjects(
            self.raw_dir,
            self.subject_prefix,
            max_subjects=limit_subjects
            or self.config["data"].get("max_subjects"),
        )
        if not subject_dirs:
            raise FileNotFoundError(
                f"No subject folders found under {self.raw_dir.resolve()}"
            )

        all_rows: List[Dict] = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            try:
                features = self._process_subject(subject_id, subject_dir)
                all_rows.append(features)
                self._write_subject_features(subject_id, features)
            except FileNotFoundError as err:
                self.logger.warning("Skipping %s: %s", subject_id, err)
            except Exception as err:  # pragma: no cover - guard rail
                self.logger.error("Failed on %s: %s", subject_id, err)

        if not all_rows:
            raise RuntimeError("No subjects processed successfully.")

        df = pd.DataFrame(all_rows)
        ensure_dir(self.aggregated_path.parent)
        df.to_csv(self.aggregated_path, index=False)
        self.logger.info(
            "Saved aggregated preprocessing features for %d subjects to %s",
            len(df),
            self.aggregated_path,
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_subject(self, subject_id: str, subject_dir: Path) -> Dict:
        t1_path = self._find_modality(subject_dir, "t1")
        t2_path = self._find_modality(subject_dir, "t2")
        if t1_path is None or t2_path is None:
            raise FileNotFoundError("Missing T1 or T2 file")

        t1_img, t1_data = self._load_nifti(t1_path)
        _, t2_data = self._load_nifti(t2_path)

        t1_norm = self._normalize_volume(t1_data)
        t2_norm = self._normalize_volume(t2_data)

        brain_mask = self._compute_brain_mask(t1_data)
        voxel_dims = t1_img.header.get_zooms()[:3]
        voxel_volume = float(np.prod(voxel_dims))

        tissues = self._segment_tissues(t1_norm, brain_mask)
        volumetric = self._compute_volumetrics(
            brain_mask, tissues, voxel_volume, subject_id
        )
        morphometric = self._compute_morphometrics(
            brain_mask, tissues["gray"], voxel_dims, volumetric
        )
        regional = self._compute_regional_intensity_stats(
            t1_norm, t2_norm, brain_mask, subject_id
        )

        intensity_stats = self._compute_intensity_stats(t1_norm, t2_norm, brain_mask)

        subject_features = {
            "subject_id": subject_id,
            "voxel_volume_mm3": voxel_volume,
            **volumetric,
            **morphometric,
            **regional,
            **intensity_stats,
        }

        if self.config["preprocessing"].get("save_normalized_volumes", False):
            self._save_normalized_volume(subject_id, t1_norm, t1_img.affine, "t1")
            self._save_normalized_volume(subject_id, t2_norm, t1_img.affine, "t2")

        return subject_features

    def _save_normalized_volume(
        self, subject_id: str, data: np.ndarray, affine: np.ndarray, modality: str
    ) -> None:
        out_dir = ensure_dir(self.features_dir / subject_id)
        img = nib.Nifti1Image(data.astype(np.float32), affine=affine)
        nib.save(img, out_dir / f"{subject_id}_{modality}_normalized.nii.gz")

    def _write_subject_features(self, subject_id: str, features: Dict) -> None:
        """Persist per-subject feature summary as CSV + JSON for transparency."""
        subject_frame = pd.DataFrame([features])
        csv_path = self.features_dir / f"{subject_id}_features.csv"
        subject_frame.to_csv(csv_path, index=False)
        json_path = self.features_dir / f"{subject_id}_features.json"
        save_json(features, json_path)

    def _find_modality(self, subject_dir: Path, modality: str) -> Optional[Path]:
        """Locate file path for a modality (t1/t2)."""
        candidates = [
            subject_dir / f"{modality}.nii",
            subject_dir / f"{modality}.nii.gz",
            subject_dir / f"{modality}w.nii",
            subject_dir / f"{modality}w.nii.gz",
        ]
        def _matches_extension(path: Path) -> bool:
            return any(str(path).lower().endswith(ext) for ext in self.allowed_ext)

        for cand in candidates:
            if cand.exists() and _matches_extension(cand):
                return cand
        for cand in subject_dir.glob(f"*{modality}*.nii*"):
            if _matches_extension(cand):
                return cand
        return None

    @staticmethod
    def _load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
        img = nib.load(str(path))
        data = np.asanyarray(img.get_fdata(), dtype=np.float32)
        return img, data

    def _normalize_volume(self, data: np.ndarray) -> np.ndarray:
        clip_low, clip_high = self.config["preprocessing"]["intensity_clip_percentiles"]
        positive = data[data > 0]
        if positive.size < 10:
            positive = data.reshape(-1)
        lower, upper = np.percentile(positive, [clip_low, clip_high])
        clipped = np.clip(data, lower, upper)
        valid = clipped[np.abs(clipped) > 1e-6]
        if valid.size == 0:
            valid = clipped.reshape(-1)
        mean = valid.mean()
        std = valid.std()
        if std == 0:
            std = 1.0
        normalized = (clipped - mean) / std
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized.astype(np.float32)

    def _compute_brain_mask(self, data: np.ndarray) -> np.ndarray:
        mask = data > 0
        if mask.sum() == 0:
            threshold = np.percentile(data, 60)
            mask = data > threshold
        mask = ndimage.binary_fill_holes(mask)
        return mask

    def _segment_tissues(
        self, normalized_volume: np.ndarray, mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        voxels = normalized_volume[mask].reshape(-1, 1)
        if voxels.size < 10:
            blank = np.zeros_like(mask, dtype=bool)
            return {"gray": blank, "white": blank, "csf": blank}

        clusters = self.config["preprocessing"]["kmeans_clusters"]
        best_labels = None
        min_inertia = np.inf
        n_init = max(5, self.config["preprocessing"]["kmeans_repeats"])
        for attempt in range(self.config["preprocessing"]["kmeans_repeats"]):
            model = KMeans(
                n_clusters=clusters,
                random_state=self.random_state + attempt,
                n_init=n_init,
            )
            model.fit(voxels)
            if model.inertia_ < min_inertia:
                min_inertia = model.inertia_
                best_labels = model.labels_
                centers = model.cluster_centers_.flatten()

        if best_labels is None:
            raise RuntimeError("Failed to segment tissues with KMeans.")

        order = np.argsort(centers)
        csf_idx, gray_idx, white_idx = order
        label_volume = np.zeros_like(mask, dtype=np.uint8)
        label_volume[mask] = best_labels + 1

        return {
            "csf": label_volume == (csf_idx + 1),
            "gray": label_volume == (gray_idx + 1),
            "white": label_volume == (white_idx + 1),
        }

    def _compute_volumetrics(
        self,
        mask: np.ndarray,
        tissues: Dict[str, np.ndarray],
        voxel_volume: float,
        subject_id: str,
    ) -> Dict[str, float]:
        volumes = {}
        brain_voxels = mask.sum()
        volumes["brain_volume_mm3"] = brain_voxels * voxel_volume

        for tissue_name, tissue_mask in tissues.items():
            volumes[f"{tissue_name}_volume_mm3"] = float(
                tissue_mask.sum() * voxel_volume
            )

        volumes["gm_wm_ratio"] = self._safe_ratio(
            volumes["gray_volume_mm3"], volumes["white_volume_mm3"]
        )
        volumes["csf_brain_ratio"] = self._safe_ratio(
            volumes["csf_volume_mm3"], volumes["brain_volume_mm3"]
        )

        x_mid = mask.shape[0] // 2
        left = mask[:x_mid]
        right = mask[x_mid:]
        volumes["left_hemisphere_volume_mm3"] = float(left.sum() * voxel_volume)
        volumes["right_hemisphere_volume_mm3"] = float(right.sum() * voxel_volume)
        volumes["hemispheric_asymmetry_index"] = self._safe_ratio(
            volumes["left_hemisphere_volume_mm3"] - volumes["right_hemisphere_volume_mm3"],
            volumes["left_hemisphere_volume_mm3"] + volumes["right_hemisphere_volume_mm3"],
        )

        anterior = mask[:, : mask.shape[1] // 2, :]
        posterior = mask[:, mask.shape[1] // 2 :, :]
        volumes["anterior_posterior_ratio"] = self._safe_ratio(
            anterior.sum(), posterior.sum()
        )

        superior = mask[:, :, mask.shape[2] // 2 :]
        inferior = mask[:, :, : mask.shape[2] // 2]
        volumes["superior_inferior_ratio"] = self._safe_ratio(
            superior.sum(), inferior.sum()
        )

        return volumes

    def _compute_morphometrics(
        self,
        brain_mask: np.ndarray,
        gray_mask: np.ndarray,
        voxel_dims: Tuple[float, float, float],
        volumetric: Dict[str, float],
    ) -> Dict[str, float]:
        morpho: Dict[str, float] = {}
        if brain_mask.sum() == 0:
            return morpho

        try:
            verts, faces, _, _ = measure.marching_cubes(
                brain_mask.astype(np.float32), level=0.5, spacing=voxel_dims
            )
            surface_area = measure.mesh_surface_area(verts, faces)
        except ValueError:
            # Fallback: approximate via boundary voxels
            eroded = ndimage.binary_erosion(brain_mask)
            boundary = brain_mask & (~eroded)
            face_area = (
                voxel_dims[0] * voxel_dims[1]
                + voxel_dims[1] * voxel_dims[2]
                + voxel_dims[0] * voxel_dims[2]
            )
            surface_area = boundary.sum() * face_area

        morpho["surface_area_mm2"] = float(surface_area)
        brain_volume = volumetric.get("brain_volume_mm3", 0.0)
        morpho["volume_surface_ratio"] = self._safe_ratio(
            brain_volume, surface_area
        )

        gray_distances = ndimage.distance_transform_edt(
            gray_mask, sampling=voxel_dims
        )
        gray_vals = gray_distances[gray_mask]
        if gray_vals.size > 0:
            thickness = gray_vals * 2.0
            morpho["cortical_thickness_mean_mm"] = float(thickness.mean())
            morpho["cortical_thickness_std_mm"] = float(thickness.std())
        else:
            morpho["cortical_thickness_mean_mm"] = 0.0
            morpho["cortical_thickness_std_mm"] = 0.0

        morpho["fractal_compactness"] = self._safe_ratio(
            surface_area**1.5, brain_volume
        )
        return morpho

    def _compute_regional_intensity_stats(
        self,
        t1_norm: np.ndarray,
        t2_norm: np.ndarray,
        mask: np.ndarray,
        subject_id: str,
    ) -> Dict[str, float]:
        grid = self.config["preprocessing"]["regional_grid"]
        x_bins = np.linspace(0, t1_norm.shape[0], grid[0] + 1, dtype=int)
        y_bins = np.linspace(0, t1_norm.shape[1], grid[1] + 1, dtype=int)
        z_bins = np.linspace(0, t1_norm.shape[2], grid[2] + 1, dtype=int)

        features: Dict[str, float] = {}
        region_idx = 0
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    xs = slice(x_bins[i], x_bins[i + 1])
                    ys = slice(y_bins[j], y_bins[j + 1])
                    zs = slice(z_bins[k], z_bins[k + 1])
                    region_mask = mask[xs, ys, zs]
                    if region_mask.sum() == 0:
                        region_idx += 1
                        continue

                    t1_vals = t1_norm[xs, ys, zs][region_mask]
                    t2_vals = t2_norm[xs, ys, zs][region_mask]
                    features[
                        f"region_{region_idx:02d}_t1_mean"
                    ] = float(np.mean(t1_vals))
                    features[
                        f"region_{region_idx:02d}_t2_mean"
                    ] = float(np.mean(t2_vals))
                    features[
                        f"region_{region_idx:02d}_t1_std"
                    ] = float(np.std(t1_vals))
                    features[
                        f"region_{region_idx:02d}_t2_std"
                    ] = float(np.std(t2_vals))
                    region_idx += 1
        return features

    def _compute_intensity_stats(
        self, t1_norm: np.ndarray, t2_norm: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        masked_t1 = t1_norm[mask]
        masked_t2 = t2_norm[mask]
        ratio = np.divide(
            masked_t1,
            masked_t2,
            out=np.zeros_like(masked_t1),
            where=np.abs(masked_t2) > 1e-6,
        )
        ratio = np.clip(ratio, -5, 5)
        features = {
            "t1_mean": float(masked_t1.mean()),
            "t1_std": float(masked_t1.std()),
            "t2_mean": float(masked_t2.mean()),
            "t2_std": float(masked_t2.std()),
            "t1_t2_ratio_mean": float(np.mean(ratio)),
            "t1_t2_ratio_std": float(np.std(ratio)),
        }
        if masked_t1.size > 1 and masked_t2.size > 1:
            corr = np.corrcoef(masked_t1, masked_t2)[0, 1]
            features["t1_t2_intensity_correlation"] = float(corr)
        else:
            features["t1_t2_intensity_correlation"] = 0.0
        return features

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Zenodo infant MRI volumes and extract features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of subjects processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = InfantMRIPreprocessor(args.config)
    preprocessor.run(limit_subjects=args.limit)


if __name__ == "__main__":
    main()
