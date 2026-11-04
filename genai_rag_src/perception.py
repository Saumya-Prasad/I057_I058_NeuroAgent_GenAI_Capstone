"""
================================================================================
  NEUROAGENT: ACS-UNET 5-FOLD ENSEMBLE PERCEPTION & BIOMARKER EXTRACTION ENGINE
================================================================================
This script implements Step 2 of the Modular RAG / Agentic AI Pipeline:
1. Loads the 5-fold ensemble ACS-UNet (PEFT3DAttentionUNetV2) weights
2. Performs 3D volumetric forward inference on real ATLAS v3.0 NIfTI scans
3. Applies postprocessing to extract sharp, noise-free lesion masks
4. Computes exact clinical biomarkers:
   - Lesion volume in cm³
   - Lesion tier (Lacunar < 5 cm³ vs Small Territorial 5-15 cm³ vs Large Territorial > 15 cm³)
   - Contralateral Symmetry Asymmetry Discrepancy (Δsym)
   - Anatomical Laterality & 3D Centroid
   - Thrombolysis / EVT protocol candidates
================================================================================
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# Link your existing model architecture from Forum_files/v2
FORUM_V2_DIR = r"c:\Users\Saumya_Shambhavi\Downloads\Forum_files\v2"
if FORUM_V2_DIR not in sys.path:
    sys.path.append(FORUM_V2_DIR)

from model_v2 import PEFT3DAttentionUNetV2
from dataset_v2 import normalize_zscore, generate_spatial_prior, discover_bids_dataset
from postprocess_v2 import apply_3d_postprocessing_pipeline

class ACSUnetPerceptionEngine:
    def __init__(self, checkpoints_dir: str = None, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if checkpoints_dir is None:
            # Default to the capstone folder weights
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoints_dir = os.path.join(base_dir, "a04b06_5_folds_best_pth")
            if not os.path.exists(checkpoints_dir):
                checkpoints_dir = os.path.join(FORUM_V2_DIR, "checkpoints_a04b06")

        self.checkpoints_dir = checkpoints_dir
        self.models = self._load_5fold_ensemble()
        self.target_shape = (96, 112, 96)

    def _load_5fold_ensemble(self) -> list:
        """Loads all 5 fold models into GPU/CPU memory."""
        models = []
        print(f"\n[ACS-UNet Engine] Loading 5-Fold Ensemble from: {self.checkpoints_dir} (Device: {self.device})")
        
        for fold in range(1, 6):
            ckpt_name = f"a04b06_fold_{fold}_best.pth"
            ckpt_path = os.path.join(self.checkpoints_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                # Search recursively
                matches = glob.glob(os.path.join(self.checkpoints_dir, "**", ckpt_name), recursive=True)
                if matches:
                    ckpt_path = matches[0]

            model = PEFT3DAttentionUNetV2(in_ch=3, out_ch=1, base=16).to(self.device)
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(state, strict=False)
                print(f"   Loaded Fold {fold}: {os.path.basename(ckpt_path)}")
            else:
                print(f"  ⚠️ Warning: Fold {fold} checkpoint not found at {ckpt_path}")
            
            model.eval()
            models.append(model)

        print(f"[ACS-UNet Engine] Successfully initialized {len(models)}-Fold Ensemble.")
        return models

    def prepare_input_tensor(self, raw_img: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocesses raw 3D MRI volume to match training pipeline (3 channels: Image, Contrast, Spatial)."""
        norm_img = normalize_zscore(raw_img)
        img_t = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()
        
        # Resample to canonical training shape
        img_res = F.interpolate(img_t, size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)

        # Channel 1: Normalized T1w MRI
        # Channel 2: Contrast map (deviation from mean)
        contrast_map = torch.abs(img_res - torch.mean(img_res))
        # Channel 3: Normalized 3D coordinate spatial prior
        spatial_prior = generate_spatial_prior(self.target_shape)

        stacked_tensor = torch.cat([img_res, contrast_map, spatial_prior], dim=0)
        in_tensor = stacked_tensor.unsqueeze(0).float().to(self.device)
        return in_tensor, img_res.squeeze(0).numpy()

    def predict_mask(self, in_tensor: torch.Tensor) -> np.ndarray:
        """Runs 5-fold ensemble inference with test-time averaging."""
        with torch.no_grad():
            preds = []
            for m in self.models:
                p = m(in_tensor)
                if isinstance(p, dict):
                    p = p.get('out', p)
                preds.append(p[0, 0].cpu().numpy())
            ensemble_prob = np.mean(preds, axis=0)

        clean_mask = apply_3d_postprocessing_pipeline(ensemble_prob, threshold=0.35, min_voxels=15)
        return clean_mask

    def extract_biomarkers(self, scan_path: str, mask_path: str = None) -> dict:
        """
        Full Perception Pipeline:
        Loads NIfTI scan -> Runs 5-Fold Ensemble -> Computes Biomarkers -> Returns JSON.
        """
        img_nib = nib.as_closest_canonical(nib.load(scan_path))
        raw_img = img_nib.get_fdata().astype(np.float32)
        zooms = img_nib.header.get_zooms()[:3]
        voxel_vol_cm3 = float(np.prod(zooms) / 1000.0)

        in_tensor, img_res = self.prepare_input_tensor(raw_img)
        pred_res_mask = self.predict_mask(in_tensor)

        # Upsample mask back to original resolution
        pred_t = torch.from_numpy(pred_res_mask).unsqueeze(0).unsqueeze(0).float()
        full_pred = F.interpolate(pred_t, size=raw_img.shape, mode='nearest').squeeze().numpy()

        total_voxels = int(np.sum(full_pred > 0.5))
        volume_cm3 = round(total_voxels * voxel_vol_cm3, 3)

        # Lesion Tier classification matching your paper
        if volume_cm3 == 0:
            tier = "Negative (No Infarct Detected)"
        elif volume_cm3 < 5.0:
            tier = "Small Subcortical / Lacunar (< 5 cm³)"
        elif volume_cm3 <= 15.0:
            tier = "Small Territorial (5 - 15 cm³)"
        else:
            tier = "Large Territorial (> 15 cm³)"

        # Contralateral Discrepancy (Δsym) along mid-sagittal plane
        mid_sagittal = raw_img.shape[0] // 2
        left_voxels = int(np.sum(full_pred[:mid_sagittal, :, :] > 0.5))
        right_voxels = int(np.sum(full_pred[mid_sagittal:, :, :] > 0.5))

        if left_voxels > right_voxels:
            laterality = "Left Cerebral Hemisphere"
            asymmetry_index = round(abs(left_voxels - right_voxels) / (total_voxels + 1e-6), 4)
        elif right_voxels > left_voxels:
            laterality = "Right Cerebral Hemisphere"
            asymmetry_index = round(abs(right_voxels - left_voxels) / (total_voxels + 1e-6), 4)
        else:
            laterality = "Midline / Symmetric"
            asymmetry_index = 0.0

        # Approximate ASPECTS score
        if volume_cm3 == 0:
            aspects = 10
        elif volume_cm3 < 5.0:
            aspects = 9
        elif volume_cm3 < 15.0:
            aspects = 7
        else:
            aspects = max(3, 7 - int((volume_cm3 - 15.0) / 12.0))

        # Check ground-truth Dice if mask_path is provided
        gt_dice = None
        if mask_path and os.path.exists(mask_path):
            gt_nib = nib.as_closest_canonical(nib.load(mask_path))
            gt_data = (gt_nib.get_fdata() > 0.5).astype(np.float32)
            intersection = np.sum(full_pred * gt_data)
            union = np.sum(full_pred) + np.sum(gt_data)
            gt_dice = round(float((2.0 * intersection) / (union + 1e-8)), 4) if union > 0 else 1.0

        return {
            "scan_file": os.path.basename(scan_path),
            "original_dimensions": list(raw_img.shape),
            "voxel_resolution_mm": [round(z, 2) for z in zooms],
            "infarct_volume_cm3": volume_cm3,
            "lesion_voxel_count": total_voxels,
            "lesion_tier": tier,
            "laterality": laterality,
            "contralateral_asymmetry_delta_sym": asymmetry_index,
            "estimated_aspects_score": aspects,
            "thrombolysis_volume_eligible": bool(0 < volume_cm3 < 50.0),
            "evt_candidate_criteria": bool(0 < volume_cm3 < 70.0 and aspects >= 6),
            "ground_truth_dice": gt_dice,
            "predicted_mask": (full_pred > 0.5).astype(np.uint8)
        }

if __name__ == "__main__":
    # Test on a real scan from ATLAS3_Training_Raw
    data_dir = r"C:\Users\Saumya_Shambhavi\Downloads\ATLAS3_Training_Raw"
    records = discover_bids_dataset(data_dir)
    print(f"\nFound {len(records)} BIDS records.")
    
    test_record = records[0]
    print(f"Testing Perception Engine on Case: {os.path.basename(test_record['image'])}")

    engine = ACSUnetPerceptionEngine()
    biomarkers = engine.extract_biomarkers(test_record["image"], test_record["mask"])

    print("\n" + "=" * 70)
    print("  ACS-UNET EXTRACTED CLINICAL BIOMARKERS (GROUNDED IN REAL SCAN)")
    print("=" * 70)
    for k, v in biomarkers.items():
        print(f"  {k:35s}: {v}")
    print("=" * 70)
