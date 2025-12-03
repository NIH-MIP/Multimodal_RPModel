# Contributor: Benjamin D. Simon and Katie Merriman
# Email: benjamin.simon@nih.gov
# Dec 03, 2025
#
# By downloading or otherwise receiving the SOFTWARE, RECIPIENT may 
# use and/or redistribute the SOFTWARE, with or without modification, 
# subject to RECIPIENT’s agreement to the following terms:
# 
# 1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS 
# OF CANINE OR HUMAN SUBJECTS.  RECIPIENT is responsible for 
# compliance with all laws and regulations applicable to the use 
# of the SOFTWARE.
# 
# 2. The SOFTWARE that is distributed pursuant to this Agreement 
# has been created by United States Government employees. In 
# accordance with Title 17 of the United States Code, section 105, 
# the SOFTWARE is not subject to copyright protection in the 
# United States.  Other than copyright, all rights, title and 
# interest in the SOFTWARE shall remain with the PROVIDER.   
# 
# 3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and 
# the name of the author of the SOFTWARE in all written publications 
# containing any data or information regarding or resulting from use 
# of the SOFTWARE. 
# 
# 4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT 
# ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.  
# 
# 5.	RECIPIENT agrees not to use any trademarks, service marks, trade names, 
# logos or product names of NCI or NIH to endorse or promote products derived 
# from the SOFTWARE without specific, prior and written permission.
# 
# 6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its 
# own copyright statement to its modifications or derivative works of the SOFTWARE 
# and may provide additional or different license terms and conditions in its 
# sublicenses of modifications or derivative works of the SOFTWARE provided that 
# RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies 
# with the conditions stated in this Agreement. Whenever Recipient distributes or 
# redistributes the SOFTWARE, a copy of this Agreement must be included with 
# each copy of the SOFTWARE.

from __future__ import annotations
import argparse
import logging
import os
import json
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Helpers
# ----------------------------

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_test_data(path: str, target_col: str) -> pd.DataFrame:
    logging.info("Loading test CSV: %s", path)
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data. Available columns: {list(df.columns)}")
    # don't drop rows with missing target here — but we will require target for metrics
    return df


def preprocess_for_inference(df: pd.DataFrame, feature_cols: List[str], impute_strategy: str = "constant",
                             constant_fill: float = -1.0, scale_minmax: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess test features to match training preprocessing.
    Missing columns are added as NaN, then imputed.
    """
    # Add any missing feature columns so the column order is identical to training
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning("Missing columns in test data; filling these with NaN: %s", missing)
        for c in missing:
            df[c] = np.nan

    X = df[feature_cols].copy()
    X = X.replace("missing", np.nan)

    if impute_strategy == "constant":
        imp = SimpleImputer(strategy="constant", fill_value=constant_fill)
    elif impute_strategy == "median":
        imp = SimpleImputer(strategy="median")
    else:
        raise ValueError("impute_strategy must be 'constant' or 'median'")

    X_imp = imp.fit_transform(X)

    if scale_minmax:
        scaler = MinMaxScaler()
        X_imp = scaler.fit_transform(X_imp)

    return X_imp, feature_cols


def compute_optimal_threshold_from_probs(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Compute Youden's J optimal threshold and AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    J = tpr - fpr
    idx = np.nanargmax(J)
    return float(thresholds[idx]), float(auc)


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    Compute metrics and return a dictionary.
    """
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold_used": float(threshold),
        "auc": float(auc),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": clf_report
    }


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------
# Main inference flow
# ----------------------------

def main(args: argparse.Namespace):
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    df_test = load_test_data(args.test_csv, args.target_col)
    y_test = df_test[args.target_col].astype(int).values

    # discover model files in models_dir if models list not provided
    model_files = []
    if args.model_files:
        model_files = args.model_files
    else:
        # look for files named like best_model_<name>.joblib
        for fn in os.listdir(args.models_dir):
            if fn.endswith(".joblib") and fn.startswith("best_model_"):
                model_files.append(os.path.join(args.models_dir, fn))
        model_files = sorted(model_files)
    if not model_files:
        raise RuntimeError("No model files found. Provide --model-files or ensure models_dir contains best_model_*.joblib")

    ensure_dir(args.outdir)

    results = {}
    all_probas = {}  # store per-model probabilities for optional ensembling
    # attempt to read feature sets info (if provided as JSON or we can infer names from filenames)
    for mf in model_files:
        model_name = os.path.splitext(os.path.basename(mf))[0].replace("best_model_", "")
        logging.info("Loading model %s from %s", model_name, mf)
        model = joblib.load(mf)

        # Try to infer expected feature columns from the training estimator if possible.
        # RandomForest itself doesn't store column names, so we rely on a convention:
        # If user passed a sidecar JSON with the same base name (e.g., best_model_M1.meta.json) use that.
        meta_json = os.path.join(os.path.dirname(mf), f"{os.path.splitext(os.path.basename(mf))[0]}.meta.json")
        if os.path.exists(meta_json):
            logging.info("Found meta file for %s -> %s", model_name, meta_json)
            meta = json.load(open(meta_json, "r"))
            feature_cols = meta.get("feature_columns", [])
            impute_strategy = meta.get("impute_strategy", "constant")
            constant_fill = meta.get("constant_fill", -1.0)
            scale_minmax = meta.get("scale_minmax", False)
        else:
            # fallback: user must provide a features CSV mapping or we attempt to use --feature-sets-csv
            if args.feature_sets_csv:
                fs_df = pd.read_csv(args.feature_sets_csv)
                # expect columns: name, cols (comma-separated)
                mapping = {row['name']: [c.strip() for c in row['cols'].split(",")] for _, row in fs_df.iterrows()}
                if model_name in mapping:
                    feature_cols = mapping[model_name]
                    impute_strategy = args.impute_strategy
                    constant_fill = args.constant_fill
                    scale_minmax = args.scale_minmax
                else:
                    raise RuntimeError(f"Model name {model_name} not found in feature_sets_csv. Provide a meta JSON or feature_sets_csv.")
            else:
                raise RuntimeError(f"No meta JSON found for {mf} and no feature_sets_csv provided to infer feature columns.")

        # Preprocess features from test set
        X_test, used_cols = preprocess_for_inference(df_test, feature_cols,
                                                     impute_strategy=impute_strategy,
                                                     constant_fill=constant_fill,
                                                     scale_minmax=scale_minmax)

        # get predicted probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            # fallback: many sklearn estimators have decision_function; map to probs with sigmoid if needed
            try:
                scores = model.decision_function(X_test)
                proba = 1 / (1 + np.exp(-scores))
                logging.warning("Model %s had no predict_proba; used sigmoid(decision_function) as approx probability.", model_name)
            except Exception as e:
                raise RuntimeError(f"Model {model_name} doesn't support probability prediction: {e}")

        all_probas[model_name] = proba

        # determine threshold to use: user-supplied, or infer via Youden's J on test set
        if args.use_train_thresholds and os.path.exists(meta_json):
            # try to use threshold saved in meta JSON (if present)
            train_thresh = meta.get("optimal_threshold", None)
            if train_thresh is not None:
                threshold = train_thresh
            else:
                threshold, auc = compute_optimal_threshold_from_probs(y_test, proba)
        else:
            # infer best threshold on test set (Youden's J) or accept user override
            if args.threshold is not None:
                threshold = args.threshold
            else:
                threshold, auc = compute_optimal_threshold_from_probs(y_test, proba)

        metrics = evaluate_predictions(y_test, proba, threshold)
        results[model_name] = {
            "model_file": mf,
            "feature_columns_used": used_cols,
            "impute_strategy": impute_strategy,
            "constant_fill": constant_fill,
            "scale_minmax": scale_minmax,
            **metrics
        }

        # Save per-model predictions to CSV
        out_df = df_test.copy()
        out_df[f"{model_name}_proba"] = proba
        out_df[f"{model_name}_pred"] = (proba >= threshold).astype(int)
        out_csv = os.path.join(args.outdir, f"predictions_{model_name}.csv")
        out_df.to_csv(out_csv, index=False)
        logging.info("Saved predictions for %s to %s", model_name, out_csv)

    # Optional ensemble: average probabilities across selected models
    if args.ensemble_models:
        sel = args.ensemble_models
        # if sel == ["ALL"], ensemble all models
        if len(sel) == 1 and sel[0].upper() == "ALL":
            sel = list(all_probas.keys())
        missing_sel = [s for s in sel if s not in all_probas]
        if missing_sel:
            raise RuntimeError(f"Requested ensemble models not found: {missing_sel}")
        stacked = np.vstack([all_probas[name] for name in sel])
        avg_proba = np.mean(stacked, axis=0)
        # threshold: user-supplied or Youden on test set
        if args.ensemble_threshold is not None:
            thresh = args.ensemble_threshold
        else:
            thresh, auc = compute_optimal_threshold_from_probs(y_test, avg_proba)
        ensemble_metrics = evaluate_predictions(y_test, avg_proba, thresh)
        results["ENSEMBLE_" + "_".join(sel)] = {
            "member_models": sel,
            "threshold_used": thresh,
            **ensemble_metrics
        }
        # save ensemble preds
        out_df = df_test.copy()
        out_df["ensemble_proba"] = avg_proba
        out_df["ensemble_pred"] = (avg_proba >= thresh).astype(int)
        out_csv = os.path.join(args.outdir, f"predictions_ensemble_{'_'.join(sel)}.csv")
        out_df.to_csv(out_csv, index=False)
        logging.info("Saved ensemble predictions to %s", out_csv)

    # Optional CAPRA baseline
    if args.capra_col and args.capra_col in df_test.columns:
        capra_raw = df_test[args.capra_col].astype(float).values
        if args.scale_minmax_for_capra:
            capra_raw = (capra_raw - np.nanmin(capra_raw)) / (np.nanmax(capra_raw) - np.nanmin(capra_raw))
        capra_proba = np.nan_to_num(capra_raw, nan=0.0)
        capra_thresh, capra_auc = compute_optimal_threshold_from_probs(y_test, capra_proba)
        capra_metrics = evaluate_predictions(y_test, capra_proba, capra_thresh)
        results["CAPRA"] = {"capra_auc": capra_auc, "threshold": capra_thresh, **capra_metrics}
        # save CAPRA preds
        out_df = df_test.copy()
        out_df["capra_proba"] = capra_proba
        out_df["capra_pred"] = (capra_proba >= capra_thresh).astype(int)
        out_csv = os.path.join(args.outdir, "predictions_CAPRA.csv")
        out_df.to_csv(out_csv, index=False)
        logging.info("Saved CAPRA predictions to %s", out_csv)

    # Save aggregated results JSON and summary CSV
    summary_json = os.path.join(args.outdir, "inference_results.json")
    with open(summary_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logging.info("Saved inference summary JSON to %s", summary_json)

    # Create a concise summary table
    rows = []
    for name, detail in results.items():
        rows.append({
            "model": name,
            "auc": detail.get("auc") or detail.get("roc_auc") or detail.get("capra_auc"),
            "accuracy": detail.get("accuracy"),
            "threshold": detail.get("threshold_used") or detail.get("threshold") or detail.get("optimal_threshold"),
        })
    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(args.outdir, "inference_summary_table.csv")
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Saved concise inference summary to %s", summary_csv)
    logging.info("Inference complete. Detailed results in %s", args.outdir)


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run inference with saved RandomForest models on a test set")
    p.add_argument("--test-csv", required=True, help="Path to test CSV with same columns as training set.")
    p.add_argument("--target-col", default="Event status (1 = BCR, 0 = Censored)", help="Name of the binary target column in test CSV.")
    p.add_argument("--models-dir", default="rf_results", help="Directory containing saved model joblib files.")
    p.add_argument("--model-files", nargs="*", help="Explicit list of model joblib file paths. If omitted, script will search models-dir for best_model_*.joblib")
    p.add_argument("--feature-sets-csv", help="Optional CSV mapping model names to columns (name,cols). Used if no .meta.json exists for models.")
    p.add_argument("--outdir", default="rf_inference_results", help="Directory to write predictions and summary.")
    p.add_argument("--impute-strategy", choices=["constant", "median"], default="constant", help="Imputation fallback if meta not present.")
    p.add_argument("--constant-fill", type=float, default=-1.0, help="Constant fill value for missing imputation.")
    p.add_argument("--scale-minmax", action="store_true", help="Scale features to 0..1 after imputation (fallback).")
    p.add_argument("--use-train-thresholds", action="store_true", help="If model meta JSON contains optimal_threshold, use it rather than recomputing on test.")
    p.add_argument("--threshold", type=float, default=None, help="Force a threshold to binarize probabilities for all models (overrides Youden if set).")
    p.add_argument("--ensemble-models", nargs="*", default=None, help="List of model names to ensemble or 'ALL' to ensemble all. E.g. --ensemble-models ALL")
    p.add_argument("--ensemble-threshold", type=float, default=None, help="Optional threshold for ensemble predictions. Otherwise computed by Youden on test.")
    p.add_argument("--capra-col", default="CAPRA", help="Optional CAPRA column to evaluate baseline.")
    p.add_argument("--scale-minmax-for-capra", action="store_true", help="Scale CAPRA 0..1 before ROC.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
