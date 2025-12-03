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
import time
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Configuration / Hyperparams
# ----------------------------
RANDOM_STATE = 42
N_JOBS = -1

DEFAULT_PARAM_DIST = {
    "n_estimators": list(range(50, 700, 10)),
    "max_depth": [None] + list(range(10, 31, 10)),
    "min_samples_split": list(range(2, 11)),
    "min_samples_leaf": list(range(1, 5)),
    "max_features": [None, "sqrt", "log2"] + [float(i) / 100 for i in range(1, 100, 5)],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "class_weight": [{0: .5, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5},
                     {0: 1, 1: 6}, None],
}

# Default number of RandomizedSearchCV iterations per model (tunable)
DEFAULT_N_ITER = 300
CV_FOLDS = 5

# ----------------------------
# Helper functions
# ----------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_data(csv_path: str, target_col: str = "Event status (1 = BCR, 0 = Censored)") -> pd.DataFrame:
    """
    Load CSV and do light validation.

    Args:
        csv_path: path to CSV file
        target_col: name of the target/event column

    Returns:
        DataFrame with rows missing the target dropped.
    """
    logging.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")
    # drop samples with missing event status
    df = df.dropna(subset=[target_col])
    return df


def preprocess_features(df: pd.DataFrame, feature_cols: List[str], impute_strategy: str = "constant",
                        constant_fill: float = -1.0, scale_minmax: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and preprocess feature matrix X for given columns.

    Options:
     - impute_strategy: "constant" (fills with constant_fill) or "median"
     - scale_minmax: whether to scale features to [0,1] (useful for e.g., CAPRA normalization)

    Returns:
        X (numpy ndarray), list of final column names used
    """
    # validate requested columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning("The following requested columns are missing from the DataFrame and will be filled with NaN: %s", missing)
        # add missing columns filled with NaN
        for c in missing:
            df[c] = np.nan

    X = df[feature_cols].copy()

    # replace textual 'missing' with NaN, then impute
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


def compute_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Compute Youden's J optimal threshold from ROC curve.

    Returns:
        optimal_threshold, roc_auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    logging.debug("Computed ROC AUC: %.4f, optimal threshold: %.4f", roc_auc, optimal_threshold)
    return float(optimal_threshold), float(roc_auc)


def run_random_search_for_model(X: np.ndarray, y: np.ndarray, param_dist: Dict[str, Any],
                                n_iter: int = DEFAULT_N_ITER, cv: int = CV_FOLDS,
                                random_state: int = RANDOM_STATE) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV with RandomForestClassifier and return the fitted RandomizedSearchCV object.
    """
    logging.info("Starting RandomizedSearchCV: n_iter=%d, cv=%d", n_iter, cv)
    rfc = RandomForestClassifier(random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS,
        random_state=random_state,
        verbose=0,
    )
    rs.fit(X, y)
    logging.info("RandomizedSearchCV finished. Best score: %.4f", rs.best_score_)
    return rs


def summarize_model(rs: RandomizedSearchCV, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Compute predictions/probabilities, optimal threshold, and standard metrics for the best estimator.
    """
    best = rs.best_estimator_
    proba = best.predict_proba(X)[:, 1]
    opt_thresh, roc_auc = compute_optimal_threshold(y, proba)

    # binary predictions at optimal threshold
    y_pred = (proba >= opt_thresh).astype(int)
    acc = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    summary = {
        "best_params": rs.best_params_,
        "best_score_cv": rs.best_score_,
        "roc_auc": roc_auc,
        "optimal_threshold": opt_thresh,
        "accuracy_at_opt_thresh": acc,
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),  # JSON serializable
    }
    return summary


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------
# Main pipeline
# ----------------------------

def main(args: argparse.Namespace) -> None:
    setup_logging()

    df = load_data(args.csv, target_col=args.target_col)

    # Replace literal 'missing' string with NaN globally and optionally fill NaN later
    df = df.replace("missing", np.nan)

    # default feature sets from the snippet (names may be customized to match the paper)
    FEATURE_SETS: Dict[str, List[str]] = {
        "M0": ['Margins', 'Gleason secondary', 'Gleason primary', 'LNI', 'SVI', 'PSA', 'EPE'],
        "M1": ['ISUP GGG Sx', 'Gleason primary sx', 'Gleason secondary sx', 'Age at time of MRI', 'PSA'],
        "M2": ['ISUP GGG Sx', 'Gleason primary sx', 'Gleason secondary sx', 'Gleason secondary fx', 'Gleason primary fx', 'Age at time of MRI', 'PSA'],
        "M3": ['Indexed Relative Lesion Volume', 'ISUP GGG Sx', 'Age at time of MRI', 'PSA', 'Gleason primary sx', 'Gleason secondary sx', 'Gleason primary fx', 'Gleason secondary fx', 'dist_inside', 'area_outside', 'area_inside', 'area_dist3D', 'area_distXY'],
        "M4": ['Age at time of MRI', 'PSA', 'dist_inside', 'area_outside', 'area_inside', 'area_dist3D', 'area_distXY'],
    }

    # If user provided a feature set file, override
    if args.feature_sets_csv:
        logging.info("Loading feature sets from %s", args.feature_sets_csv)
        user_fs = pd.read_csv(args.feature_sets_csv)
        # expect two columns: 'name' and 'cols' (cols comma-separated)
        FEATURE_SETS = {}
        for _, row in user_fs.iterrows():
            name, cols = row['name'], row['cols']
            FEATURE_SETS[name] = [c.strip() for c in cols.split(",")]

    target = df[args.target_col].astype(int).values
    # optional normalization for CAPRA if present and requested
    capra_col = args.capra_col

    # prepare output folder
    ensure_dir(args.outdir)

    results_summary = {}

    start = time.time()
    for name, cols in FEATURE_SETS.items():
        logging.info("Processing feature set %s: %s", name, cols)
        X, used_cols = preprocess_features(df, cols, impute_strategy=args.impute_strategy,
                                           constant_fill=args.constant_fill, scale_minmax=args.scale_minmax)
        logging.info("Shape X: %s, y: %s", X.shape, target.shape)

        n_iter = args.n_iter_per_model
        # allow a smaller number of iterations for very large feature sets if specified
        rs = run_random_search_for_model(X, target, DEFAULT_PARAM_DIST, n_iter=n_iter, cv=CV_FOLDS, random_state=RANDOM_STATE)

        summary = summarize_model(rs, X, target)
        results_summary[name] = summary

        # save model and search object
        model_fname = os.path.join(args.outdir, f"best_model_{name}.joblib")
        joblib.dump(rs.best_estimator_, model_fname)
        logging.info("Saved best estimator for %s to %s", name, model_fname)

        # save RandomizedSearchCV results (cv results) table
        cv_results_df = pd.DataFrame(rs.cv_results_)
        cv_csv = os.path.join(args.outdir, f"cv_results_{name}.csv")
        cv_results_df.to_csv(cv_csv, index=False)
        logging.info("Saved cv results for %s to %s", name, cv_csv)

        # save summary JSON-ish (pandas-friendly)
        summary_df = pd.json_normalize(summary)
        summary_csv = os.path.join(args.outdir, f"summary_{name}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logging.info("Saved summary for %s to %s", name, summary_csv)

    # optional CAPRA baseline if available
    if capra_col and capra_col in df.columns:
        logging.info("Evaluating CAPRA baseline column: %s", capra_col)
        capra_raw = df[capra_col].values.astype(float)
        if args.scale_minmax_for_capra:
            # normalize 0..1
            capra_raw = (capra_raw - np.nanmin(capra_raw)) / (np.nanmax(capra_raw) - np.nanmin(capra_raw))
        capra_proba = np.nan_to_num(capra_raw, nan=0.0)
        capra_thresh, capra_auc = compute_optimal_threshold(target, capra_proba)
        results_summary["CAPRA"] = {"roc_auc": capra_auc, "optimal_threshold": capra_thresh}
        logging.info("CAPRA AUC: %.4f, optimal threshold: %.4f", capra_auc, capra_thresh)

    elapsed = time.time() - start
    logging.info("Pipeline finished in %.1f seconds. Results written to %s", elapsed, args.outdir)

    # save overall summary table
    overall_rows = []
    for name, summ in results_summary.items():
        row = {
            "model": name,
            "roc_auc": summ.get("roc_auc"),
            "optimal_threshold": summ.get("optimal_threshold"),
            "best_score_cv": summ.get("best_score_cv"),
        }
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(args.outdir, "overall_summary.csv")
    overall_df.to_csv(overall_csv, index=False)
    logging.info("Saved overall summary to %s", overall_csv)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RandomForest pipeline for predicting BCR following prostatectomy")
    p.add_argument("--csv", required=True, help="Path to the input CSV file containing clinical/MRI features and target.")
    p.add_argument("--target-col", default="Event status (1 = BCR, 0 = Censored)", help="Name of the target column (binary).")
    p.add_argument("--outdir", default="rf_results", help="Directory to save models and results.")
    p.add_argument("--n-iter-per-model", type=int, default=DEFAULT_N_ITER, help="RandomizedSearchCV n_iter per model.")
    p.add_argument("--impute-strategy", choices=["constant", "median"], default="constant", help="Imputation strategy for missing values.")
    p.add_argument("--constant-fill", type=float, default=-1.0, help="Constant to use when imputation strategy is 'constant'.")
    p.add_argument("--scale-minmax", action="store_true", help="Scale features to 0..1 after imputation for all models.")
    p.add_argument("--feature-sets-csv", default=None, help="Optional CSV with custom feature sets (columns: name, cols(comma-separated)).")
    p.add_argument("--capra-col", default="CAPRA", help="Optional column name to evaluate CAPRA baseline (if present).")
    p.add_argument("--scale-minmax-for-capra", action="store_true", help="If set, normalize CAPRA to 0..1 before ROC analysis.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    main(args)