from client_utils import post_file
from data_utils import window_data, check_uniform, time_to_feature, subset_scaler
from kafka_utils import produce_message, publish_error
import numpy as np
import pandas as pd
import mlflow
from mlflow.artifacts import download_artifacts  # type: ignore
import os
import pickle
import tempfile
from typing import Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Constants - These should all be defined by the service later
TIME_FEATURES = ["min_of_day", "day_of_week", "day_of_year"]
TIME_FEATURES = [f"{feature}_sin" for feature in TIME_FEATURES] + [f"{feature}_cos" for feature in TIME_FEATURES]
SAMPLE_IDX = int(os.environ.get("SAMPLE_IDX", 0))
INFERENCE_LENGTH = int(os.environ.get("INFERENCE_LENGTH", 1))

INFER_VERSION = "infer_v20251002_04"

class Inferencer:
    def __init__(self, gateway_url: str, producer, dlq_topic: str, output_topic: str):
        self.gateway_url = gateway_url
        self.producer = producer
        self.dlq_topic = dlq_topic
        self.output_topic = output_topic
        self.df = None
        self.input_seq_len = 0
        self.output_seq_len = 0
        self.current_model = None
        self.current_scaler: Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, None] = None
        self.current_experiment_name = "Default"
        self.current_run_name = ""
        self.model_type = ""
        self.model_class = ""  # "pytorch", "prophet", "statsforecast"
        # Track emitted (run_id, prediction_hash) to prevent duplicate JSONL rows
        self._emitted_prediction_keys: set[tuple[str, str]] = set()
        # Busy flag to prevent concurrent long-running inference overlap
        self.busy = False
        # Track which run_ids we've already attempted scaler resolution for (prevents spammy logs)
        self._scaler_checked_run_ids = set()

    def load_model(self, experiment_name: str, run_name: str, sort: str="Recent"):
        print(f"[Inferencer:{INFER_VERSION}] Attempting to load model for experiment: {experiment_name}, run: {run_name}")

        try:
            if sort == "Recent":
                order = ["start_time desc"]
            elif sort == "Best":
                order = ["mse desc"] # not sure if this is correct
            else:
                raise TypeError("Invalid sort argument")
            
            runs_df: pd.DataFrame = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.mlflow.runName = '{run_name}'", # Filter by run name
                order_by=order,
                max_results=1,
                output_format="pandas"
            ) # type: ignore (output_format="pandas" ensures we get a DataFrame)

            if runs_df.empty:
                raise Exception(f"No runs found for experiment '{experiment_name}' with run name '{run_name}'.")

            run_id = runs_df.loc[0, "run_id"]
            self.current_run_id = run_id

                    # Extract model parameters and store them in self.params
            run_row = runs_df.iloc[0]
            self.params = {}
            for col in run_row.index:
                if col.startswith("params."):
                    param_name = col.replace("params.", "")
                    self.params[param_name] = run_row[col]
            
            print(f"Extracted parameters: {self.params}")
            
            # Detect model type from experiment name or parameters
            self.model_type, self.model_class = self._detect_model_type(runs_df.iloc[0])
            
            print(f"Found run with ID: {run_id}, Model type: {self.model_type}, Model class: {self.model_class}")

            if self.model_class == "pytorch":
                self.input_seq_len = int(runs_df["params.input_sequence_length"][0])
                self.output_seq_len = int(runs_df["params.output_sequence_length"][0])

            base_uri = f"runs:/{run_id}"
            # Candidate artifact subpaths to try (ordered). run_name first (legacy), then 'model' (mlflow.autolog default)
            candidates = [run_name, "model"]
            # De-duplicate if run_name already 'model'
            seen = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]

            last_err: Optional[Exception] = None
            for subpath in candidates:
                candidate_uri = f"{base_uri}/{subpath}"
                try:
                    print(f"Attempting to load model from: {candidate_uri}")
                    reqs = mlflow.pyfunc.get_model_dependencies(candidate_uri)
                    print(f"Model dependencies (candidate='{subpath}'): {reqs}")
                    model = mlflow.pyfunc.load_model(candidate_uri)
                    self.current_model = model
                    self.current_experiment_name = experiment_name
                    self.current_run_name = run_name
                    print(f"✅ Model loaded successfully from subpath '{subpath}'.")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"Model load attempt failed for subpath '{subpath}': {e}")
                    last_err = e
                    continue

            if self.current_model is None:
                # Exhausted candidates - list artifacts at root for debug
                try:
                    from mlflow.tracking import MlflowClient  # type: ignore
                    client = MlflowClient()
                    arts = client.list_artifacts(run_id, path="")
                    print({"service": "inference", "event": "artifact_root_list_on_fail", "run_id": run_id, "items": [a.path for a in arts]})
                except Exception as le:  # noqa: BLE001
                    print({"service": "inference", "event": "artifact_list_fail", "run_id": run_id, "error": str(le)})
                raise last_err or Exception("Unknown model load failure (no candidates tried)")

            # Attempt to load scaler artifact (optional). Flexible discovery:
            # 1. Any *.pkl file directly under 'scaler/' artifact directory (preferred if name contains 'scaler')
            # 2. Any *.pkl at artifact root with 'scaler' in its name
            # Avoid repeated warnings by caching run_ids we've inspected.
            if run_id in self._scaler_checked_run_ids and self.current_scaler is None:
                print("[Info] Skipping scaler search (previously not found for this run).")
            elif self.current_scaler is not None and run_id in self._scaler_checked_run_ids:
                # Already loaded earlier; nothing to do
                pass
            else:
                scaler_loaded = False
                try:
                    from mlflow.tracking import MlflowClient  # type: ignore
                    client = MlflowClient()
                    # List artifacts under 'scaler' directory first
                    try:
                        scaler_dir_items = client.list_artifacts(run_id, path="scaler")
                    except Exception:
                        scaler_dir_items = []
                    pkl_candidates = [a.path for a in scaler_dir_items if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                    # If nothing found in scaler/, look at root for any scaler-related pkl
                    if not pkl_candidates:
                        try:
                            root_items = client.list_artifacts(run_id, path="")
                        except Exception:
                            root_items = []
                        root_pkls = [a.path for a in root_items if not getattr(a, 'is_dir', False) and a.path.lower().endswith('.pkl')]
                        # Prefer names containing 'scaler'
                        root_pkls_sorted = sorted(root_pkls, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                        # Keep only scaler-related first if exists
                        if root_pkls_sorted:
                            pkl_candidates = root_pkls_sorted
                    # Rank candidates: contain 'scaler' first, then shorter name
                    pkl_candidates = sorted(pkl_candidates, key=lambda n: (0 if 'scaler' in n.lower() else 1, len(n)))
                    if pkl_candidates:
                        chosen_rel_path = pkl_candidates[0]
                        scaler_artifact_uri = f"{base_uri}/{chosen_rel_path}"
                        try:
                            scaler_path = download_artifacts(artifact_uri=scaler_artifact_uri, dst_path=tempfile.gettempdir())
                            with open(scaler_path, "rb") as f:
                                self.current_scaler = pickle.load(f)
                            if self.current_scaler is not None:
                                scaler_loaded = True
                                print({
                                    "service": "inference",
                                    "event": "scaler_loaded",
                                    "run_id": run_id,
                                    "artifact_path": chosen_rel_path
                                })
                        except Exception as le:  # noqa: BLE001
                            print({
                                "service": "inference",
                                "event": "scaler_load_failed",
                                "run_id": run_id,
                                "artifact_path": chosen_rel_path,
                                "error": str(le)
                            })
                    if not scaler_loaded:
                        print({
                            "service": "inference",
                            "event": "scaler_not_found",
                            "run_id": run_id,
                            "note": "No scaler .pkl located under scaler/ or root; proceeding without scaler"
                        })
                except Exception as se:  # noqa: BLE001
                    print({
                        "service": "inference",
                        "event": "scaler_search_error",
                        "run_id": run_id,
                        "error": str(se)
                    })
                finally:
                    self._scaler_checked_run_ids.add(run_id)

        except Exception as e:
            print(f"Error loading model: {e}")
            publish_error(
                self.producer,
                self.dlq_topic,
                "Model Load",
                "Failure",
                str(e),
                {"experiment": experiment_name, "run_name": run_name}
            )

    def _detect_model_type(self, run_row: pd.Series) -> Tuple[str, str]:
        """Detect [model_type, model_class] from MLflow run parameters or tags."""

        # Check for explicit model type parameter
        if "params.model_type" in run_row and pd.notna(run_row["params.model_type"]):
            model_type = run_row["params.model_type"].upper()
            if model_type in ["LSTM", "GRU", "TETS", "TCN"]:
                return model_type, "pytorch"
            elif model_type in ["AUTOARIMA", "AUTOETS", "AUTOTHETA", "AUTOMFLES", "AUTOTBATS"]:
                return model_type, "statsforecast"
            elif model_type == "PROPHET":
                return "PROPHET", "prophet"

        # Check experiment name
        exp_name = self.current_experiment_name.lower()
        if "prophet" in exp_name:
            return "PROPHET", "prophet"

        for sf_model in ["autoarima", "autoets", "autotheta", "automfles", "autotbats"]:
            if sf_model in exp_name:
                return sf_model.upper(), "statsforecast"

        for pt_model in ["lstm", "gru", "tets", "tcn"]:
            if pt_model in exp_name:
                return pt_model.upper(), "pytorch"

        # Check params to infer framework
        if any(param.startswith("params.seasonality") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "PROPHET", "prophet"

        if any(param.startswith("params.season_length") 
            for param in run_row.index if pd.notna(run_row.get(param))):
            return "", "statsforecast"  # fallback default for statsforecast

        # Default fallback
        return "", "pytorch"

    def perform_inference(self, df_eval: Optional[pd.DataFrame] = None, inference_length: Optional[int] = None):
        """Execute inference.

        Parameters
        ----------
        df_eval : Optional[pd.DataFrame]
            Optional override dataframe. If None uses self.df.
        inference_length : Optional[int]
            Override number of forecast steps (defaults to env INFERENCE_LENGTH).

        Returns
        -------
        Optional[pd.DataFrame]
            Predictions dataframe (inverse scaled when possible) or None if skipped.
        """
        if self.busy:
            print({"service": "inference", "event": "predict_skip_busy"})
            return None
        if df_eval is None:
            if self.df is None:
                print("No data provided for inference and service dataframe is empty.")
                return None
            df_eval = self.df
        else:
            if os.getenv("INFER_VERBOSE_DATA", "0") in {"1","true","TRUE"}:
                try:
                    print("Raw data")
                    print(df_eval.head(3))
                    print(df_eval.tail(3))
                except Exception:
                    pass
        if self.current_model is None:
            print("[INFO] Model not loaded yet. Deferring inference (no DLQ).")
            return None
        local_inference_length = int(inference_length) if inference_length is not None else INFERENCE_LENGTH
        self.busy = True
        print({"service": "inference", "event": "predict_inference_start", "inference_length": int(local_inference_length)})
        try:
            total_rows = len(df_eval.index)
            min_needed = self.input_seq_len + self.output_seq_len
            if self.input_seq_len > 0 and total_rows < min_needed:
                print({
                    "service": "inference",
                    "event": "insufficient_rows",
                    "rows": int(total_rows),
                    "input_seq_len": int(self.input_seq_len),
                    "output_seq_len": int(self.output_seq_len),
                    "min_required": int(min_needed),
                    "action": "skip_inference"
                })
                return None
            required_index = SAMPLE_IDX + self.input_seq_len
            if total_rows == 0:
                print("[Inferencer] Empty dataframe passed to inference; aborting.")
                return None
            if required_index >= total_rows:
                adjusted_start_pos = total_rows - 1
                print({
                    "service": "inference",
                    "event": "adjust_start_index",
                    "reason": "index_out_of_bounds",
                    "requested_start_pos": int(required_index),
                    "adjusted_start_pos": int(adjusted_start_pos),
                    "input_seq_len": int(self.input_seq_len),
                    "sample_idx": int(SAMPLE_IDX),
                    "rows": int(total_rows)
                })
            else:
                adjusted_start_pos = required_index

            timedelta = check_uniform(df_eval)
            start_timestamp = df_eval.index[adjusted_start_pos]
            df_predictions = pd.DataFrame(
                index=pd.date_range(
                    start=start_timestamp,
                    periods=local_inference_length,
                    freq=timedelta
                ),
                columns=df_eval.columns
            )
            df_predictions = time_to_feature(df_predictions)

            if self.model_class == "pytorch":
                df_transformed_predictions = self._perform_pytorch_inference(df_eval, df_predictions, local_inference_length)
            elif self.model_class == "prophet":
                df_transformed_predictions = self._perform_prophet_inference(df_eval, df_predictions, local_inference_length)
            elif self.model_class == "statsforecast":
                df_transformed_predictions = self._perform_statsforecast_inference(df_eval, df_predictions, local_inference_length)
            else:
                raise ValueError(f"Unsupported model class: {self.model_class}")

            self._save_and_publish_predictions(df_transformed_predictions, df_eval)
            print({"service": "inference", "event": "predict_inference_end", "rows": int(df_transformed_predictions.shape[0])})
            return df_transformed_predictions
        finally:
            self.busy = False

    def _perform_pytorch_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int) -> pd.DataFrame:
        """PyTorch inference logic"""
        import torch

        FEATURES = df_eval.columns.difference(TIME_FEATURES, sort=False).tolist()
        # Heuristic target column resolution (training created synthetic 'value' from fallback 'down')
        target_col = 'value' if 'value' in df_eval.columns else 'down'
        if target_col not in df_predictions.columns:
            # Ensure target column exists to receive predictions
            df_predictions[target_col] = np.nan

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_eval, _ = window_data(df_eval, TIME_FEATURES, self.input_seq_len, self.output_seq_len)
        X_eval_tensor = torch.from_numpy(X_eval).float().to(device)

        remaining_real_data = X_eval.shape[0] - SAMPLE_IDX
        available_future_steps = min(remaining_real_data, local_inference_length)

        progress_interval = int(os.environ.get("PREDICT_PROGRESS_INTERVAL", 25))
        per_step_errors = []  # will store (step, mae_over_features) for overlapping steps with real data
        with torch.no_grad():
            current_sequence = X_eval_tensor[SAMPLE_IDX].unsqueeze(0).to(device)

            for step in range(local_inference_length):
                # Predict a full block of up to output_seq_len steps
                multi_step_pred = self.current_model.predict(current_sequence.cpu().numpy())  # type: ignore
                steps_to_use = min(self.output_seq_len, local_inference_length - step)

                for i in range(steps_to_use):
                    absolute_step = step + i
                    if absolute_step >= local_inference_length:
                        break

                    current_pred = multi_step_pred[:, i, :].flatten()
                    # Shape alignment: model may output only target (dim=1) or full feature vector.
                    if current_pred.shape[0] == len(FEATURES):
                        df_predictions.loc[df_predictions.index[absolute_step], FEATURES] = current_pred
                    elif current_pred.shape[0] == 1:
                        df_predictions.loc[df_predictions.index[absolute_step], target_col] = float(current_pred.item())
                    else:
                        print({
                            "service": "inference",
                            "event": "unexpected_pred_dim",
                            "pred_dim": int(current_pred.shape[0]),
                            "n_features": int(len(FEATURES))
                        })

                    next_step_idx = SAMPLE_IDX + absolute_step + 1
                    if next_step_idx < X_eval_tensor.shape[0]:
                        # Safe: use the next real row
                        current_sequence = X_eval_tensor[next_step_idx].unsqueeze(0).to(device)
                    else:
                        # Need to extend with predictions (recursive mode). We may only have endogenous prediction(s).
                        extension_idx = absolute_step + 1 - available_future_steps
                        if extension_idx < df_predictions.shape[0]:
                            # Build a full feature vector of the SAME dimensionality as the original input sequence.
                            feature_dim = current_sequence.shape[-1]
                            pred_dim = current_pred.shape[0]
                            # Allocate full vector and copy endogenous prediction(s) at the front.
                            pred_tensor_full = torch.zeros(1, 1, feature_dim, device=device)
                            pred_tensor_full[0, 0, :pred_dim] = torch.tensor(current_pred, dtype=torch.float32, device=device)
                            # Fill remaining exogenous feature slots with last known real values (persistence strategy)
                            if pred_dim < feature_dim:
                                pred_tensor_full[0, 0, pred_dim:] = current_sequence[0, -1, pred_dim:]
                            # Log when dimensionality repair occurs
                            if pred_dim != feature_dim:
                                print({
                                    "service": "inference",
                                    "event": "recursive_extension_pad",
                                    "pred_dim": int(pred_dim),
                                    "feature_dim": int(feature_dim),
                                    "strategy": "pad_with_last_exogenous"
                                })
                            current_sequence = torch.cat((current_sequence[:, 1:, :], pred_tensor_full), dim=1)
                        else:
                            print(f"[Warning] df_predictions extension exhausted at index {extension_idx}. Stopping inference.")
                            break

                # Compute per-step error if within available_future_steps
                if step < available_future_steps:
                    try:
                        actual_idx = SAMPLE_IDX + self.input_seq_len + step
                        if actual_idx < df_eval.shape[0]:
                            if current_pred.shape[0] == len(FEATURES):
                                compare_cols = FEATURES
                            else:
                                compare_cols = [c for c in [target_col] if c in df_eval.columns and c in df_predictions.columns]
                            if compare_cols:
                                actual_row = df_eval.iloc[actual_idx][compare_cols]
                                pred_row = df_predictions.iloc[step][compare_cols]
                                mae_step = float(np.mean(np.abs(pred_row.values - actual_row.values)))
                                per_step_errors.append({"step": step, "mae": mae_step})
                    except Exception:
                        pass

                if progress_interval > 0 and (step + 1) % progress_interval == 0:
                    print({
                        "service": "inference",
                        "event": "progress",
                        "step": step + 1,
                        "total": local_inference_length,
                        "pct": round(100.0 * (step + 1) / local_inference_length, 2)
                    })

                step += steps_to_use - 1  # outer loop also increments


        df_predictions = df_predictions.drop(columns=TIME_FEATURES)

        if self.current_scaler is not None:
            try:
                original_cols = (
                    list(getattr(self.current_scaler, "feature_names_in_", []))
                    or list(getattr(self.current_scaler, "feature_names", []))
                )
                if original_cols:
                    sub_scaler = subset_scaler(self.current_scaler, original_cols, df_predictions.columns.tolist())
                else:
                    # Fallback: scaler may not have feature names stored (older sklearn); use length check
                    sub_scaler = self.current_scaler
                inv = sub_scaler.inverse_transform(df_predictions)
                df_transformed_predictions = pd.DataFrame(inv, index=df_predictions.index, columns=df_predictions.columns)
            except Exception as e:  # noqa: BLE001
                print(f"[Warning] inverse scaling failed ({e}); returning raw predictions.")
                df_transformed_predictions = df_predictions.copy()
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"PyTorch Inference completed:")
        print(f"- Used actual future values for first {min(available_future_steps, local_inference_length)} steps")
        if local_inference_length > available_future_steps:
            print(f"- Switched to recursive mode after step {available_future_steps}")
        print(f"- Model predicts {self.output_seq_len} step(s) at a time")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        try:
            print({
                "service": "inference",
                "event": "inference_shape_summary",
                "feature_dim": int(X_eval_tensor.shape[-1]),
                "output_seq_len": int(self.output_seq_len),
                "input_seq_len": int(self.input_seq_len),
                "pred_columns": df_transformed_predictions.columns.tolist()
            })
        except Exception:
            pass

        # Attach per-step error list to instance for later logging in save method (trim long list)
        try:
            max_err_steps = int(os.environ.get("PREDICT_MAX_ERROR_STEPS", 200))
            self._last_per_step_errors = per_step_errors[:max_err_steps]
        except Exception:
            self._last_per_step_errors = per_step_errors

        return df_transformed_predictions

    def _perform_prophet_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int) -> pd.DataFrame:
        """Prophet inference logic"""
        
        # Get predictions from Prophet model
        df_predictions = self.current_model.predict(df_predictions) # type: ignore
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"Prophet Inference completed:")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        print(f"- Features forecasted: {list(df_transformed_predictions.columns)}")

        return df_transformed_predictions

    def _perform_statsforecast_inference(self, df_eval: pd.DataFrame, df_predictions: pd.DataFrame, local_inference_length: int) -> pd.DataFrame:
        """StatsForecast inference logic"""

        if self.params["downsampling"] == "0" or self.params["downsampling"] == self.params["frequency"]:
            exog_df = df_predictions[TIME_FEATURES] if TIME_FEATURES else None

            input_dict = {
                "h": INFERENCE_LENGTH,
                "X": exog_df,
                "level": None
            }
        else:
            downsampling = pd.Timedelta(self.params["downsampling"])
            frequency = pd.Timedelta(self.params["frequency"])
            inf_len: int = int(np.ceil(local_inference_length*frequency/downsampling))

            if TIME_FEATURES:
                df_predictions = pd.DataFrame(
                    index=pd.date_range(
                        start=df_eval.index[SAMPLE_IDX],
                        periods=inf_len,
                        freq=frequency
                    ),
                    columns=df_eval.columns
                )

                df_predictions = time_to_feature(df_predictions)
                exog_df = df_predictions[TIME_FEATURES]
            else:
                exog_df = None

            input_dict = {
                "h": inf_len,
                "X": exog_df,
                "level": None
            }


        df_predictions = self.current_model.predict(input_dict) # type: ignore
        
        # Apply inverse scaling if scaler is available
        if self.current_scaler is not None:
            df_transformed_predictions = pd.DataFrame(
                self.current_scaler.inverse_transform(df_predictions),
                index=df_predictions.index,
                columns=df_predictions.columns
            )
        else:
            print("[Warning] current_scaler is None. Returning raw predictions.")
            df_transformed_predictions = df_predictions.copy()

        print(f"StatsForecast Inference completed:")
        print(f"- Total predictions generated: {df_transformed_predictions.shape[0]}")
        print(f"- Features forecasted: {df_transformed_predictions.columns.to_list()}")

        return df_transformed_predictions

    def _save_and_publish_predictions(self, df_transformed_predictions: pd.DataFrame, df_eval: Optional[pd.DataFrame] = None):
        """Write a single JSON object (one line) per inference batch to MinIO (JSONL) and emit Kafka confirmation.

        Required JSON fields:
          - timestamp (UTC ISO8601 Z)
          - identifier
          - run_id
          - model_type
          - config_hash
          - status (SUCCESS/FAILURE)
          - metrics (dict: mae_mean, mse_mean, rmse, per-feature errors, step_mae, etc.)
          - samples (list of sample prediction dicts)
        Storage layout: bucket=inference-logs (or INFERENCE_LOG_BUCKET env override), object key: {identifier}/{YYYYMMDD}/results.jsonl
        Append only: fetch existing object (if any), add one new line, re-upload.
        """
        from datetime import datetime
        import json, math, hashlib
        from client_utils import get_file

        identifier = os.environ.get("IDENTIFIER", "default") or "default"
        bucket = os.environ.get("INFERENCE_LOG_BUCKET", "inference-logs")
        date_part = datetime.utcnow().strftime("%Y%m%d")
        object_key = f"{identifier}/{date_part}/results.jsonl"
        status = "SUCCESS"

        # --- Metrics & Samples -------------------------------------------------
        metrics_block: dict = {}
        samples_block: list = []
        try:
            if df_eval is not None and not df_eval.empty:
                pred_idx = df_transformed_predictions.index
                overlap_idx = pred_idx.intersection(df_eval.index)
                if len(overlap_idx) > 0:
                    predicted_cols = df_transformed_predictions.columns.tolist()
                    actual_subset = df_eval.loc[overlap_idx, predicted_cols].copy()
                    # Try inverse scaling to original scale for actuals
                    if self.current_scaler is not None:
                        try:
                            original_cols = list(getattr(self.current_scaler, "feature_names_in_", [])) or []
                            if original_cols:
                                sub_scaler = subset_scaler(self.current_scaler, original_cols, predicted_cols)
                                actual_inv = sub_scaler.inverse_transform(actual_subset)
                                actual_subset_df = pd.DataFrame(actual_inv, index=actual_subset.index, columns=predicted_cols)
                            else:
                                actual_subset_df = actual_subset
                        except Exception:
                            actual_subset_df = actual_subset
                    else:
                        actual_subset_df = actual_subset
                    preds_overlap = df_transformed_predictions.loc[overlap_idx]
                    err_mae = (preds_overlap - actual_subset_df).abs().mean().to_dict()
                    err_mse = ((preds_overlap - actual_subset_df)**2).mean().to_dict()
                    # Collect non-NaN values for aggregate means
                    mae_vals = [float(v) for v in err_mae.values() if v is not None and not math.isnan(float(v))]
                    mse_vals = [float(v) for v in err_mse.values() if v is not None and not math.isnan(float(v))]
                    mae_mean = float(np.mean(mae_vals)) if mae_vals else math.nan
                    mse_mean = float(np.mean(mse_vals)) if mse_vals else math.nan
                    rmse = float(math.sqrt(mse_mean)) if (not math.isnan(mse_mean)) else math.nan
                    metrics_block = {
                        "overlap_rows": int(len(overlap_idx)),
                        "mae_mean": mae_mean,
                        "mse_mean": mse_mean,
                        "rmse": rmse,
                        # Filter out features that are entirely NaN so logs stay cleaner
                        "mae": {k: float(v) for k, v in err_mae.items() if v is not None and not math.isnan(float(v))},
                        "mse": {k: float(v) for k, v in err_mse.items() if v is not None and not math.isnan(float(v))},
                    }
                    # Samples (bounded)
                    feature_limit = int(os.environ.get("PREDICT_LOG_FEATURE_LIMIT", 3))
                    feats_for_samples = predicted_cols[:feature_limit]
                    n_pred = len(df_transformed_predictions)
                    sample_positions = sorted({0, n_pred-1, n_pred//10, n_pred//2, (9*n_pred)//10})
                    for pos in sample_positions:
                        if 0 <= pos < n_pred:
                            ts = df_transformed_predictions.index[pos]
                            row_pred = df_transformed_predictions.iloc[pos][feats_for_samples].to_dict()
                            if ts in actual_subset_df.index:
                                row_actual = actual_subset_df.loc[ts][feats_for_samples].to_dict()
                                row_err = {f: float(abs(row_pred[f]-row_actual[f])) for f in feats_for_samples}
                            else:
                                row_actual = None
                                row_err = {}
                            samples_block.append({
                                "step": int(pos),
                                "ts": ts.isoformat(),
                                "pred": {k: float(v) for k, v in row_pred.items()},
                                "actual": ({k: float(v) for k, v in row_actual.items()} if row_actual else None),
                                "abs_err": row_err
                            })
        except Exception as e:  # noqa: BLE001
            metrics_block = {"metrics_error": str(e)}

        # Step MAE sequence if available
        if hasattr(self, "_last_per_step_errors") and getattr(self, "_last_per_step_errors"):
            metrics_block["step_mae"] = self._last_per_step_errors

        # Prediction hash for quick diff / lineage
        try:
            pred_hash = hashlib.sha256(str(df_transformed_predictions.head(3).to_dict()).encode()).hexdigest()[:16]
        except Exception:
            pred_hash = ""
        metrics_block["prediction_hash"] = pred_hash
        metrics_block["rows_predicted"] = int(len(df_transformed_predictions))

        # Deduplication: skip if we've already emitted this exact prediction hash for this run
        run_id = getattr(self, "current_run_id", "")
        pred_key = (run_id, pred_hash)
        if run_id and pred_hash and pred_key in self._emitted_prediction_keys:
            print({
                "service": "inference",
                "event": "duplicate_prediction_skip",
                "run_id": run_id,
                "prediction_hash": pred_hash
            })
            return  # Do not append another identical line
        if run_id and pred_hash:
            self._emitted_prediction_keys.add(pred_key)

        # Build JSON line
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "identifier": identifier,
            "run_id": getattr(self, "current_run_id", ""),
            "model_type": self.model_type or getattr(self, "current_run_name", ""),
            "config_hash": getattr(self, "current_config_hash", None),
            "status": status,
            "metrics": metrics_block,
            "samples": samples_block,
        }

        line = json.dumps(record, default=str) + "\n"

        # --- Append to MinIO object (download + append + re-upload) -----------
        from client_utils import post_file
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    existing_obj = get_file(self.gateway_url, bucket, object_key)
                except Exception:
                    existing_obj = None
                if existing_obj is None:
                    existing_bytes = b""
                else:
                    try:
                        existing_bytes = existing_obj.getvalue()  # type: ignore[attr-defined]
                    except Exception:
                        existing_bytes = existing_obj if isinstance(existing_obj, (bytes, bytearray)) else b""
                new_body = existing_bytes + line.encode()
                post_file(self.gateway_url, bucket, object_key, new_body)
                print({
                    "service": "inference",
                    "event": "inference_log_write",
                    "bucket": bucket,
                    "object_key": object_key,
                    "identifier": identifier,
                    "lines_appended": 1,
                    "bytes_appended": len(line)
                })
                break
            except Exception as e:  # noqa: BLE001
                if attempt == max_retries:
                    publish_error(
                        self.producer,
                        dlq_topic=os.environ.get("DLQ_PERFORMANCE_TOPIC", "DLQ-performance-eval"),
                        operation="Inference Log Write",
                        status="Failure",
                        error_details=str(e),
                        payload={"object_key": object_key, "identifier": identifier, "attempts": attempt},
                    )
                else:
                    print(f"[Warning] inference JSONL log write attempt {attempt} failed: {e}")

        # --- Publish Kafka success event --------------------------------------
        try:
            produce_message(self.producer, self.output_topic, {
                "operation": "Inference",
                "status": status,
                "identifier": identifier,
                "log_bucket": bucket,
                "log_object_key": object_key,
                "run_id": record.get("run_id"),
                "model_type": record.get("model_type"),
                "config_hash": record.get("config_hash"),
                "rows": metrics_block.get("rows_predicted", 0)
            })
        except Exception as e:  # noqa: BLE001
            print(f"Kafka inference publish error (non-fatal): {e}")