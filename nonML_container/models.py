import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc.model import PythonModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from prophet import Prophet
from typing import List, Dict, Any, Optional, Tuple
from prophet.diagnostics import cross_validation, performance_metrics
from statsforecast import StatsForecast
from utilsforecast.losses import mae, mse
from statsforecast.models import (                                                                                              # type: ignore
    AutoARIMA, # AutoRegressive Integrated Moving Average
    AutoETS, # Exponential Smoothing
    AutoTheta,
    AutoMFLES,
    AutoTBATS # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend components, Seasonal components
)

class ProphetMultiFeatureModel(PythonModel):
    """
    Wrapper class to bundle multiple Prophet models for different features
    """
    
    def __init__(self):
        self.models: Dict[str, Prophet] = {}
        self.feature_columns: List[str] = []
        self.model_params: Dict[str, Any] = {}
    
    @staticmethod
    def _fit_feature(args) -> Tuple[str, Prophet]:
        y_column, df, model_params, exog_columns = args

        # Prepare data for this specific feature
        prophet_df = df[["ds", y_column]].rename(columns={y_column: "y"})
        prophet_df = prophet_df.dropna() # Just in case

        # Add country holidays if specified
        if "country" in model_params:
            country = model_params.pop("country")
        else:
            country = None

        # Initialize the Prophet model with the specified parameters
        model = Prophet(**model_params)

        if country is not None:
            model.add_country_holidays(country_name=country)

        # Add exogenous regressors if any
        if exog_columns is not None and len(exog_columns) > 0:
            for exog_col in exog_columns:
                model.add_regressor(exog_col)

        # Fit the model
        model.fit(prophet_df)

        return y_column, model

    def fit(self, df: pd.DataFrame, feature_columns: List[str], model_params: Dict[str, Any], exog_columns: List[str]=[], max_workers: Optional[int]=None):
        """
        Fit Prophet models for each feature column, with optional exogenous variables.
        
        Args:
            df: DataFrame with 'ds' column, feature columns, and optional exogenous columns.
            feature_columns: List of column names to forecast.
            model_params: Dictionary of parameters for Prophet model.
            exog_columns: List of column names to use as exogenous variables.
            max_workers: The maximum number of worker processes to use.
        """

        self.feature_columns = feature_columns
        self.model_params = model_params

        # Prepare arguments for each task
        tasks = [
            (col, df[["ds", col, *exog_columns]], self.model_params, exog_columns)
            for col in feature_columns if col != "ds"
        ]

        # Fit models sequentially to avoid Prophet stan_backend serialization issues
        for task in tasks:
            column, model = self._fit_feature(task)
            self.models[column] = model
            print(f"Finished fitting model for: {column}")

    @staticmethod
    def _predict_feature(args) -> Tuple[str, pd.DataFrame]:
            column, model, future_df = args
            forecast = model.predict(future_df[["ds"]])
            predictions = forecast[["yhat", "yhat_lower", "yhat_upper"]]
            predictions.columns = [f"{column}", f"{column}_yhat_lower", f"{column}_yhat_upper"]
            return column, predictions

    def predict(self, context, model_input, params=None):
        """
        Generate predictions for all features
        
        Args:
            context: MLflow context (unused)
            model_input: DataFrame with 'ds' column for prediction dates
            params: Optional parameters (unused)
            
        Returns:
            DataFrame with predictions for all features
        """
        
        # Accept a variety of inputs and normalize to a DataFrame containing a 'ds' column.
        if isinstance(model_input, pd.DataFrame):
            future_df: pd.DataFrame = model_input.copy()
        else:
            future_df = pd.DataFrame(model_input)

        # If 'ds' already present, don't attempt to reset index to 'ds'. The earlier logic caused
        # a pandas error: "cannot insert ds, already exists" when the caller passed a frame that
        # already had a 'ds' column. Only create/rename if missing.
        if 'ds' not in future_df.columns:
            # If the index looks like a datetime index, promote it; otherwise require explicit 'ds'.
            if isinstance(future_df.index, pd.DatetimeIndex):
                # reset_index will create a column named 'index'; rename to ds.
                future_df = future_df.reset_index().rename(columns={'index': 'ds'})
            else:
                # Last resort: try to find a single datetime-like column to treat as ds.
                candidate = None
                for c in future_df.columns:
                    if 'ds' in c.lower() or 'date' in c.lower() or 'time' in c.lower():
                        candidate = c
                        break
                if candidate and candidate != 'ds':
                    future_df = future_df.rename(columns={candidate: 'ds'})
                elif candidate is None:
                    raise ValueError("Predict input must include a 'ds' datetime column or have a DatetimeIndex.")

        predictions: List[pd.DataFrame] = []
        
        tasks = [
            (col, self.models[col], future_df)
            for col in self.feature_columns if col in self.models
        ]

        # Predict sequentially to avoid Prophet serialization issues
        for task in tasks:
            _, predictions_df = self._predict_feature(task)
            predictions.append(predictions_df)

        # Concatenate the ds column with all prediction DataFrames
        all_predictions = pd.concat(predictions, axis=1)

        return all_predictions.set_index(future_df["ds"])
    
    # Unused
    @staticmethod
    def _run_cv(args: Tuple) -> Tuple[str, Dict[str, float]]:
        """
        Runs cross-validation for a single model.
        
        Args:
            args: A tuple containing (column, model, horizon)
            
        Returns:
            A tuple containing the column name and a dictionary of performance metrics.
        """
        column, model, horizon = args
        metrics = {}
        try:
            # Perform cross-validation
            cv_results = cross_validation(model, horizon=horizon)
            # Calculate performance metrics
            df_p = performance_metrics(cv_results)
            
            # Aggregate metrics, handling potential empty results
            if not df_p.empty:
                metrics[f"{column}_mae"] = df_p['mae'].mean()
                metrics[f"{column}_mape"] = df_p['mape'].mean()
                metrics[f"{column}_rmse"] = df_p['rmse'].mean()
            else:
                metrics[f"{column}_mae"] = None
                metrics[f"{column}_mape"] = None
                metrics[f"{column}_rmse"] = None
                
        except Exception as e:
            print(f"Cross-validation failed for {column}: {e}")
            metrics[f"{column}_mae"] = None
            metrics[f"{column}_mape"] = None
            metrics[f"{column}_rmse"] = None
            
        return column, metrics
    # Unused
    def get_cross_validation_metrics(self, horizon: str) -> Dict[str, float]:
        """
        Perform cross-validation for all features in parallel and return aggregated metrics.
        Prophet retrains the model for each fold, so this can be slow.
        
        Args:
            horizon: The forecast horizon string (e.g., '30 days').
            
        Returns:
            A dictionary of aggregated performance metrics for all features.
        """
        print("Starting cross-validation for all features...")

        all_metrics = {}
        
        # Prepare arguments for each parallel task
        tasks = [
            (col, self.models[col], horizon)
            for col in self.feature_columns if col in self.models
        ]
        
        # Use ProcessPoolExecutor to run CV in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._run_cv, task) for task in tasks]
            for future in as_completed(futures):
                _, metrics = future.result()
                all_metrics.update(metrics)
        
        return all_metrics
    
    def validate(self, df: pd.DataFrame):
        """
        Validate the model on a test set and return performance metrics.
        
        Args:
            df: DataFrame with 'ds' column and feature columns for validation.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        print("Starting validation on test set...")

        non_feature_columns = [col for col in df.columns if col not in self.feature_columns]

        input_df = df[non_feature_columns]

        pred_df = self.predict(None, input_df)
        pred_df.drop(columns=[col for col in pred_df.columns if col not in self.feature_columns], inplace=True)
        true_df = df.drop(columns=non_feature_columns)

        metrics = {
            "mae": mean_absolute_error(true_df, pred_df),
            "mse": mean_squared_error(true_df, pred_df)
        }

        print(f"Validation metrics: {metrics}")

        return metrics


class StatsForecastMultiFeatureModel(PythonModel):
    """
    Wrapper class to bundle multiple StatsForecast models for different features
    """
    
    def __init__(self):
        self.models: Dict[str, StatsForecast] = {}
        self.feature_columns: List[str] = []
        self.model_params: Dict[str, Any] = {}
        self.model_type: str = ""
        self.exog_df: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, feature_columns: List[str], model_params: Dict[str, Any], exog_columns: List[str]=[], max_workers: int=-1):
        """
        Fit StatsForecast models for each feature column
        
        Args:
            df: DataFrame with 'ds', 'unique_id' columns and feature columns
            feature_columns: List of column names to forecast
            exog_columns: List of exogenous feature columns
            model_params: Dictionary of parameters for StatsForecast models
            freq: Frequency string for StatsForecast
            model_type: Type of model (AUTOARIMA, AUTOETS, etc.)
        """
        self.feature_columns = feature_columns
        self.model_params = model_params
        self.model_type = model_params.get("model_type", "undefined")
        
        # Filter exogenous columns to avoid rank deficiency
        # Remove constant columns and low-variance columns that cause multicollinearity
        filtered_exog = []
        if exog_columns:
            for col in exog_columns:
                if col in df.columns:
                    col_std = df[col].std()
                    col_nunique = df[col].nunique()
                    # Keep column if it has variance > 1e-6 and more than 1 unique value
                    if col_std > 1e-6 and col_nunique > 1:
                        filtered_exog.append(col)
                    else:
                        print(f"Dropping exog column {col}: std={col_std:.2e}, nunique={col_nunique}")
        
        self.exog_columns = filtered_exog
        exog_df = df[filtered_exog] if filtered_exog else None
        
        # Create model based on type and parameters
        season_length = model_params.get("season_length", [1])
        sl = season_length[0] if isinstance(season_length, list) else season_length
        
        models_dict = {
            "AUTOARIMA": AutoARIMA(season_length=sl),
            "AUTOETS": AutoETS(season_length=sl),
            "AUTOTHETA": AutoTheta(season_length=sl),
            "AUTOMFLES": AutoMFLES(season_length=season_length, test_size=model_params.get("output_sequence_length", 1)),
            "AUTOTBATS": AutoTBATS(season_length=season_length)
        }
        
        for column in feature_columns:
            if column in ["ds", "unique_id"]:
                continue
            
            print(f"Fitting {self.model_type} model for feature: {column}")

            # Prepare data for this feature - use filtered_exog instead of exog_columns
            feature_df = df[["ds", "unique_id", column] + filtered_exog].rename(columns={column: "y"})
            
            # Create and fit StatsForecast model with error handling for rank deficiency
            try:
                sf = StatsForecast(
                    models=[models_dict[self.model_type]],
                    freq=model_params.get("frequency", 0),
                    n_jobs=max_workers,
                )
                sf.fit(feature_df) 
                self.models[column] = sf
                print(f"Successfully fitted {self.model_type} for feature: {column}")
            except (ValueError, np.linalg.LinAlgError) as e:
                error_str = str(e)
                if "rank deficient" in error_str or "singular" in error_str:
                    # Retry without exogenous variables
                    print(f"Rank deficient error for {column}, retrying without exog features")
                    feature_df_no_exog = df[["ds", "unique_id", column]].rename(columns={column: "y"})
                    sf = StatsForecast(
                        models=[models_dict[self.model_type]],
                        freq=model_params.get("frequency", 0),
                        n_jobs=max_workers,
                    )
                    sf.fit(feature_df_no_exog)
                    self.models[column] = sf
                    print(f"Successfully fitted {self.model_type} for {column} WITHOUT exog")
                else:
                    raise
    
    def predict(self, context, model_input, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate predictions for all features.

        Args:
            context: MLflow context (unused).
            model_input:
                - "h": forecast horizon (int, default=1)
                - "exog": exogenous DataFrame with datetime index (optional but necessary if the model was fitted with exogenous variables)
                - "level": list of confidence intervals (optional)
            params: Optional extra parameters.

        Returns:
            DataFrame with forecasts for all feature columns.
        """
        # Extract parameters
        h: int = 1
        X: Optional[pd.DataFrame] = None
        level: Optional[List[int]] = None

        if isinstance(model_input, dict):
            h = model_input.get("h", 1)
            X = model_input.get("X", None)
            if X is not None:
                X = X.copy()
            level = model_input.get("level", None)

        if params:
            if h == 1 and "h" in params:
                h = params["h"]
            if level is None and "level" in params:
                level = params["level"]

        if not level:
            level = None

        if X is not None:    
            X.loc[:, "unique_id"] = "1"
            X.reset_index(names="ds", inplace=True)

        all_predictions = []

        for column in self.feature_columns:
            if column in ["ds", "unique_id"] or column not in self.models:
                continue

            forecast: pd.DataFrame = self.models[column].predict(h=h, X_df=X, level=level)  # type: ignore

        # Extract the forecast series and align on time index
            match = next((c for c in forecast.columns if c.lower() == self.model_type.lower()), None)
            if match:
                forecast = forecast.rename(columns={match: column})

            series = forecast[forecast.columns.difference(["unique_id"], sort=False)]

            series = series.set_index("ds")
            all_predictions.append(series)

        if all_predictions:
            # Join on the "ds" index to ensure time alignment
            df_predictions = pd.concat(all_predictions, axis=1)

            # Reorder columns to match the original input DataFrame
            df_predictions = df_predictions[self.feature_columns]

            return df_predictions
        else:
            return pd.DataFrame()
    
    # Unused
    def validate(self, df: pd.DataFrame):
        """
        Perform cross-validation to evaluate model performance (MAE, MSE).
        Cross-validation parameters (h, step_size, n_windows) are chosen 
        automatically based on the series length.

        Args:
            df: DataFrame with ['ds', 'unique_id'] and feature columns.
        
        Returns:
            metrics_df: DataFrame with MAE and MSE per feature column.
        """
        from utilsforecast.evaluation import evaluate
        from utilsforecast.losses import mae, mse
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        # Assuming 'exog_df' is the DataFrame of your exogenous features
        # VIF requires a constant (intercept) term to be added
        exog_df = df[[col for col in df.columns if col not in ["ds", "unique_id"]]]
        print(exog_df.info())
        X = add_constant(exog_df)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=df.columns.insert(0, 'const'))

        # Calculate VIF for each feature
        vif_df = pd.DataFrame()
        vif_df["feature"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        print("VIF Scores:")
        print(vif_df)

        split_idx = int(df.shape[0]*0.8)

        results = []

        for column, model in self.models.items():
            df_single = df[["unique_id", "ds", column] + self.exog_columns].rename(columns={column: "y"})
            train_df = df_single[:split_idx].copy()
            test_df = df_single[split_idx:].copy()
            
            try:
                # fit on train
                model.fit(train_df)

                # forecast horizon = test length
                fcst = model.forecast(df=train_df, h=split_idx)

                # align with test set
                merged = pd.merge(test_df, fcst, on=["unique_id", "ds"])

                # evaluate
                metrics = evaluate(merged, metrics=[mae, mse])
                metrics["feature"] = column
                results.append(metrics)

            except Exception as e:
                print(f"Skipping {column} due to error: {e}")

        print(results)


        self.exog_columns.remove('day_of_year_cos') 

        for column in self.feature_columns:
            if column in ["ds", "unique_id"] or column not in self.models:
                continue

            feature_df = df[["ds", "unique_id", column] + self.exog_columns].rename(columns={column: "y"})
            n_obs = feature_df.groupby("unique_id").size().min()

            # Auto CV parameters
            h = max(1, n_obs // 3)              # horizon ~33% of series length
            step_size = h
            n_windows = max(1, (n_obs // h) - 1)

            print(f"[validate] Feature={column}, h={h}, step_size={step_size}, n_windows={n_windows}")

            # Cross-validation
            cv_df = self.models[column].cross_validation(
                df=feature_df,
                h=h,
                step_size=step_size,
                n_windows=n_windows
            )

            # Compute metrics for this feature/model
            model_name = self.model_type
            mae_scores = mae(cv_df, models=[model_name])
            mse_scores = mse(cv_df, models=[model_name])

            # Merge MAE/MSE into single row per feature
            merged = mae_scores.merge(
                mse_scores, on=["unique_id"], suffixes=("_mae", "_mse")
            )
            merged["feature"] = column
            merged = merged.rename(
                columns={f"{model_name}_mae": "mae", f"{model_name}_mse": "mse"}
            )
            results.append(merged[["feature", "unique_id", "mae", "mse"]])

        metrics_df = pd.concat(results, ignore_index=True)
        print("Validation results:\n", metrics_df)

        return metrics_df
    
    # Unused
    def get_signature(self) -> ModelSignature: # Defining a signature is kind of a bait, mlflow is REALLY restrictive about it but leaving this here just in case
        """
        Get the model signature for MLflow logging.
        
        Returns:
            ModelSignature: Signature of the model inputs and outputs.
        """
        from mlflow.models import infer_signature

        example_input = {"h:": 10, "X": pd.DataFrame(), "level": [95]}

        # if self.exog_df is not None:
        #     example_input = (self.exog_df.head(1).copy(), {"h": 10})
        # else:
        #     example_input = (pd.DataFrame(), {"h": 10})
        # return infer_signature(example_input, pd.DataFrame(columns=self.feature_columns))

        # Create example input as single DataFrame with metadata
        if self.exog_df is not None:
            example_input["X"] = self.exog_df.head(1).copy()
    
        return infer_signature(example_input, pd.DataFrame(columns=self.feature_columns))
