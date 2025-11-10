import numpy as np # type: ignore
import pandas as pd # type: ignore
import io
import os
from typing import Optional, Any, List, Dict
from sklearn.impute import KNNImputer # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # type: ignore
from trace_utils import trace_df_operation, trace_dataframe, trace_operation, trace_error, TRACE_ENABLED

# ===== DIAGNOSTIC TRACING =====
DEBUG_PAYLOAD_TRACE = os.getenv("DEBUG_PAYLOAD_TRACE", "0") in {"1", "true", "TRUE"}

def _trace_data_utils(tag: str, message: str):
    """Diagnostic trace logging for data_utils transformations"""
    if DEBUG_PAYLOAD_TRACE:
        print(f"[DATA_UTILS_{tag}] {message}")
# ===== END DIAGNOSTIC TRACING =====


def strip_timezones(df: Optional[pd.DataFrame]) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Remove timezone information from datetime columns and index.

    Returns the (possibly mutated) dataframe along with metadata describing what was adjusted.
    """
    info: Dict[str, Any] = {"index": False, "columns": []}
    if df is None:
        return df, info

    # ===== TRACE: Before timezone strip =====
    if DEBUG_PAYLOAD_TRACE and isinstance(df.index, pd.DatetimeIndex):
        unique_before = df.index.nunique()
        sample_before = list(df.index[:3])
        has_tz = df.index.tz is not None
        _trace_data_utils("STRIP_TZ_BEFORE", f"unique={unique_before} has_tz={has_tz} sample={sample_before}")

    # Handle datetime index with timezone
    try:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            try:
                df.index = df.index.tz_convert(None)
            except (TypeError, AttributeError, ValueError):
                df.index = df.index.tz_localize(None)
            info["index"] = True
    except Exception:
        pass

    # ===== TRACE: After timezone strip =====
    if DEBUG_PAYLOAD_TRACE and isinstance(df.index, pd.DatetimeIndex):
        unique_after = df.index.nunique()
        sample_after = list(df.index[:3])
        _trace_data_utils("STRIP_TZ_AFTER", f"unique={unique_after} sample={sample_after}")
        if unique_after != unique_before:
            _trace_data_utils("STRIP_TZ_COLLAPSE", f"❌ DETECTED: unique count changed from {unique_before} to {unique_after}!")

    # Handle dataframe columns that are timezone-aware datetimes
    adjusted_cols: List[str] = []
    for col in df.columns:
        try:
            series = df[col]
            if pd.api.types.is_datetime64tz_dtype(series):
                try:
                    df[col] = series.dt.tz_convert(None)
                except (TypeError, AttributeError, ValueError):
                    df[col] = series.dt.tz_localize(None)
                adjusted_cols.append(col)
        except Exception:
            continue

    info["columns"] = adjusted_cols
    return df, info

def read_data(file_input, filter: Optional[str] = None) -> pd.DataFrame:
    '''
    Reads a csv file with optional filtering by identifier
    '''
    try:
        # Check if input is a BytesIO object
        if isinstance(file_input, io.BytesIO):
            # Reset position to beginning of stream
            file_input.seek(0)
            df = pd.read_csv(file_input)
        
        # Check if input is a file path string
        elif isinstance(file_input, str):
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"File not found: {file_input}")
            df = pd.read_csv(file_input)
        
        # Handle other file-like objects
        else:
            df = pd.read_csv(file_input)
        
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV: {e}")
    except UnicodeDecodeError:
        # Try different encodings
        try:
            if isinstance(file_input, io.BytesIO):
                file_input.seek(0)
                df = pd.read_csv(file_input, encoding='latin-1')
            else:
                df = pd.read_csv(file_input, encoding='latin-1')
            return df
        except:
            if isinstance(file_input, io.BytesIO):
                file_input.seek(0)
                df = pd.read_csv(file_input, encoding='cp1252')
            else:
                df = pd.read_csv(file_input, encoding='cp1252')
            return df
    except Exception as e:
        raise ValueError(f"Error reading CSV data: {e}")

    try:
        df['time'] = pd.to_datetime(df['time'])
    except:
        raise ValueError('No column labeled \'time\'')

    # Selects only data for user-defined identifer
    if filter is not None:
        df = df.loc[df['District'] == filter]
        df.drop(['District'], axis=1, inplace=True)
        if isinstance(df, pd.Series):
            df = df.to_frame().T  # Ensure return type is always DataFrame
    
    # Index is set to the time so it doesn't clog other functions that can only operate on numerical data types
    try:
        df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
        df.drop(['time'], axis=1, inplace=True)
    except:
        raise Exception("Error: Could not set 'time' column as index")

    # Casts all numeric values to float64 for minimal data loss in the case of high precision values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    return df

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().select_dtypes(include=np.number)

def handle_nans(df: pd.DataFrame, threshold: float = 0.33, window: int = 2, no_drop: bool = True) -> pd.DataFrame:
    """
    Handle NaN values in the DaSaFrame by dropping rows with too many NaNs.
    Threshold is the highest percentage of NaNs allowed in a row.
    Rows with more than this percentage of NaNs will be dropped.
    Remaining NaNs will be filled by KNN imputation.
    Window is the number of nearby non-NaN values used in imputation.
    """
    df_return = select_numeric(df)
    features = df_return.columns

    # Drop rows with too many NaNs
    if not no_drop:
        thresh_not_missing = np.ceil(len(features) * (1 - threshold))
        df_return.dropna(axis=0, thresh=thresh_not_missing, inplace=True)

    # Drop columns with only NaNs
    df_return.dropna(axis=1, how='all', inplace=True)

    # Impute missing values via KNNImputer (sci-kit learn)
    for col in features:
        imputer = KNNImputer(n_neighbors=window, weights='distance', copy=True)
        array_col = df_return[col].to_numpy().reshape(-1, 1)
        preserve_index = df_return[col].index
        df_return[col] = pd.Series(imputer.fit_transform(array_col).flatten(), index=preserve_index)

    return df_return

def scale_data(df: pd.DataFrame, scale: Any="StandardScaler") -> tuple[pd.DataFrame, Any]:
    df_return = select_numeric(df)
    
    if scale is None:
        return df_return, None

    if isinstance(scale, str):
        match scale:
            case 'StandardScaler':
                scaler = StandardScaler()
            case 'MinMaxScaler':
                scaler = MinMaxScaler()
            case 'RobustScaler':
                scaler = RobustScaler()
            case 'MaxAbsScaler':
                scaler = MaxAbsScaler()
            case _:
                raise ValueError('Argument passed is not a supported scaler')
    else:
        if not isinstance(scale, (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)):
            raise TypeError('Argument passed is not a supported scaler')
        scaler = scale
        
    df_return = pd.DataFrame(scaler.fit_transform(df_return), index=df.index, columns=df.columns)

    return df_return, scaler

def clip_outliers(df: pd.DataFrame, cols: Optional[List] = None, method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
    """
    Clip outliers in specified columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list, optional
        List of columns to clip. If None, all numeric columns are clipped.
    method : str, default "iqr"
        Method to detect outliers: "iqr" or "percentile".
    factor : float, default 1.5
        Factor for IQR method, or percentile range (e.g., 0.01 → clip 1st/99th percentile).

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped values.
    """
    df_clipped = select_numeric(df)
    if cols is None:
        cols = df_clipped.columns.tolist()
    
    for col in cols:
        if method == "iqr":
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
        elif method == "percentile":
            lower = df_clipped[col].quantile(factor)
            upper = df_clipped[col].quantile(1 - factor)
        else:
            raise ValueError("method must be 'iqr' or 'percentile'")
        
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
    
    return df_clipped

def generate_lags(df: pd.DataFrame, n_lags: int, step: int=1) -> pd.DataFrame:
    '''
    Generates duplicate n_lags duplicate features that are lagged by step
    '''
    df_n = df.copy()
    columns = df.columns
    sampling = check_uniform(df)
    dfs = []

    for n in range(step, step*n_lags + 1, step):
        for col in columns:
            df_n[f"{col}_lag(-{n})"] = df_n[col].shift(n, freq=sampling)
    df_n = df_n.iloc[n_lags:]

    dfs.append(df_n)
    df_return = pd.concat(dfs, ignore_index=False)

    return df_return

@trace_df_operation
def check_uniform(df: pd.DataFrame) -> pd.Timedelta:
    '''
    Assuming a dataframe with a datetime index, this function checks for uniformity
    in sampling and returns the most common Timedelta
    '''
    trace_dataframe("check_uniform_entry", df, {}, "check_uniform")

    # Ensure the index is a DatetimeIndex for reliable Timedelta operations
    if not isinstance(df.index, pd.DatetimeIndex):
        trace_error("check_uniform", TypeError("Index not DatetimeIndex"), index_type=str(type(df.index)))
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    
    time_diffs: pd.Series = df.index.to_series().diff()
    trace_operation("time_diffs_calculated", func="check_uniform", 
                   diffs_count=len(time_diffs), non_null_count=int(time_diffs.notna().sum()))
    
    time_diffs_mode = time_diffs.mode()

    if not time_diffs_mode.empty:
        # Explicitly extract the first element and cast it to pd.Timedelta.
        most_common_frequency = pd.Timedelta(time_diffs_mode.iloc[0])
        trace_operation("frequency_detected", func="check_uniform",
                       frequency_str=str(most_common_frequency), 
                       frequency_seconds=float(most_common_frequency.total_seconds()))
    else:
        # If there's less than 2 data points, diff() will be all NaT or empty,
        # leading to an empty mode.
        trace_error("check_uniform", ValueError("Cannot determine frequency"), 
                   time_diffs_len=len(time_diffs), mode_empty=True)
        raise ValueError('Cannot determine most common frequency, not enough valid time differences.')

    # Safeguard against zero frequency (would cause division by zero in pd.date_range)
    if most_common_frequency == pd.Timedelta(0) or most_common_frequency.total_seconds() == 0:
        trace_error("check_uniform", ValueError("Zero frequency detected"),
                   frequency=str(most_common_frequency), 
                   index_sample=[str(ts) for ts in df.index[:10].tolist()],
                   unique_timestamps=int(df.index.nunique()))
        raise ValueError('Time frequency is zero - all timestamps are identical. Cannot generate date range.')

    tolerance = pd.Timedelta(milliseconds=10)
    time_diffs[abs(time_diffs - most_common_frequency) <= tolerance] = most_common_frequency

    diff_counts = time_diffs.value_counts().sort_index()

    if diff_counts.empty:
        trace_error("check_uniform", ValueError("No valid diffs"), diff_counts_empty=True)
        raise ValueError('No valid time differences found to calculate frequency.')

    total_observations = len(time_diffs)
    count_most_common = diff_counts.loc[most_common_frequency]
    percentage_most_common = (count_most_common / total_observations) * 100
        
    if percentage_most_common < 75:
        print(f"Most common frequency accounts for {percentage_most_common:.2f}% of the time steps.")
        print('Warning: sampling frequency is highly irregular. Resampling is strongly recommended')
        trace_operation("irregular_frequency_warning", func="check_uniform",
                       percentage=float(percentage_most_common), threshold=75)
    elif percentage_most_common < 98:
        print(f"Most common frequency accounts for {percentage_most_common:.2f}% of the time steps.")
        print('Warning: sampling frequency is irregular. Resampling is recommended')
        trace_operation("irregular_frequency_info", func="check_uniform",
                       percentage=float(percentage_most_common), threshold=98)

    return most_common_frequency

@trace_df_operation
def time_to_feature(df: pd.DataFrame):
    '''
    Creates features for cyclical representations of time to help ML models identify seasonality.
    Cyclic representations are necessary to show that 23:59 is very close to 0:00
    '''
    # ===== TRACE: Before time_to_feature =====
    if DEBUG_PAYLOAD_TRACE and isinstance(df.index, pd.DatetimeIndex):
        unique_before = df.index.nunique()
        sample_before = list(df.index[:3])
        _trace_data_utils("TIME_TO_FEATURE_BEFORE", f"unique={unique_before} rows={len(df)} sample={sample_before}")
    
    trace_dataframe("time_to_feature_entry", df, {}, "time_to_feature")
    
    df_return = df.copy()
    trace_operation("after_copy", func="time_to_feature", shape=list(df_return.shape))

    if not isinstance(df.index, pd.DatetimeIndex):
        trace_error("time_to_feature", TypeError("Index not DatetimeIndex"), index_type=str(type(df.index)))
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    
    # ===== TRACE: After copy =====
    if DEBUG_PAYLOAD_TRACE:
        unique_after_copy = df_return.index.nunique()
        _trace_data_utils("TIME_TO_FEATURE_AFTER_COPY", f"unique={unique_after_copy}")

    df_return = (
        df_return
        .assign(min_of_day=(df.index.hour*60 + df.index.minute))
        .assign(day_of_week=df.index.dayofweek)
        .assign(day_of_year=df.index.dayofyear)
    )
    
    trace_operation("after_assign", func="time_to_feature", shape=list(df_return.shape), columns=list(df_return.columns))

    # ===== TRACE: After assign =====
    if DEBUG_PAYLOAD_TRACE:
        unique_after_assign = df_return.index.nunique()
        _trace_data_utils("TIME_TO_FEATURE_AFTER_ASSIGN", f"unique={unique_after_assign}")

    time_features = {'min_of_day': 1440, 'day_of_week': 7, 'day_of_year': 365.25}

    for feature, period in time_features.items():
        df_return[f"{feature}_sin"] = np.sin((df_return[feature]) * (2 * np.pi / period))
        df_return[f"{feature}_cos"] = np.cos((df_return[feature]) * (2 * np.pi / period))
        df_return = df_return.drop(columns=[feature])

    # ===== TRACE: After feature engineering =====
    if DEBUG_PAYLOAD_TRACE:
        unique_final = df_return.index.nunique()
        sample_final = list(df_return.index[:3])
        _trace_data_utils("TIME_TO_FEATURE_FINAL", f"unique={unique_final} rows={len(df_return)} cols={len(df_return.columns)} sample={sample_final}")
    
    trace_dataframe("time_to_feature_exit", df_return, {}, "time_to_feature")

    return df_return

def window_data(df: pd.DataFrame, exo_features: Optional[List[str]] = None, input_len: int = 10, output_len: int = 1) -> tuple:
    """
    Splits the DataFrame into input features and target variables.
    """
    n_rows = df.shape[0]

    if exo_features is not None:
        x_df = df.copy()
        y_df = df.drop(columns=exo_features)
    else:
        x_df = df.copy()
        y_df = df
    
    x = np.zeros((n_rows, input_len, x_df.shape[1]), dtype=np.float32)
    y = np.zeros((n_rows, output_len, y_df.shape[1]), dtype=np.float32)

    for row in range(n_rows - input_len - output_len + 1):
        x[row, :input_len, :] = x_df.iloc[row:row+input_len, :].values
        y[row, :output_len, :] = y_df.iloc[row+input_len:row+input_len+output_len, :].values
    
    return x, y

def _fix_zero_scale(scaler, scaler_type_name="Scaler"):
    """
    Fix division by zero issue in sklearn scalers by replacing zero scale_ values with 1.0.
    
    When a feature has zero variance in training data, sklearn sets scale_ to 0.
    During inverse_transform, it divides by scale_, causing ZeroDivisionError.
    This function prevents that by replacing 0 with 1.0 (no scaling).
    
    Args:
        scaler: The scaler object to fix (modifies in place)
        scaler_type_name: Name for logging purposes
        
    Returns:
        The fixed scaler (same object, modified in place)
    """
    if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        zero_scale_mask = scaler.scale_ == 0
        if np.any(zero_scale_mask):
            scaler.scale_ = scaler.scale_.copy()  # Ensure we're not modifying shared array
            scaler.scale_[zero_scale_mask] = 1.0
            print(f"[Warning] {scaler_type_name} has {np.sum(zero_scale_mask)} features with zero variance/range (scale_=0). "
                  f"Replaced with 1.0 to prevent division by zero during inverse_transform.")
    return scaler


def subset_scaler(original_scaler, original_columns, subset_columns):
    """
    Creates a new scaler for a subset of features from an existing fitted scaler.
    Supports StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, and LogScaler.

    Args:
        original_scaler: The fitted scaler object.
        original_columns (list): The column names from the original DataFrame.
        subset_columns (list): A list of column names for the new scaler.

    Returns:
        A new, configured scaler for the subset of data.
    """
    if original_columns == subset_columns:
        # Return the original scaler, but fix zero scale_ values first
        return _fix_zero_scale(original_scaler, scaler_type_name=original_scaler.__class__.__name__)

    # Find the integer indices of the subset columns
    subset_indices = [original_columns.index(col) for col in subset_columns]

    if isinstance(original_scaler, StandardScaler):
        subset = StandardScaler()
        if original_scaler.mean_ is not None:
            subset.mean_ = original_scaler.mean_[subset_indices]
        else:
            subset.mean_ = None
        if original_scaler.scale_ is not None:
            subset.scale_ = original_scaler.scale_[subset_indices].copy()
        else:
            subset.scale_ = None

    elif isinstance(original_scaler, MinMaxScaler):
        subset = MinMaxScaler()
        subset.min_ = original_scaler.min_[subset_indices]
        subset.scale_ = original_scaler.scale_[subset_indices].copy()
        subset.data_min_ = original_scaler.data_min_[subset_indices]
        subset.data_max_ = original_scaler.data_max_[subset_indices]
        subset.data_range_ = original_scaler.data_range_[subset_indices]

    elif isinstance(original_scaler, RobustScaler):
        subset = RobustScaler()
        subset.center_ = original_scaler.center_[subset_indices]
        subset.scale_ = original_scaler.scale_[subset_indices].copy()

    elif isinstance(original_scaler, MaxAbsScaler):
        subset = MaxAbsScaler()
        subset.max_abs_ = original_scaler.max_abs_[subset_indices]
        subset.scale_ = original_scaler.scale_[subset_indices].copy()

    else:
        raise TypeError("Unsupported scaler type. This function only supports "
                        "StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, and LogScaler.")

    # Set feature info for scikit-learn validation
    subset.n_features_in_ = len(subset_columns)
    subset.feature_names_in_ = np.array(subset_columns, dtype=object)

    # Fix zero scale_ values to prevent division by zero during inverse_transform
    _fix_zero_scale(subset, scaler_type_name=subset.__class__.__name__)

    return subset
