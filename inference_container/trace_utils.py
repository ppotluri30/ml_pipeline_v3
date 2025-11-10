"""Diagnostic tracing utilities for pandas DataFrame operations.

Usage:
    Set environment variable DEBUG_PANDAS_TRACE=1 to enable detailed logging
    of all DataFrame transformations, timestamp parsing, and error conditions.
"""
import os
import functools
import traceback
import json
from typing import Any, Callable, Optional
import pandas as pd


TRACE_ENABLED = os.getenv("DEBUG_PANDAS_TRACE", "0") in {"1", "true", "TRUE"}


def _format_df_summary(df: Optional[pd.DataFrame], prefix: str = "") -> dict:
    """Extract safe summary of DataFrame for logging."""
    if df is None:
        return {f"{prefix}df": "None"}
    
    try:
        summary = {
            f"{prefix}shape": list(df.shape) if hasattr(df, 'shape') else "unknown",
            f"{prefix}columns": list(df.columns) if hasattr(df, 'columns') else [],
            f"{prefix}index_type": str(type(df.index).__name__) if hasattr(df, 'index') else "unknown",
        }
        
        # Timestamp-specific diagnostics
        if isinstance(df.index, pd.DatetimeIndex):
            summary[f"{prefix}index_unique"] = int(df.index.nunique()) if hasattr(df.index, 'nunique') else 0
            summary[f"{prefix}index_sample"] = [str(ts) for ts in df.index[:3].tolist()] if len(df.index) > 0 else []
            summary[f"{prefix}index_has_tz"] = df.index.tz is not None
        
        # Check for timestamp columns
        for col in df.columns:
            if 'time' in str(col).lower() or 'ts' in str(col).lower():
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        unique_count = df[col].nunique()
                        sample_vals = [str(v) for v in df[col].head(3).tolist()]
                        summary[f"{prefix}col_{col}_unique"] = int(unique_count)
                        summary[f"{prefix}col_{col}_sample"] = sample_vals
                except Exception:
                    pass
        
        return summary
    except Exception as e:
        return {f"{prefix}error": str(e)}


def trace_dataframe(stage: str, df: Optional[pd.DataFrame], context: dict = None, func_name: str = None):
    """Log DataFrame state at a specific processing stage."""
    if not TRACE_ENABLED:
        return
    
    try:
        log_entry = {
            "trace_type": "dataframe",
            "stage": stage,
            "func": func_name or "unknown",
        }
        
        if context:
            log_entry.update(context)
        
        log_entry.update(_format_df_summary(df))
        
        print(f"[PANDAS_TRACE] {json.dumps(log_entry)}", flush=True)
    except Exception as e:
        print(f"[PANDAS_TRACE_ERROR] Failed to log DataFrame: {e}", flush=True)


def trace_operation(operation: str, **kwargs):
    """Log a specific pandas operation with context."""
    if not TRACE_ENABLED:
        return
    
    try:
        log_entry = {
            "trace_type": "operation",
            "operation": operation,
        }
        log_entry.update(kwargs)
        print(f"[PANDAS_TRACE] {json.dumps(log_entry)}", flush=True)
    except Exception:
        pass


def trace_error(func_name: str, error: Exception, **context):
    """Log error with full traceback and context."""
    if not TRACE_ENABLED:
        return
    
    try:
        log_entry = {
            "trace_type": "error",
            "func": func_name,
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "traceback": traceback.format_exc(),
        }
        log_entry.update(context)
        print(f"[PANDAS_ERROR] {json.dumps(log_entry, indent=2)}", flush=True)
    except Exception:
        print(f"[PANDAS_ERROR] {func_name}: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()


def trace_df_operation(func: Callable) -> Callable:
    """Decorator to trace DataFrame operations with entry/exit logging.
    
    Usage:
        @trace_df_operation
        def my_function(df: pd.DataFrame, ...) -> pd.DataFrame:
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not TRACE_ENABLED:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        
        # Log entry
        try:
            # Try to find DataFrame in args
            df_arg = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df_arg = arg
                    break
            
            if df_arg is None:
                for key, val in kwargs.items():
                    if isinstance(val, pd.DataFrame):
                        df_arg = val
                        break
            
            trace_dataframe("entry", df_arg, {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}, func_name)
        except Exception as e:
            print(f"[PANDAS_TRACE] Entry trace failed for {func_name}: {e}", flush=True)
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Log exit with result
            try:
                if isinstance(result, pd.DataFrame):
                    trace_dataframe("exit_success", result, {}, func_name)
                elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], pd.DataFrame):
                    trace_dataframe("exit_success", result[0], {"return_type": "tuple"}, func_name)
                else:
                    trace_operation("exit_success", func=func_name, return_type=str(type(result).__name__))
            except Exception as e:
                print(f"[PANDAS_TRACE] Exit trace failed for {func_name}: {e}", flush=True)
            
            return result
            
        except Exception as error:
            trace_error(func_name, error)
            raise
    
    return wrapper


def trace_call_stack():
    """Log current call stack for debugging."""
    if not TRACE_ENABLED:
        return
    
    try:
        stack_summary = traceback.format_stack()
        print(f"[PANDAS_TRACE] Call stack:", flush=True)
        for line in stack_summary[-10:]:  # Last 10 frames
            print(f"  {line.strip()}", flush=True)
    except Exception:
        pass
