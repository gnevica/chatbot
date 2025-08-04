import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------- HELPER FUNCTIONS ----------------------

def is_forecast_query(question: str) -> bool:
    """Check if the query is about forecasting."""
    forecast_keywords = ["forecast", "predict", "projection", "future", "next"]
    return any(keyword in question.lower() for keyword in forecast_keywords)

def extract_periods_from_query(query: str) -> int:
    """Extract forecast periods from query like 'next 5 years' or 'next 12 months'."""
    match = re.search(r'next\s+(\d+)', query.lower())
    if match:
        return int(match.group(1))
    return 12  # Default 12 periods

def detect_best_numeric_column(df: pd.DataFrame, query: str) -> str:
    """Select the best numeric column for forecasting based on the query keywords."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        raise ValueError("❌ No numeric column found to forecast!")
    for col in numeric_cols:
        if col.lower() in query.lower():
            return col
    return numeric_cols[0]  # fallback

def detect_date_column(df: pd.DataFrame) -> str:
    """Auto-detect a date column (any column with datetime or containing 'date'/'year')."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "year" in col.lower():
            return col
    return None

def compute_metrics(actual: pd.Series, predicted: pd.Series):
    """Compute MAE, RMSE, and R² Score."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2

def detect_date_frequency(df: pd.DataFrame, date_col: str) -> str:
    """
    Detects date frequency (D for daily, M for monthly, Y for yearly)
    using pandas infer_freq or median difference fallback.
    """
    sorted_dates = df[date_col].sort_values().dropna()
    inferred_freq = pd.infer_freq(sorted_dates)

    # Manual fallback if pandas can't infer
    if inferred_freq is None:
        diffs = sorted_dates.diff().dropna().dt.days
        median_diff = diffs.median()
        if median_diff <= 2:
            return 'D'  # daily
        elif median_diff <= 40:
            return 'M'  # monthly
        else:
            return 'Y'  # yearly
    else:
        # Map inferred freq to Prophet freq
        if inferred_freq[0] in ['D']:
            return 'D'
        elif inferred_freq[0] in ['M']:
            return 'M'
        else:
            return 'Y'

# ---------------------- MAIN FORECAST FUNCTION ----------------------

def perform_forecast(df: pd.DataFrame, query: str):
    # --- 1. Detect date column ---
    date_col = detect_date_column(df)
    if not date_col:
        raise ValueError("❌ No suitable date column found for forecasting!")

    # --- 2. Detect numeric column ---
    target_col = detect_best_numeric_column(df, query)

    # --- 3. Prepare data ---
    df[date_col] = pd.to_datetime(df[date_col])
    prophet_df = df[[date_col, target_col]].dropna().rename(columns={date_col: "ds", target_col: "y"})

    if prophet_df.empty:
        raise ValueError("❌ Not enough valid data to forecast!")

    # --- 4. Detect frequency ---
    freq = detect_date_frequency(df, date_col)
    periods = extract_periods_from_query(query)
    freq_map = {'D': 'D', 'M': 'M', 'Y': 'Y'}
    prophet_freq = freq_map.get(freq, 'Y')

    # --- 5. Train Prophet model ---
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=prophet_freq)
    forecast = model.predict(future)

    # --- 6. Compute metrics ---
    metrics_text = ""
    try:
        merged = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='inner')
        if len(merged) > 5:
            mae, rmse, r2 = compute_metrics(merged['y'], merged['yhat'])
            metrics_text += f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f} | **R²:** {r2:.2f}\n\n"
    except Exception:
        metrics_text += "Metrics not available.\n\n"

    # --- 7. Identify extremes ---
    max_row = forecast.loc[forecast['yhat'].idxmax()]
    min_row = forecast.loc[forecast['yhat'].idxmin()]
    metrics_text += f"""
    **Highest** `{target_col}`: {max_row['yhat']:.2f} on {max_row['ds'].date()}  
    **Lowest** `{target_col}`: {min_row['yhat']:.2f} on {min_row['ds'].date()}
    """

    # --- 8. Plot custom forecast ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prophet_df['ds'], prophet_df['y'], label='Actual', color='blue', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='darkorange', marker='D')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='plum', alpha=0.3, label='Confidence Interval')
    ax.set_title(f"Forecast of {target_col} for Next {periods} {freq}", fontsize=14, fontweight='bold')
    ax.set_xlabel(date_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # --- 9. Prepare forecast table ---
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_df.rename(columns={'ds': date_col, 'yhat': target_col}, inplace=True)

    return forecast_df, fig, metrics_text
