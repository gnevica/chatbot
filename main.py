'''from langchain_experimental.agents import create_csv_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import matplotlib.pyplot as plt
import io
import contextlib

def is_csv_related(question: str) -> bool:
    """Basic keyword check to decide if the question is about the CSV."""
    keywords = [
        "column", "row", "data", "csv", "table", "mean", "sum", "average", "plot",
        "graph", "null", "missing", "max", "min", "count", "value", "filter", "sort","dataset"
    ]
    return any(keyword in question.lower() for keyword in keywords)

def main():
    load_dotenv()

    st.set_page_config(page_title="AI CHATBOT")
    st.header("Ask your Chatbot")

    csv_file = st.file_uploader("Upload your CSV file", type="csv")

    if csv_file is not None:
        # Initialize Azure LLM
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
        )

        # Create CSV agent
        agent = create_csv_agent(
            llm=llm,
            path=csv_file,
            verbose=True,
            allow_dangerous_code=True,
            #handle_parsing_errors=True
        )

        user_question = st.text_input("Ask a question:")

        if user_question:
            with st.spinner("In progress..."):
                try:
                    if is_csv_related(user_question):
                        # Capture stdout and any plot generated
                        with contextlib.redirect_stdout(io.StringIO()):
                            response = agent.run(user_question)

                        st.write(response)

                        # ✅ Only show plot if a figure exists (no white box)
                        if plt.get_fignums():
                            st.pyplot(plt.gcf())
                            plt.clf()  # Clear after rendering
                    else:
                        response = llm.invoke(user_question)
                        st.write(response.content)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
from langchain_experimental.agents import create_csv_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import matplotlib.pyplot as plt
import io
import contextlib
import pandas as pd
import numpy as np
from prophet import Prophet
import tempfile
import re


# ----------------- FORECAST FUNCTION -----------------
def forecast_with_extremes(df, user_input, target, time_col, periods=60):
    df2 = df[[time_col, target]].dropna()
    df2.columns = ['ds', 'y']
    df2['ds'] = pd.to_datetime(df2['ds'], errors='coerce')
    df2 = df2.dropna()

    model = Prophet()
    model.fit(df2)

    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df2['ds'], df2['y'], label='Actual Values', color='black', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Values', color='darkviolet', marker='s', linewidth=2)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='plum', alpha=0.3, label='Confidence Interval')

    ax.set_title(f"Forecast of {target}", fontsize=16, fontweight='bold')
    ax.set_xlabel(time_col, fontsize=12)
    ax.set_ylabel(target, fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Evaluate accuracy
    merged = forecast[['ds', 'yhat']].merge(df2, on='ds', how='inner')
    result_text = ""
    if len(merged) > 10:
        mae = np.mean(np.abs(merged['y'] - merged['yhat']))
        rmse = np.sqrt(np.mean((merged['y'] - merged['yhat']) ** 2))
        r2 = 1 - (np.sum((merged['y'] - merged['yhat']) ** 2) / np.sum((merged['y'] - np.mean(merged['y'])) ** 2))
        result_text += f"\n**Forecast Accuracy:**\n- MAE: {mae:.2f}\n- RMSE: {rmse:.2f}\n- R² Score: {r2:.4f}\n"
    else:
        result_text += "\nNot enough overlapping data to evaluate accuracy.\n"

    # Max/Min forecast
    if "highest" in user_input or "maximum" in user_input:
        max_row = forecast.loc[forecast['yhat'].idxmax()]
        result_text += f"\n**Highest forecasted '{target}'** on {max_row['ds'].date()} → {max_row['yhat']:.2f}\n"
    elif "lowest" in user_input or "minimum" in user_input:
        min_row = forecast.loc[forecast['yhat'].idxmin()]
        result_text += f"\n**Lowest forecasted '{target}'** on {min_row['ds'].date()} → {min_row['yhat']:.2f}\n"
    else:
        result_text += f"\n**Forecasted values for '{target}':**\n"
        for _, row in forecast.tail(periods).iterrows():
            result_text += f"{row['ds'].date()}: {row['yhat']:.2f}\n"

    return fig, result_text


# ----------------- HELPER FUNCTIONS -----------------
def is_csv_related(question: str) -> bool:
    keywords = [
        "column", "row", "data", "csv", "table", "mean", "sum", "average", "plot",
        "graph", "null", "missing", "max", "min", "count", "value", "filter", "sort", "dataset"
    ]
    return any(keyword in question.lower() for keyword in keywords)


def is_forecasting_query(question: str) -> bool:
    forecast_keywords = ["forecast", "predict", "future", "next year", "next month", "projection"]
    return any(keyword in question.lower() for keyword in forecast_keywords)


def detect_target_column(df, user_question: str):
    for col in df.columns[1:]:
        if col.lower() in user_question.lower():
            return col
    return df.columns[1]


def detect_forecast_periods(user_question: str) -> int:
    match = re.search(r"next (\d+) (year|month)", user_question.lower())
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if "year" in unit:
            return num * 12
        else:
            return num
    return 60


def detect_datetime_column(df: pd.DataFrame) -> str:
    """
    Detect the datetime column in the dataframe.
    Priority:
    1. Columns already parsed as datetime
    2. Columns with 'date' or 'time' in their name
    3. Fallback: first column
    """
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col

    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time"]):
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except Exception:
                continue

    return df.columns[0]


# ----------------- MAIN APP -----------------
def main():
    load_dotenv()
    st.set_page_config(page_title="AI CHATBOT")
    st.header("Ask your Chatbot")

    csv_file = st.file_uploader("Upload your CSV file", type="csv")

    if csv_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(csv_file.getvalue())
            tmp_csv_path = tmp_file.name

        df = pd.read_csv(tmp_csv_path)

        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
        )

        agent = create_csv_agent(
            llm=llm,
            path=tmp_csv_path,
            verbose=True,
            allow_dangerous_code=True,
        )

        user_question = st.text_input("Ask a question:")

        if user_question:
            with st.spinner("In progress..."):
                try:
                    if is_forecasting_query(user_question):
                        st.write("🔮 Performing time series forecasting...")
                        time_col = detect_datetime_column(df)   # <--- AUTO DETECT DATETIME COL
                        target_col = detect_target_column(df, user_question)
                        periods = detect_forecast_periods(user_question)
                        fig, forecast_text = forecast_with_extremes(df, user_question, target_col, time_col, periods)
                        st.pyplot(fig)
                        st.text(forecast_text)

                    elif is_csv_related(user_question):
                        with contextlib.redirect_stdout(io.StringIO()):
                            response = agent.run(user_question)

                        st.write(response)

                        if plt.get_fignums():
                            st.pyplot(plt.gcf())
                            plt.clf()

                    else:
                        response = llm.invoke(user_question)
                        st.write(response.content)

                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


