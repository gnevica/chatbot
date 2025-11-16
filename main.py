from langchain_experimental.agents import create_csv_agent
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_executor import AgentExecutor

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df2['ds'], df2['y'], label='Actual', color='black', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='purple', marker='s')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='violet', alpha=0.3)

    ax.set_title(f"Forecast of {target}")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target)
    plt.xticks(rotation=45)
    ax.grid()
    ax.legend()
    plt.tight_layout()

    merged = forecast[['ds', 'yhat']].merge(df2, on='ds', how='inner')
    result_text = ""
    if len(merged) > 10:
        mae = np.mean(np.abs(merged['y'] - merged['yhat']))
        rmse = np.sqrt(np.mean((merged['y'] - merged['yhat']) ** 2))
        r2 = 1 - (np.sum((merged['y'] - merged['yhat']) ** 2) / np.sum((merged['y'] - np.mean(merged['y'])) ** 2))
        result_text += f"\n**Accuracy:** MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}\n"

    if "highest" in user_input:
        row = forecast.loc[forecast['yhat'].idxmax()]
        result_text += f"\nHighest forecast → {row['ds'].date()} : {row['yhat']:.2f}"
    elif "lowest" in user_input:
        row = forecast.loc[forecast['yhat'].idxmin()]
        result_text += f"\nLowest forecast → {row['ds'].date()} : {row['yhat']:.2f}"
    else:
        result_text += "\nForecasted values:\n"
        for _, row in forecast.tail(periods).iterrows():
            result_text += f"{row['ds'].date()} : {row['yhat']:.2f}\n"

    return fig, result_text


# ----------------- HELPERS -----------------
def is_csv_related(q: str) -> bool:
    keywords = [
        "column", "row", "data", "csv", "table", "mean", "sum", "average",
        "plot", "graph", "null", "missing", "max", "min", "count", "value",
        "filter", "sort", "dataset"
    ]
    return any(k in q.lower() for k in keywords)


def is_forecasting_query(q: str) -> bool:
    return any(k in q.lower() for k in ["forecast", "predict", "future", "next year", "next month"])


def detect_target_column(df, q):
    for col in df.columns[1:]:
        if col.lower() in q.lower():
            return col
    return df.columns[1]


def detect_forecast_periods(q):
    m = re.search(r"next (\d+) (year|month)", q.lower())
    if not m:
        return 60
    num, unit = int(m.group(1)), m.group(2)
    return num * 12 if "year" in unit else num


def detect_datetime_column(df):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except:
                continue
    return df.columns[0]


# ----------------- MAIN APP -----------------
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Chatbot")
    st.header("Ask your AI Chatbot")

    csv_file = st.file_uploader("Upload CSV", type="csv")

    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.getvalue())
            path = tmp.name

        df = pd.read_csv(path)

        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
        )

        # ⭐ WORKING CSV AGENT (NO AgentExecutor!)
        agent = create_csv_agent(
            llm=llm,
            path=path,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True   # ★ FIXES YOUR ORIGINAL ERROR
        )

        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Processing..."):
                try:
                    if is_forecasting_query(question):
                        st.write("🔮 Forecasting...")

                        time_col = detect_datetime_column(df)
                        target = detect_target_column(df, question)
                        periods = detect_forecast_periods(question)

                        fig, text = forecast_with_extremes(df, question, target, time_col, periods)
                        st.pyplot(fig)
                        st.text(text)

                    elif is_csv_related(question):
                        with contextlib.redirect_stdout(io.StringIO()):
                            response = agent.run(question)

                        st.write(response)

                        if plt.get_fignums():
                            st.pyplot(plt.gcf())
                            plt.clf()

                    else:
                        resp = llm.invoke(question)
                        st.write(resp.content)

                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
