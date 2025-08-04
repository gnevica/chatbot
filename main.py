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

                        # Try to display the last generated plot
                        try:
                            st.pyplot(plt.gcf())
                            plt.clf()  # Clear after rendering
                        except Exception:
                            pass  # No plot was created
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------- FORECAST FUNCTION -----------------
def forecast_with_extremes(df, user_input, target, time_col, start_y=None, end_y=None, periods=60):
    df2 = df[[time_col, target]].dropna()
    df2.columns = ['ds', 'y']
    df2['ds'] = pd.to_datetime(df2['ds'], errors='coerce')
    df2 = df2.dropna()

    model = Prophet()
    model.fit(df2)

    future = (pd.date_range(start=f"{start_y}-01-01", end=f"{end_y}-12-31", freq='MS')
              if start_y and end_y else
              model.make_future_dataframe(periods=periods, freq='M')['ds'])
    
    forecast = model.predict(pd.DataFrame({'ds': future}))

    # --- Plot Forecast ---
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

    # Merge for evaluation
    merged = forecast[['ds', 'yhat']].merge(df2, on='ds', how='inner')
    result_text = ""
    if len(merged) > 10:
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
        r2 = r2_score(merged['y'], merged['yhat'])
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
        forecast['year'] = forecast['ds'].dt.year
        filtered = forecast[(forecast['year'] >= (start_y or 0)) & (forecast['year'] <= (end_y or 3000))] if start_y else forecast.tail(periods)
        result_text += f"\n**Forecasted values for '{target}':**\n"
        for _, row in filtered.iterrows():
            result_text += f"{row['ds'].date()}: {row['yhat']:.2f}\n"

    return fig, result_text


# ----------------- CSV & CHATBOT APP -----------------
def is_csv_related(question: str) -> bool:
    keywords = [
        "column", "row", "data", "csv", "table", "mean", "sum", "average", "plot",
        "graph", "null", "missing", "max", "min", "count", "value", "filter", "sort","dataset"
    ]
    return any(keyword in question.lower() for keyword in keywords)

def is_forecasting_query(question: str) -> bool:
    forecast_keywords = ["forecast", "predict", "future", "next year", "next month", "projection"]
    return any(keyword in question.lower() for keyword in forecast_keywords)


def main():
    load_dotenv()
    st.set_page_config(page_title="AI CHATBOT")
    st.header("Ask your Chatbot")

    csv_file = st.file_uploader("Upload your CSV file", type="csv")

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # Initialize Azure LLM
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
        )

        agent = create_csv_agent(
            llm=llm,
            path=csv_file,
            verbose=True,
            allow_dangerous_code=True,
        )

        user_question = st.text_input("Ask a question:")

        if user_question:
            with st.spinner("In progress..."):
                try:
                    if is_forecasting_query(user_question):
                        # ---- Forecasting Handling ----
                        st.write("🔮 Performing time series forecasting...")
                        
                        # Simple assumption: first column = time, second column = target
                        time_col = df.columns[0]
                        target_col = df.columns[1]

                        fig, forecast_text = forecast_with_extremes(df, user_question, target_col, time_col)
                        st.pyplot(fig)
                        st.text(forecast_text)

                    elif is_csv_related(user_question):
                        with contextlib.redirect_stdout(io.StringIO()):
                            response = agent.run(user_question)
                        st.write(response)

                        try:
                            st.pyplot(plt.gcf())
                            plt.clf()
                        except Exception:
                            pass
                    else:
                        response = llm.invoke(user_question)
                        st.write(response.content)
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
