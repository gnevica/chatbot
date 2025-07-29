from langchain_experimental.agents import create_csv_agent
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
        "graph", "null", "missing", "max", "min", "count", "value", "filter", "sort"
    ]
    return any(keyword in question.lower() for keyword in keywords)

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

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
