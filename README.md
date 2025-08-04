# AI-Powered CSV Analysis, Forecasting & General Query Chatbot

## Project Overview

This project is an **AI-powered Streamlit chatbot** that allows users to:

1. **Upload CSV files** and analyze them using **natural language queries**.  
2. **Generate visualizations and statistics** without writing any code.  
3. **Perform time-series forecasting** with Prophet, including confidence intervals and evaluation metrics (MAE, RMSE, R²).  
4. **Answer general knowledge queries** using Azure OpenAI GPT.

The chatbot leverages **LangChain**, **Azure OpenAI**, **Prophet**, **Pandas**, and **Matplotlib** to make **data analysis and forecasting accessible to non-technical users**.  


## Features

- **CSV Data Analysis**:
  - Summarize, filter, and explore datasets.
  - Detect missing values, min/max, and descriptive statistics.

- **Forecasting**:
  - Automatically performs **time-series forecasting** using Prophet.
  - Generates plots with **confidence intervals (yhat, lower, upper)**.
  - Provides **evaluation metrics** like MAE, RMSE, and R².

- **Visualization**:
  - Dynamic **Matplotlib plots** rendered in Streamlit.
  - Auto-generated trend and forecast charts.

- **Query Types Supported**:
  1. **CSV-related queries** (e.g., “What is the average rainfall in 2020?”)
  2. **Forecasting queries** (e.g., “Forecast rainfall for the next 5 years”)
  3. **General queries** (e.g., “What is AI?”)

- **General AI Q&A**:
  - Uses **Azure GPT via LangChain** for general knowledge answers.


##  Tech Stack

- **Frontend & Deployment**: Streamlit  
- **Language Model**: Azure OpenAI (GPT) via LangChain  
- **Data Analysis & Forecasting**: Pandas, NumPy, Prophet, Matplotlib  
- **Metrics**: scikit-learn (MAE, RMSE, R²)  
- **Environment Management**: python-dotenv, Streamlit Secrets

---

## Project Structure

```
.
├── main.py                # Main Streamlit chatbot application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .streamlit/
    └── secrets.toml       # Streamlit Cloud secrets (API keys)
```


## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/AI-CSV-Forecast-Chatbot.git
   cd AI-CSV-Forecast-Chatbot
   ```

2. **Create Virtual Environment (Optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   - Create a `.env` file or use **Streamlit Secrets** with:
     ```
     AZURE_OPENAI_API_KEY=your_key
     AZURE_OPENAI_API_BASE=your_endpoint
     AZURE_OPENAI_API_VERSION=2024-02-15-preview
     AZURE_OPENAI_DEPLOYMENT_NAME=your_model
     ```

5. **Run Streamlit App**
   ```bash
   streamlit run main.py
   ```

6. **Open in Browser**
   - Default: `http://localhost:8501`

---

## ☁Deployment on Streamlit Cloud

1. Push your project to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app.  
3. Add your **Azure OpenAI API keys** in **Streamlit Secrets**.  
4. Deploy! Your chatbot will run 24/7.

---

##  Example Queries

- **CSV Queries:**
  - “Show me the total sales in 2021.”
  - “Find the average temperature for June.”

- **Forecasting Queries:**
  - “Forecast rainfall for the next 5 years.”
  - “Predict the lowest expected sales in the next 12 months.”

- **General Queries:**
  - “What is LangChain?”
  - “Explain time-series forecasting.”

---

##  Example Output

- **Time-Series Forecast Plot**
  - Forecast line with upper/lower bounds.
  - Confidence intervals shaded in purple.
  - Automatic evaluation metrics displayed.

---

##  Contributing

Contributions are welcome! 🎉

1. Fork the repository  
2. Create a new branch (`feature/my-feature`)  
3. Commit changes (`git commit -m "Added new feature"`)  
4. Push to your fork  
5. Create a **Pull Request**

---

##  License

This project is released under the **MIT License**.
