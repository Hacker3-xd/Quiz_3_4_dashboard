Quiz 3 & 4 Dashboard — Bitcoin Forecasting & EDA

An interactive data visualization dashboard built with Streamlit for Bitcoin price forecasting and exploratory data analysis (EDA), developed as part of the Advanced Data Visualization course.

---

## 📁 Project Structure
├── app.py                        # Main entry point
├── quiz3_dashboard.py            # Quiz 3: EDA Dashboard
├── quiz4_dashboard.py            # Quiz 4: ML Forecasting Dashboard
├── eda_generator.py              # EDA utility functions
├── ml_forecasting.py             # ML model logic (ARIMA, Linear Regression)
├── utils.py                      # Shared helper functions
├── data/                         # Raw datasets
├── models/                       # Saved ML models
├── diagrams/                     # Generated plots/diagrams
├── bitcoin_arima_forecast_30d.csv
├── bitcoin_arima_forecast_60d.csv
├── bitcoin_arima_forecast_90d.csv
├── bitcoin_lr_predictions.csv
├── model_metrics.csv
├── requirements.txt



---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Hacker3-xd/Quiz_3_4_dashboard.git
cd Quiz_3_4_dashboard
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

---

## 📊 Features

### Quiz 3 — Exploratory Data Analysis
- Interactive EDA on Bitcoin price data
- Missing value analysis with Missingno
- Statistical summaries and correlation heatmaps
- Distribution plots and time-series visualization

### Quiz 4 — ML Forecasting
- Bitcoin price forecasting using **ARIMA** (30, 60, 90 days)
- **Linear Regression** predictions
- Model explainability with **SHAP** and **LIME**
- Performance metrics comparison

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Dashboard UI |
| Pandas / NumPy | Data processing |
| Plotly / Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML models |
| Statsmodels | ARIMA forecasting |
| SHAP / LIME | Model explainability |
| Missingno | Missing data visualization |
| SciPy / NetworkX | Statistical & graph analysis |

---

## 👨‍🎓 Course Info

- **Course:** Advanced Data Visualization  
- **Instructor:** Dr. Muhammad Zeeshan  
- **Institution:** Pak-Austria Fachhochschule  
- **Semester:** Master in AI — Semester 2

---

## 📬 Author

**Hacker3-xd**  
GitHub: [https://github.com/Hacker3-xd](https://github.com/Hacker3-xd)
