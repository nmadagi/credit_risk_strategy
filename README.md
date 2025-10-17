# credit_risk_strategy
Credit Risk Strategy Model using Logistic Regression &amp; XGBoost | Streamlit Dashboard | UCI German Credit Data
# Credit Risk Strategy — German Credit Dataset  

### 🎯 Objective  
Build an **end-to-end consumer credit-risk strategy model** that predicts loan default probability (PD), computes **Expected Loss (PD × LGD × EAD)**, and defines **data-driven approval policies** with an interactive dashboard.

---

### 🚀 Project Overview  
- Developed **Probability of Default (PD)** models using **Logistic Regression** and **XGBoost**.  
- Computed **Expected Loss** and created **risk-band segmentation (AA/A/B/C)**.  
- Designed a **Streamlit dashboard** for real-time scoring and portfolio simulation.  
- Implemented caching for fast performance and quick reloads.  
- Aligned with **credit-risk governance**, **stress testing**, and **profitability analysis** principles.  

---

### 🧠 Key Features  
- **Dual-Model Comparison:** Logistic (baseline) vs. XGBoost (champion).  
- **Expected-Loss Computation:** PD × LGD × EAD with adjustable parameters.  
- **Portfolio Simulation:** Risk-band summaries and approval/review/decline strategy.  
- **Interactive Dashboard:** Sliders for applicant inputs and instant PD prediction.  
- **Explainability:** Feature importance + SHAP visualizations for model transparency.  

---

### 🧰 Tech Stack  
**Languages & Libraries:** Python | Pandas | NumPy | Scikit-learn | XGBoost | Matplotlib | Seaborn | Streamlit  
**Dataset:** [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (1 000 rows)  
**Concepts:** Credit-risk modeling | Expected-loss estimation | Risk-appetite framework | Stress testing  

---

### 📊 Streamlit App  
🔗 **Live App Demo:** *(coming soon)*  
To run locally:  
```bash
pip install -r requirements.txt
streamlit run credit_risk_german.py
