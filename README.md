# EquityLens (US): Gender-Blind Wage Benchmark

**EquityLens** is a data science project that measures the gender wage gap using US CPS data and builds a **gender-blind market wage benchmark** to support fair-pay audits.  
It combines **SQL data engineering**, **econometric inference**, and a **deployed ML model** accessible through a Streamlit app.

---

## 1. Project Overview

Companies struggle to evaluate pay equity because internal comparisons mix job differences with potential discrimination.  
This project answers two key questions:

- **How much of the gender wage gap is due to job characteristics?**
- **What would be a fair, "neutral" market wage for each worker profile?**

Using more than **200,000 observations** from the Current Population Survey (CPS), the project delivers:

- A **Blau & Kahn–style (2017) adjusted wage gap estimate**, and  
- A **gender-blind machine learning benchmark wage** for auditing salaries.

---

## 2. Data Pipeline (SQL + Python)

The workflow:

1. Load CPS data into a **SQLite database**
2. Build a curated SQL view (`vw_model_cohort`)
3. Filter the cohort:
   - Valid wages and hours  
   - Ages 25–64  
   - Clean occupation and industry variables  
4. Exclude sensitive or irrelevant fields (race, marital status, sex for ML)

This produces a **clean and reproducible dataset** for inference and modeling.

---

## 3. Gender Wage Gap Inference (Notebook 3)

A simplified **Blau & Kahn decomposition** is implemented using OLS.

### **Model A: Job Factors Only**
Controls include:
- Education  
- Potential experience  
- Occupation (22 dummies)  
- Industry (15 dummies)  
- Hours, union status, class of worker, full-time  

**R² ≈ 0.40**

### **Model B: Job Factors + Gender**
Adds only a `female` indicator.

- Coefficient: **–0.2097**
- Adjusted pay gap: **≈ –19%**
- Gender contributes **ΔR² ≈ +0.02**

A Ridge stability check shows the same effect (≈ –19%), confirming the result is not due to multicollinearity.

**Conclusion:** a substantial gender wage gap persists even after controlling for job characteristics.

---

## 4. Gender-Blind ML Benchmark (Notebook 4)

A Ridge Regression model predicts log real hourly wage using only job characteristics:

- Age  
- Hours and annual hours  
- Education (educ99, BA, ADV)  
- Potential experience (potexp + potexp²)  
- Occupation and industry dummies  
- Union status, class of worker, full-time indicator  

**Performance on the test year:**

- **R² ≈ 0.41**
- **MAE stable across train/test**

Random Forest confirms that wages follow a mostly linear structure, validating the choice of Ridge.

The resulting model serves as a **gender-blind market benchmark**.

---

## 5. Streamlit Application

`streamlit_app.py` provides two functionalities:

### **A. CSV Workforce Audit**
- Upload a dataset of workers  
- The app generates **benchmark wages** using the gender-blind model  
- Extra columns are dropped; missing model features are filled with zeros  
- Output includes a downloadable results CSV

### **B. Single-Worker Benchmark**
Interactive form that computes:
- Predicted log-wage  
- Benchmark hourly wage  

Artifacts used:
- `ridge_model.pkl`
- `feature_list.json`
- `model_metrics.json`

---


Run the Streamlit app:

pip install -r requirements.txt
`streamlit run streamlit_app.py`

## 6. Repository Structure

```text
notebooks/
  1_sql_build.ipynb
  2_eda_and_stats.ipynb
  3_gender_gap_inference.ipynb
  4_ml_model.ipynb

sql/
  01_load_curate.sql

artifacts/
  ridge_model.pkl
  feature_list.json
  model_metrics.json
  inference_metrics.json

streamlit_app.py
requirements.txt
README.md

