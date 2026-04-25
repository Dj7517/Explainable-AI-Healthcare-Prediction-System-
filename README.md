# 🫀 Explainable AI Healthcare Prediction

An **Explainable AI (XAI)** Streamlit application for predicting heart disease risk using multiple ML models with SHAP-based explanations.

---

## 🚀 Features

- **Multi-Model Prediction**: XGBoost, Random Forest, Gradient Boosting, Logistic Regression
- **SHAP Explanations**: Per-patient feature contribution analysis
- **Interactive Dashboard**: Real-time risk gauge, SHAP bar charts
- **Model Analytics**: Accuracy & AUC comparison across all models
- **Dataset Explorer**: Distributions, correlations, and raw data

---

## 📂 Project Structure

```
xai_healthcare/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── README.md
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator
│   └── heart_disease_dataset.csv # Dataset (1000 patients)
└── utils/
    └── model_utils.py            # Model training, SHAP, prediction utilities
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset (already included)
```bash
python data/generate_dataset.py
```

### 3. Launch Streamlit app
```bash
streamlit run app.py
```

---

## 🧬 Features Used

| Feature | Description |
|---------|-------------|
| Age | Patient age (20–80) |
| Gender | Male / Female |
| Blood Pressure | Systolic BP in mmHg |
| Cholesterol | Normal / Above Normal / Well Above Normal |
| BMI | Body Mass Index |
| Glucose Level | Normal / Above Normal / Well Above Normal |
| Smoking | Yes / No |
| Alcohol Intake | Yes / No |
| Physical Activity | Yes / No |

---

## 🔬 How XAI Works

**SHAP (SHapley Additive exPlanations)** decomposes each prediction into feature-level contributions:
- **Positive SHAP value** → Feature increases heart disease risk
- **Negative SHAP value** → Feature decreases heart disease risk
- Magnitude indicates strength of contribution

---

## 📊 Model Performance (typical)

| Model | Accuracy | AUC |
|-------|----------|-----|
| XGBoost | ~89% | ~0.93 |
| Random Forest | ~87% | ~0.91 |
| Gradient Boosting | ~88% | ~0.92 |
| Logistic Regression | ~82% | ~0.87 |

---

## 🔧 Using Your Own Dataset

Replace `data/heart_disease_dataset.csv` with your own CSV. Ensure it contains:
- All 9 feature columns (see table above)
- A `heart_disease` target column (0 = No, 1 = Yes)

Then update `FEATURE_NAMES` in `utils/model_utils.py` if needed.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It should not replace professional medical advice.
