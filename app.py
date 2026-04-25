import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import (
    load_data, train_models, get_shap_values,
    predict_risk, FEATURE_NAMES, FEATURE_DISPLAY, MODEL_CLASSES
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Healthcare | Heart Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #faf0ff 50%, #f0f8ff 100%);
}

/* Main title */
.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1a237e, #7b1fa2, #1565c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.3rem 0;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border: 1px solid rgba(255,255,255,0.8);
    margin-bottom: 1rem;
}

.patient-card {
    background: linear-gradient(135deg, #3b4fd8 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    margin-bottom: 1rem;
}

.patient-card h3 {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: white !important;
}

.risk-high {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}

.risk-low {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}

.risk-label-high {
    font-family: 'Sora', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #dc2626;
}

.risk-label-low {
    font-family: 'Sora', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #16a34a;
}

.prob-value {
    font-family: 'Sora', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #1a237e;
}

.section-header {
    font-family: 'Sora', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.shap-factor {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f1f5f9;
}

.factor-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: #374151;
}

.factor-val-pos {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    color: #dc2626;
    font-size: 0.9rem;
}

.factor-val-neg {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    color: #16a34a;
    font-size: 0.9rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-red { background: #fee2e2; color: #dc2626; }
.badge-green { background: #dcfce7; color: #16a34a; }
.badge-blue { background: #dbeafe; color: #1d4ed8; }

.model-info {
    font-size: 0.85rem;
    color: #6b7280;
    padding: 0.3rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #311b92 50%, #4a148c 100%);
    color: white;
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: white !important;
}

section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.9) !important;
}

.sidebar-section {
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    padding: 0.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #6b7280;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b4fd8, #7c3aed);
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b4fd8 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    box-shadow: 0 4px 15px rgba(59, 79, 216, 0.4);
    transition: all 0.2s;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 79, 216, 0.5);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'X_data' not in st.session_state:
    st.session_state.X_data = None
if 'y_data' not in st.session_state:
    st.session_state.y_data = None
if 'df_full' not in st.session_state:
    st.session_state.df_full = None

# ─── LOAD & TRAIN ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'heart_disease_dataset.csv')

@st.cache_resource(show_spinner=False)
def get_trained_models(data_path):
    X, y, df = load_data(data_path)
    results, X_test, y_test = train_models(X, y)
    return results, X, y, df

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Explainable AI Healthcare Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered heart disease risk prediction with transparent, interpretable explanations</div>', unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Input")
    st.markdown("---")
    
    st.markdown("### 👤 Demographics")
    age = st.slider("Age", 20, 80, 54)
    gender = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
    
    st.markdown("### 🩺 Clinical Measurements")
    blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 200, 150)
    bmi = st.slider("BMI", 15.0, 50.0, 31.0, step=0.1)
    
    st.markdown("### 🧪 Lab Results")
    cholesterol_map = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}
    cholesterol_label = st.selectbox("Cholesterol Level", list(cholesterol_map.keys()), index=1)
    cholesterol = cholesterol_map[cholesterol_label]
    
    glucose_map = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}
    glucose_label = st.selectbox("Glucose Level", list(glucose_map.keys()), index=1)
    glucose = glucose_map[glucose_label]
    
    st.markdown("### 🚬 Lifestyle Factors")
    smoking = 1 if st.checkbox("Smoking", value=True) else 0
    alcohol = 1 if st.checkbox("Alcohol Intake", value=False) else 0
    physical_activity = 1 if st.checkbox("Regular Physical Activity", value=False) else 0
    
    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    selected_model_name = st.selectbox("Algorithm", list(MODEL_CLASSES.keys()), index=0)
    
    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True)

# ─── LOAD MODELS ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models on healthcare dataset..."):
    if not st.session_state.models_trained:
        results, X_data, y_data, df_full = get_trained_models(DATA_PATH)
        st.session_state.model_results = results
        st.session_state.X_data = X_data
        st.session_state.y_data = y_data
        st.session_state.df_full = df_full
        st.session_state.models_trained = True
    else:
        results = st.session_state.model_results
        X_data = st.session_state.X_data
        y_data = st.session_state.y_data
        df_full = st.session_state.df_full

# ─── PATIENT DATA ────────────────────────────────────────────────────────────────
patient_data = [age, 1 if gender == "Male" else 0, blood_pressure,
                cholesterol, bmi, glucose, smoking, alcohol, physical_activity]

# ─── AUTO PREDICT ────────────────────────────────────────────────────────────────
model_info = results[selected_model_name]
model = model_info['model']
scaler = model_info.get('scaler')
accuracy = model_info['accuracy']
auc = model_info['auc']

pred, prob, X_input = predict_risk(model, selected_model_name, patient_data, scaler)
shap_vals = get_shap_values(model, selected_model_name, X_input, X_data)

# ─── MAIN TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Risk Prediction", "📊 Model Analytics", "📋 Dataset Explorer"])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1: RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1.8], gap="large")
    
    # ── LEFT: Patient Summary ──────────────────────────────────────────────────
    with col1:
        # Patient Data Card
        tags = []  #changes

        if smoking:
            tags.append('<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:999px;font-size:0.75rem;">🚬 Smoker</span>')

        if alcohol:
            tags.append('<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:999px;font-size:0.75rem;">🍺 Alcohol</span>')

        if physical_activity:
            tags.append('<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:999px;font-size:0.75rem;">🏃 Active</span>')
        else:
            tags.append('<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:999px;font-size:0.75rem;">🛋️ Sedentary</span>')

        tags_html = "".join(tags)
        st.markdown(f"""
        <div class="patient-card">
            <h3 style="margin:0;">👤 Patient Profile</h3>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">AGE</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.4rem;font-weight:700;">{age}</div>
                </div>
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">GENDER</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.4rem;font-weight:700;">{gender}</div>
                </div>
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">BLOOD PRESSURE</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.4rem;font-weight:700;">{blood_pressure} mmHg</div>
                </div>
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">BMI</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.4rem;font-weight:700;">{bmi}</div>
                </div>
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">GLUCOSE</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.1rem;font-weight:600;">{glucose_label}</div>
                </div>
                <div>
                    <div style="opacity:0.7;font-size:0.75rem;margin-bottom:2px;">CHOLESTEROL</div>
                    <div style="font-family:Sora,sans-serif;font-size:1.1rem;font-weight:600;">{cholesterol_label}</div>
                </div>
            </div>
            
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Result
        risk_pct = int(prob * 100)
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
                <div style="font-size:1.8rem;margin-bottom:0.3rem;">⚠️</div>
                <div class="risk-label-high">High Risk of Heart Disease</div>
                <div class="prob-value">{risk_pct}%</div>
                <div style="color:#6b7280;font-size:0.85rem;">Risk Probability</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <div style="font-size:1.8rem;margin-bottom:0.3rem;">✅</div>
                <div class="risk-label-low">Low Risk of Heart Disease</div>
                <div class="prob-value" style="color:#16a34a;">{risk_pct}%</div>
                <div style="color:#6b7280;font-size:0.85rem;">Risk Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem;">
            <div class="section-header">🤖 Model Information</div>
            <div class="model-info">Model Used: <b>{selected_model_name}</b></div>
            <div class="model-info">Model Accuracy: <b>{accuracy*100:.1f}%</b></div>
            <div class="model-info">ROC-AUC Score: <b>{auc:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP table
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🧬 SHAP Factor Contributions</div>', unsafe_allow_html=True)
        
        shap_pairs = list(zip(FEATURE_NAMES, shap_vals))
        shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        legend = '<div style="display:flex;gap:1rem;margin-bottom:0.8rem;font-size:0.8rem;">'
        legend += '<span>← <span style="color:#16a34a;font-weight:600;">Low Risk</span></span>'
        legend += '<span><span style="color:#dc2626;font-weight:600;">High Risk</span> →</span>'
        legend += '</div>'
        st.markdown(legend, unsafe_allow_html=True)
        
        for feat, val in shap_pairs_sorted:
            bar = '|'
            sign = '+' if val > 0 else ''
            color_class = 'factor-val-pos' if val > 0 else 'factor-val-neg'
            st.markdown(f"""
            <div class="shap-factor">
                <span class="factor-name">{bar} {FEATURE_DISPLAY[feat]}</span>
                <span class="{color_class}">{sign}{val:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── RIGHT: Gauge + SHAP Chart ─────────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-header">🎯 Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
        
        # Gauge Chart
        gauge_color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.4 else "#22c55e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={                                  #cahnges
    'text': "<b>Risk Score</b>",   # 🔥 bold
    'font': {'size': 18, 'family': 'Sora', 'color': '#000000'}  # 🔥 black
},
            number={'suffix': '%', 'font': {'size': 48, 'family': 'Sora', 'color': gauge_color}},
            gauge={
                'axis': {   #changes
    'range': [0, 100],
    'tickwidth': 2,
    'tickcolor': "#000000",     # 🔥 black ticks
    'tickfont': {'color': '#000000', 'size': 12, 'family': 'Arial Black'}  # 🔥 bold effect
},
                'bar': {'color': gauge_color, 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 33], 'color': '#dcfce7'},
                    {'range': [33, 66], 'color': '#fef9c3'},
                    {'range': [66, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "#1e293b", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # SHAP Bar Chart ------------------------------------------------
        st.markdown('<div class="section-header">📊 SHAP Explanation of Contribution</div>', unsafe_allow_html=True)
        
        shap_df = pd.DataFrame({
            'Feature': [FEATURE_DISPLAY[f] for f, v in shap_pairs_sorted],
            'SHAP Value': [v for f, v in shap_pairs_sorted],
            'Direction': ['Increases Risk' if v > 0 else 'Decreases Risk' for f, v in shap_pairs_sorted]
        })
        
        colors = ['#ef4444' if v > 0 else '#3b82f6' for v in shap_df['SHAP Value']]
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=shap_df['Feature'][::-1],
            x=shap_df['SHAP Value'][::-1],
            orientation='h',
            marker_color=colors[::-1],
            marker_line_width=0,
            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in shap_df['SHAP Value'][::-1]],
            textposition='outside',
            textfont={'family': 'Sora', 'size': 13, 'color': '#1e293b'},
        ))
        
        fig_shap.update_layout(
            height=360,
            margin=dict(l=10, r=80, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title=dict(
                text='SHAP Value (Impact on Model Output)',  #cahnges 
                font=dict(family='Inter', size=12)
                ),
                gridcolor='#f1f5f9',
                zerolinecolor='#cbd5e1',
                zerolinewidth=2
            ),
            yaxis=dict(   #changes
    tickfont=dict(
        family='Inter',
        size=13,
        color='#000000'   # 🔥 BLACK
    )
),
            showlegend=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Clinical Interpretation-----------------------
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        st.markdown('<div class="section-header">💡 Clinical Interpretation</div>', unsafe_allow_html=True)

        top_risk_factors = [(FEATURE_DISPLAY[f], v) for f, v in shap_pairs_sorted if v > 0][:3]
        protective_factors = [(FEATURE_DISPLAY[f], v) for f, v in shap_pairs_sorted if v < 0]

        # 🔴 Key Risk Factors
        if top_risk_factors:
            st.markdown(
                '<div style="color:#000000;font-weight:700;font-size:16px;">🔴 Key Risk Factors:</div>',
                unsafe_allow_html=True
            )
            for feat, val in top_risk_factors:
                st.markdown(
                    f'<div style="color:#000000;font-weight:700;">• {feat} is significantly increasing risk (SHAP: +{val:.3f})</div>',
                    unsafe_allow_html=True
                )

        # 🟢 Protective Factors
        if protective_factors:
            st.markdown(
                '<div style="color:#000000;font-weight:700;font-size:16px;margin-top:10px;">🟢 Protective Factors:</div>',
                unsafe_allow_html=True
            )
            for feat, val in protective_factors:
                st.markdown(
                    f'<div style="color:#000000;font-weight:700;">• {feat} is helping reduce risk (SHAP: {val:.3f})</div>',
                    unsafe_allow_html=True
                )

        # ⚠️ Recommendation
        if pred == 1:
            st.markdown(
                '<div style="margin-top:12px;color:#000000;font-weight:700;background:#fef3c7;padding:10px;border-radius:8px;">⚠️ Recommendation: This patient shows high cardiovascular risk. Suggest immediate lifestyle modifications and clinical consultation.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="margin-top:12px;color:#000000;font-weight:700;background:#dcfce7;padding:10px;border-radius:8px;">✅ Recommendation: Risk is currently manageable. Continue monitoring and maintain healthy lifestyle habits.</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════════
with tab2: 
    st.markdown("""          
<h3 style="color:#000000; font-weight:700; font-family:Sora;">
📈 Model Performance Comparison
</h3>
""", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2, gap="large")
    
    with col_a:
        # Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] * 100 for m in model_names]
        aucs = [results[m]['auc'] for m in model_names]
        
        colors_bar = ['#3b4fd8' if m == selected_model_name else '#c7d2fe' for m in model_names]
        
        fig_acc = go.Figure(go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=colors_bar,
            marker_line_width=0,
            text=[f"{a:.1f}%" for a in accuracies],
            textposition='outside',
            textfont={'family': 'Sora', 'size': 13}
        ))
        fig_acc.update_layout(
    title=dict(
        text="<b>Model Accuracy Comparison</b>",   # 🔥 bold title
        font=dict(family='Sora', size=16, color='#000000')  # 🔥 black
    ),
    height=300,

    xaxis=dict(
        tickfont=dict(
            family='Arial Black',   # 🔥 bold effect for model names
            size=12,
            color='#000000'
        )
    ),

    yaxis=dict(
        range=[0, 110],
        title=dict(
            text="<b>Accuracy (%)</b>",   # 🔥 bold axis title
            font=dict(size=13, color='#000000')
        ),
        tickfont=dict(
            family='Arial Black',   # 🔥 bold effect for values (0–100)
            size=12,
            color='#000000'
        ),
        gridcolor='#e5e7eb'
    ),

    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=50, b=10),

    font=dict(color='#000000'),  # 🔥 global fallback
    showlegend=False
)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col_b:
        # AUC comparison
        fig_auc = go.Figure(go.Bar(
            x=model_names,
            y=aucs,
            marker_color=['#7c3aed' if m == selected_model_name else '#ddd6fe' for m in model_names],
            marker_line_width=0,
            text=[f"{a:.3f}" for a in aucs],
            textposition='outside',
            textfont={'family': 'Sora', 'size': 13}
        ))
        fig_auc.update_layout(
            title={'text': 'ROC-AUC Score Comparison', 'font': {'family': 'Sora', 'size': 15}},
            height=300,
            yaxis=dict(range=[0, 1.15], title='AUC Score', gridcolor='#f1f5f9'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_auc, use_container_width=True)
    
    # Feature Importance (Global)
    st.markdown("""
<h3 style="color:#000000; font-weight:700; font-family:Sora;">
🌍 Global Feature Importance
</h3>
""", unsafe_allow_html=True)
    
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': [FEATURE_DISPLAY[f] for f in FEATURE_NAMES], 'Importance': fi})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        
        fig_fi = go.Figure(go.Bar(
            y=fi_df['Feature'],
            x=fi_df['Importance'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0, '#c7d2fe'], [0.5, '#818cf8'], [1, '#3b4fd8']],
                line_width=0
            ),
            text=[f"{v:.3f}" for v in fi_df['Importance']],
textposition='outside',
textfont=dict(
    family='Arial Black',
    size=12,
    color='#000000'   # 🔥 makes values black
)
        ))
        fig_fi.update_layout(                #chnages 
    height=350,
    xaxis=dict(
        title=dict(
            text='Feature Importance Score',
            font=dict(color='black', size=14, family='Arial Black')
        ),
        gridcolor='#f1f5f9',
        tickfont=dict(color='black', size=12, family='Arial Black')
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=13, family='Arial Black')
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=60, t=10, b=40)
)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance visualization is available for tree-based models (XGBoost, Random Forest, Gradient Boosting).")
    
    # Model metrics table-------------------------------------------------
    st.markdown("""
<h3 style="color:#000000; font-weight:700; font-family:Sora;">
📋 Detailed Model Metrics
</h3>
""", unsafe_allow_html=True)
    metrics_data = {
        'Model': model_names,
        'Accuracy': [f"{results[m]['accuracy']*100:.2f}%" for m in model_names],
        'ROC-AUC': [f"{results[m]['auc']:.4f}" for m in model_names],
        'Status': ['✅ Selected' if m == selected_model_name else '—' for m in model_names]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
<h3 style="color:#000000; font-weight:700; font-family:Sora;">
🗃️ Dataset Overview
</h3>
""", unsafe_allow_html=True)
    
    st.markdown("""     #chanegs
<style>
[data-testid="stMetricLabel"] {
    color: black !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: black !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)
    
    df = df_full.copy()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("Features", len(FEATURE_NAMES))
    col3.metric("High Risk Cases", int(df['heart_disease'].sum()))
    col4.metric("Low Risk Cases", int((df['heart_disease'] == 0).sum()))

    #-----------------------------------------------------------------
    #Distrubiution analysis
    #-----------------------------------------------------------------
    
    st.markdown("""
<h3 style="color:#000000; font-weight:700; font-family:Sora;">
📊 Distribution Analysis
</h3>
""", unsafe_allow_html=True)
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        # Age distribution by risk-------------------------------
        fig_age = px.histogram(     #changes all section
            df, x='age', color='heart_disease',
            nbins=20,
            color_discrete_map={0: '#22c55e', 1: '#ef4444'},
            labels={'heart_disease': 'Heart Disease', 'age': 'Age'},
            title='Age Distribution by Risk'
            )

        fig_age.update_layout(
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title='Risk', orientation='h', y=1.15),
            margin=dict(l=10, r=10, t=50, b=10),

            # 🔥 TITLE
            title_font=dict(color='black', size=16, family='Arial Black'),

            # 🔥 X-AXIS
            xaxis=dict(
                tickfont=dict(color='black', family='Arial Black'),
                title_font=dict(color='black', family='Arial Black')
            ),

            # 🔥 Y-AXIS
            yaxis=dict(
                tickfont=dict(color='black', family='Arial Black'),
                title_font=dict(color='black', family='Arial Black')
            )
        )

        fig_age.update_traces(marker_line_width=0)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col_y:
        # BP vs BMI scatter ----------------------------------------------------
        fig_bp_bmi = px.scatter(
            df,
            x='bmi',
            y='blood_pressure',
            color='heart_disease',
            color_discrete_map={0: '#22c55e', 1: '#ef4444'},
            labels={
            'bmi': 'BMI',
            'blood_pressure': 'Blood Pressure',
            'heart_disease': 'Heart Disease'
            },
            title='Blood Pressure vs BMI'
            )

        fig_bp_bmi.update_layout(
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title='Risk', orientation='h', y=1.15),
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar=dict(   
            title=dict(font=dict(color='black')),   #changes
            
        tickfont=dict(color='black')
    ),

            # 🔥 Title
            title=dict(
                text='Blood Pressure vs BMI',
                font=dict(color='black', size=16, family='Arial Black')
            ),

            # 🔥 X-axis
            xaxis=dict(
                title=dict(
                    text='BMI',
                    font=dict(color='black', family='Arial Black')
                ),
                tickfont=dict(color='black', family='Arial Black')
            ),

            # 🔥 Y-axis
            yaxis=dict(
                title=dict(
                    text='Blood Pressure',
                    font=dict(color='black', family='Arial Black')
                ),
                tickfont=dict(color='black', family='Arial Black')
            )
        )

        st.plotly_chart(fig_bp_bmi, use_container_width=True)
    
    # Correlation heatmap------------------------------------------------
    st.markdown(
    "<h3 style='color:black; font-weight:bold;'>🔥 Feature Correlation Matrix</h3>",
    unsafe_allow_html=True
    )
    corr = df[FEATURE_NAMES + ['heart_disease']].corr()
    display_labels = [FEATURE_DISPLAY.get(c, c) for c in corr.columns]
    
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=display_labels,
        y=display_labels,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
         colorbar=dict(
        tickfont=dict(color='black')
    )
    ))
    fig_corr.update_layout(
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
        tickfont=dict(size=10, color='black')
        ),
        yaxis=dict(
        tickfont=dict(size=10, color='black')
        )
        )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Raw data
    st.markdown(
    "<h3 style='color:black; font-weight:bold;'>📄 Sample Data</h3>",
    unsafe_allow_html=True
)
    display_df = df.head(20).copy()
    display_df['gender'] = display_df['gender'].map({0: 'Female', 1: 'Male'})
    display_df['heart_disease'] = display_df['heart_disease'].map({0: '✅ Low Risk', 1: '⚠️ High Risk'})
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Dataset (CSV)",
        data=csv,
        file_name="heart_disease_dataset.csv",
        mime="text/csv"
    )
