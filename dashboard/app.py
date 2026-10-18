import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="HYBRID DPI + ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #e6e6fa;
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert {
        border-radius: 10px;
        background-color: #1a1f36;
        border: 1px solid #4facfe;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Hybrid DPI + Machine Learning Traffic Dashboard")
st.write("Real-time network traffic analysis combining Deep Packet Inspection (DPI) and Machine Learning classification.")

# Resolve default paths relative to script location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
default_csv = os.path.join(BASE_DIR, "data", "flow_features.csv") if os.path.exists(os.path.join(BASE_DIR, "data", "flow_features.csv")) else "data/flow_features.csv"
default_model = os.path.join(BASE_DIR, "ml", "models", "traffic_classifier.joblib") if os.path.exists(os.path.join(BASE_DIR, "ml", "models", "traffic_classifier.joblib")) else "ml/models/traffic_classifier.joblib"

# Sidebar Controls
st.sidebar.title("Configuration")
csv_path = st.sidebar.text_input("Features CSV Path", default_csv)
model_path = st.sidebar.text_input("Model Path", default_model)

@st.cache_resource
def load_ml_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback to base dir resolution
    alt = os.path.join(BASE_DIR, path)
    if os.path.exists(alt):
        return joblib.load(alt)
    return None

model = load_ml_model(model_path)

# Load Data
if not os.path.exists(csv_path) and not os.path.exists(os.path.join(BASE_DIR, csv_path)):
    st.info(f"Waiting for flow features CSV at `{csv_path}`. Showing synthetic display in the meantime.")
    # Generate dummy data for illustration
    np.random.seed(42)
    dummy_size = 100
    df = pd.DataFrame({
        'flow_duration_ms': np.random.exponential(10000, dummy_size),
        'total_packets': np.random.randint(10, 500, dummy_size),
        'total_bytes': np.random.randint(1000, 500000, dummy_size),
        'packets_per_second': np.random.uniform(1, 100, dummy_size),
        'bytes_per_second': np.random.uniform(100, 50000, dummy_size),
        'dpi_app_type': np.random.choice(['YOUTUBE', 'FACEBOOK', 'ZOOM', 'UNKNOWN'], dummy_size, p=[0.2, 0.2, 0.1, 0.5]),
        'detected_sni': np.random.choice(['youtube.com', 'facebook.com', 'zoom.us', ''], dummy_size, p=[0.2, 0.2, 0.1, 0.5])
    })
    # Add dummy predictions
    df['ml_prediction'] = np.random.choice(['youtube', 'facebook', 'zoom', 'web_browsing', 'file_download'], dummy_size)
    df['confidence'] = np.random.uniform(0.6, 0.99, dummy_size)
else:
    df = pd.read_csv(csv_path)
    if 'ml_prediction' not in df.columns and model is not None:
        # Load feature columns list
        cols_path = "ml/models/feature_columns.joblib"
        if os.path.exists(cols_path):
            feature_cols = joblib.load(cols_path)
            # Fill missing with 0
            df_feats = df[feature_cols].fillna(0.0)
            df['ml_prediction'] = model.predict(df_feats)
            probs = model.predict_proba(df_feats)
            df['confidence'] = [max(p) for p in probs]
        else:
            df['ml_prediction'] = 'unknown'
            df['confidence'] = 0.0
    elif 'ml_prediction' not in df.columns:
        df['ml_prediction'] = 'unknown'
        df['confidence'] = 0.0

# 1. Key Metrics Section
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flows", len(df))
col2.metric("Total Packets", int(df['total_packets'].sum()))
col3.metric("Total Bytes", f"{df['total_bytes'].sum() / (1024*1024):.2f} MB")
avg_duration = df['flow_duration_ms'].mean() / 1000.0 if 'flow_duration_ms' in df.columns else 0
col4.metric("Avg Flow Duration", f"{avg_duration:.2f} s")

# 2. Charts Section
st.write("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Traffic Class Distribution")
    # Combine DPI and ML results for final visualization
    final_labels = []
    for idx, row in df.iterrows():
        dpi = row.get('dpi_app_type', 'UNKNOWN')
        ml = row.get('ml_prediction', 'unknown').upper()
        
        if dpi != 'UNKNOWN' and dpi != 'HTTPS' and dpi != 'HTTP':
            final_labels.append(dpi)
        elif ml != 'UNKNOWN' and ml != 'none':
            final_labels.append(ml)
        else:
            final_labels.append('UNKNOWN')
            
    df['Final_Class'] = final_labels
    class_counts = df['Final_Class'].value_counts().reset_index()
    class_counts.columns = ['Application', 'Count']
    fig = px.pie(class_counts, values='Count', names='Application', 
                 color_discrete_sequence=px.colors.sequential.Electric,
                 hole=0.4)
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("DPI vs ML Source Classification")
    dpi_count = sum(df['dpi_app_type'].apply(lambda x: x not in ['UNKNOWN', 'HTTPS', 'HTTP']))
    ml_count = len(df) - dpi_count
    
    source_df = pd.DataFrame({
        'Source': ['Traditional DPI', 'Machine Learning'],
        'Flow Count': [dpi_count, ml_count]
    })
    fig_bar = px.bar(source_df, x='Source', y='Flow Count', 
                     color='Source', color_discrete_sequence=['#4facfe', '#00f2fe'])
    fig_bar.update_layout(template='plotly_dark')
    st.plotly_chart(fig_bar, use_container_width=True)

# 3. Graphs Section
st.write("---")
g1, g2 = st.columns(2)

with g1:
    st.subheader("Flow Throughput Rate (Bytes/sec)")
    # Plot distribution of Bytes/sec
    if 'bytes_per_second' in df.columns:
        fig_hist = px.histogram(df, x='bytes_per_second', color='Final_Class',
                                log_y=True, title="Throughput Distribution (Log Scale)")
        fig_hist.update_layout(template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)

with g2:
    st.subheader("Model Confidence Histogram")
    if 'confidence' in df.columns:
        fig_conf = px.histogram(df, x='confidence', nbins=20, 
                                color_discrete_sequence=['#00f2fe'])
        fig_conf.update_layout(template='plotly_dark')
        st.plotly_chart(fig_conf, use_container_width=True)

# 4. Top Flows & Features
st.write("---")
f1, f2 = st.columns([2, 1])

with f1:
    st.subheader("Top Active Flows")
    display_cols = ['total_packets', 'total_bytes', 'dpi_app_type', 'ml_prediction', 'confidence']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols].sort_values(by='total_bytes', ascending=False).head(10))

with f2:
    st.subheader("SHAP Feature Importance")
    # If model has feature importances, plot them
    if model is not None and hasattr(model, 'feature_importances_'):
        cols_path = "ml/models/feature_columns.joblib"
        if os.path.exists(cols_path):
            feature_cols = joblib.load(cols_path)
            feat_imp = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(feat_imp, y='Feature', x='Importance', orientation='h',
                             color_discrete_sequence=['#4facfe'])
            fig_imp.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance will load once the model is trained.")
