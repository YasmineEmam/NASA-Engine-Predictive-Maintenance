import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="PRO-LIFT | Mission Control",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
        background: radial-gradient(circle at center, #0a192f 0%, #020c1b 100%) !important;
        background-attachment: fixed !important;
        color: #e6f1ff;
    }
    header { background-color: transparent !important; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(0, 212, 255, 0.1) !important; }
    [data-testid="stMetric"] {
        background: rgba(16, 33, 65, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.1) !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Share Tech Mono', monospace;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #0a192f !important;
    }
    .unified-title {
        text-align: center;
        background: linear-gradient(to right, #00d4ff, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin-top: -60px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        reg_model = joblib.load("reg_model.pkl")
        cls_model = joblib.load("cls_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        features_list = joblib.load("features_list.pkl")
        return reg_model, cls_model, scaler, le, features_list
    except:
        return None, None, None, None, None

reg_model, cls_model, scaler, le, features_list = load_assets()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/144/rocket.png", width=100)
    st.markdown("<h2 style='color:#00d4ff; text-align:center;'>MISSION CONTROL</h2>", unsafe_allow_html=True)
    st.divider()
    uploaded_file = st.file_uploader("📡 Sync Telemetry Stream", type=["csv"])
    if uploaded_file:
        st.success("🛰️ Link Established")

st.markdown("<h1 class='unified-title'>PRO-LIFT ANALYTICS</h1>", unsafe_allow_html=True)

if uploaded_file and reg_model:
    data = pd.read_csv(uploaded_file)
    id_col = data.columns[0]
    
    X_scaled = scaler.transform(data[features_list])
    data["Predicted_RUL"] = np.maximum(0, reg_model.predict(X_scaled)).astype(int)
    data["Status"] = le.inverse_transform(cls_model.predict(X_scaled))
    fleet_latest = data.groupby(id_col).tail(1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ACTIVE BOOSTERS", len(fleet_latest))
    m2.metric("MEAN MISSION RUL", f"{fleet_latest['Predicted_RUL'].mean():.0f} T-minus")
    m3.metric("CRITICAL ALERTS", len(fleet_latest[fleet_latest["Status"] == "Critical"]))
    m4.metric("SUCCESS INDEX", f"{(fleet_latest['Status'] == 'Healthy').sum()/len(fleet_latest)*100:.1f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🚀 FLEET TRAJECTORY", "🔭 BOOSTER TWIN", "💻 SYSTEM DIAGNOSTICS"])

    with tab1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            fig_pie = px.pie(fleet_latest, names="Status", hole=0.7, 
                             color="Status", color_discrete_map={"Healthy":"#00d4ff", "Warning":"#ffcc00", "Critical":"#ff4b4b"})
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#e6f1ff", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_hist = px.histogram(fleet_latest, x="Predicted_RUL", nbins=15, color_discrete_sequence=["#00d4ff"])
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e6f1ff")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        selected_id = st.selectbox("Select Booster ID", sorted(data[id_col].unique()))
        unit_data = data[data[id_col] == selected_id]
        curr_val = unit_data["Predicted_RUL"].iloc[-1]
        
        d1, d2 = st.columns([1, 2])
        with d1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=curr_val,
                gauge={'axis': {'range': [0, 250], 'tickcolor': "#00d4ff"}, 
                       'bar': {'color': "#00d4ff"},
                       'steps': [{'range': [0, 50], 'color': "rgba(255, 75, 75, 0.3)"}]}
            ))
            fig_gauge.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font_color="#00d4ff")
            st.plotly_chart(fig_gauge, use_container_width=True)
        with d2:
            fig_line = px.line(unit_data, x=unit_data.index, y="Predicted_RUL", title="Propulsion Degradation Curve")
            fig_line.update_traces(line_color="#00d4ff", line_width=4)
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10, 25, 47, 0.4)', font_color="#e6f1ff")
            st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        importances = reg_model.feature_importances_
        feat_df = pd.DataFrame({"Module": features_list, "Sensors": importances}).sort_values("Sensors")
        fig_bar = px.bar(feat_df, x="Sensors", y="Module", orientation='h', color="Sensors", color_continuous_scale="Viridis")
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e6f1ff")
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <img src='https://img.icons8.com/color/200/rocket.png' width='180' style='filter: drop-shadow(0 0 15px #00d4ff);'>
        <h2 style='color: #00d4ff; margin-top:20px;'>READY FOR MISSION SYNC</h2>
        <h3 style='color: #ffffff; opacity: 0.8;'>By PLAN B Team</h3>        
        <p style='color: #8892b0;'>Establish telemetry link to begin analysis...</p>
    </div>
    """, unsafe_allow_html=True)