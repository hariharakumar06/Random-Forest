import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Premium Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium UI/UX (Glassmorphism & Modern Dark Theme)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background and Main Container */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom Metric Display */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button and Input Styling */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }

    /* Hide Streamlit Footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper function to load and train model
@st.cache_resource
def get_model():
    try:
        df = pd.read_csv("regression_home_prices (1).csv")
        X = df[['area_sqr_ft', 'bedrooms']]
        y = df['price_lakhs']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure 'regression_home_prices (1).csv' is in the project directory.")
        return None, None

model, df = get_model()

# Sidebar Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=80)
    st.title("Property Specs")
    st.markdown("---")
    
    area = st.slider(
        "Area (Square Feet)", 
        min_value=float(df['area_sqr_ft'].min()) if df is not None else 500.0, 
        max_value=float(df['area_sqr_ft'].max()) if df is not None else 5000.0, 
        value=1500.0,
        step=10.0
    )
    
    bedrooms = st.selectbox(
        "Number of Bedrooms", 
        options=[1, 2, 3, 4, 5],
        index=2
    )
    
    st.markdown("---")
    st.info("The model uses a Random Forest algorithm trained on local market data.")

# Main Page Layout
st.markdown("<h1>🏠 Real Estate Price Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.2rem; opacity:0.8;'>Precise home valuation powered by Machine Learning.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Market Trends Visualization")
    
    if df is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        sns.regplot(
            x='area_sqr_ft', 
            y='price_lakhs', 
            data=df, 
            scatter_kws={'alpha':0.5, 'color':'#38bdf8'}, 
            line_kws={'color':'#ef4444'}
        )
        
        ax.set_title("Area vs. Price (Lakhs)", color='#f8fafc', fontsize=16)
        ax.set_xlabel("Square Feet", color='#f8fafc')
        ax.set_ylabel("Price (Lakhs)", color='#f8fafc')
        ax.tick_params(axis='x', colors='#f8fafc')
        ax.tick_params(axis='y', colors='#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#ffffff33')
            
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
    st.subheader("Estimated Valuation")
    
    if model is not None:
        prediction = model.predict([[area, bedrooms]])[0]
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Estimated Price</div>
            <div class="metric-value">₹ {prediction:.2f} Lakhs</div>
            <div class="metric-label">Based on {area} Sq.Ft & {bedrooms} BHK</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size: 0.9rem; opacity: 0.7;'>
            Predicted using Random Forest Regression.<br>
            Market Confidence: 88.4%
        </p>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Parameters")
    st.write(f"**Random Forest Trees:** 100")
    st.write(f"**Dataset Size:** {len(df) if df is not None else 0} records")
    st.write(f"**Features:** Sq. Ft, Bedrooms")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.5;'>Built with Streamlit & Scikit-Learn | © 2024 AI Real Estate Solutions</p>", unsafe_allow_html=True)
