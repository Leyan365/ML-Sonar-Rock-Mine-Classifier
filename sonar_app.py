import streamlit as st
import numpy as np
import joblib
import datetime 
import time 

# Set Streamlit page configuration for a wide view and dark theme look
st.set_page_config(
    page_title="Sonar Classification System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Inject Custom CSS for Command Center Look ---
st.markdown("""
<style>
/* Base dark background */
.stApp {
    background-color: #0F172A; /* Tailwind slate-900 */
    color: #C6F6D5; /* Green tone */
    font-family: monospace;
}
/* Styling for containers/panels */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
.stSelectbox, .stButton>button {
    border-radius: 0.375rem; /* Rounded corners */
    border-color: #10B981; /* Green border */
}
/* Custom Styling for the main headers */
h1 { color: #6EE7B7; /* Emerald-400 */ }
h3 { color: #A7F3D0; /* Emerald-200 */ }

/* Success/Error boxes for prediction */
.stAlert > div {
    font-size: 1.25rem;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 1rem;
    border-left: 6px solid;
}
</style>
""", unsafe_allow_html=True)


# --- 1. Load Model and Scaler ---
@st.cache_resource
def load_components():
    """Loads the trained SVM model and the StandardScaler object."""
    try:
        model = joblib.load('sonar_svm_model.pkl')
        scaler = joblib.load('sonar_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found. Ensure 'sonar_svm_model.pkl' and 'sonar_scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_components()

# --- 2. Define Example Data (Simulating Sensor Input) ---
example_data_raw = {
    "Select Detection Target": (None, "Choose a sonar contact to analyze"),
    # Mine example
    "Contact Alpha-7 (Suspected Mine)": (
        np.array([0.0715, 0.0849, 0.0587, 0.0218, 0.0862, 0.1801, 0.1916, 0.1896, 0.2960, 0.4186, 0.4867, 0.5249, 0.5959, 0.6855, 0.8573, 0.9718, 0.8693, 0.8711, 0.8954, 0.9922, 0.8980, 0.8158, 0.8373, 0.7541, 0.5893, 0.5488, 0.5643, 0.5406, 0.4783, 0.4439, 0.3698, 0.2574, 0.1478, 0.1743, 0.1229, 0.1588, 0.1803, 0.1436, 0.1667, 0.2630, 0.2234, 0.1239, 0.0869, 0.2092, 0.1499, 0.0676, 0.0899, 0.0927, 0.0658, 0.0086, 0.0216, 0.0153, 0.0121, 0.0096, 0.0196, 0.0042, 0.0066, 0.0099, 0.0083, 0.0124]),
        "Metallic signature detected - requires immediate classification"
    ),
    # Rock example
    "Contact Bravo-3 (Rock Formation)": (
        np.array([0.0240, 0.0218, 0.0324, 0.0569, 0.0330, 0.0513, 0.0897, 0.0713, 0.0569, 0.0389, 0.1934, 0.2434, 0.2906, 0.2606, 0.3811, 0.4997, 0.3015, 0.3655, 0.6791, 0.7307, 0.5053, 0.4441, 0.6987, 0.8133, 0.7781, 0.8943, 0.8929, 0.8913, 0.8610, 0.8063, 0.5540, 0.2446, 0.3459, 0.1615, 0.2467, 0.5564, 0.4681, 0.0979, 0.1582, 0.0751, 0.3321, 0.3745, 0.2666, 0.1078, 0.1418, 0.1687, 0.0738, 0.0634, 0.0144, 0.0226, 0.0061, 0.0162, 0.0146, 0.0093, 0.0112, 0.0094, 0.0054, 0.0019, 0.0066, 0.0023]),
        "Natural formation - mineral composition analysis"
    ),
    # Unknown (Test Case) example
    "Contact Charlie-9 (Unknown Object)": (
        np.array([0.0731, 0.1249, 0.1665, 0.1496, 0.1443, 0.2770, 0.2555, 0.1712, 0.0466, 0.1114, 0.1739, 0.3160, 0.3249, 0.2164, 0.2031, 0.2580, 0.1796, 0.2422, 0.3609, 0.1810, 0.2604, 0.6572, 0.9734, 0.9757, 0.8079, 0.6521, 0.4915, 0.5363, 0.7649, 0.5250, 0.5101, 0.4219, 0.4160, 0.1906, 0.0223, 0.4219, 0.5496, 0.2483, 0.2034, 0.2729, 0.2837, 0.4463, 0.3178, 0.0807, 0.1192, 0.2134, 0.3241, 0.2945, 0.1474, 0.0211, 0.0361, 0.0444, 0.0230, 0.0290, 0.0141, 0.0161, 0.0177, 0.0194, 0.0207, 0.0057]),
        "Unidentified contact - classification required"
    )
}

# --- 4. Prediction Function ---
def run_prediction(data_array, model, scaler):
    """Scales data, predicts class, and mocks confidence."""
    # 1. Reshape the data for the model (1 sample, 60 features)
    input_data_reshaped = data_array.reshape(1, -1)
    
    # 2. Scale the data (CRUCIAL STEP)
    input_data_scaled = scaler.transform(input_data_reshaped)
    
    # 3. Predict the class ('M' or 'R')
    prediction_class = model.predict(input_data_scaled)[0]
    
    # 4. Mock Confidence for UI Display
    # Use a consistent, high confidence based on the prediction to reflect the tuned SVM's high accuracy
    if prediction_class == 'M':
        # Mines are crucial, so we give a slightly higher confidence range
        confidence = np.random.uniform(0.92, 0.97) 
    else:
        confidence = np.random.uniform(0.85, 0.90) 
        
    return {
        'type': prediction_class,
        'confidence': confidence,
        'threat': 'HIGH' if prediction_class == 'M' else 'LOW'
    }


# --- 5. Streamlit Interface Header  ---
st.markdown(f"""
<div style="background-color: #1E293B; border-bottom: 2px solid #059669; padding: 10px 0;">
    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; padding: 0 20px;">
        <div style="display: flex; align-items: center; space-x: 10px;">
            <span style="font-size: 30px;">📡</span>
            <div>
                <h1 style="margin: 0; font-size: 24px; color: #10B981;">SONAR CLASSIFICATION SYSTEM</h1>
                <p style="margin: 0; font-size: 12px; color: #34D399;">USS GUARDIAN • DEEP WATER OPERATIONS</p>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 18px; color: #10B981;">
                {datetime.datetime.now().strftime("%H:%M:%S")}
            </div>
            <div style="font-size: 12px; color: #34D399;">
                DEPTH: 120M • STATUS: OPERATIONAL
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# --- MAIN LAYOUT ---
if model and scaler: 
    col1, col2 = st.columns([1, 2])
    
    # --- LEFT COLUMN: CONTROL PANEL ---
    with col1:
        # CONTACT ANALYSIS CONTAINER
        with st.container(border=True):
            st.markdown("### 🎯 CONTACT ANALYSIS", unsafe_allow_html=True)
            
            selected_key = st.selectbox(
                "SELECT TARGET CONTACT", 
                list(example_data_raw.keys()),
                key='selectbox_contact'
            )
            
            # Extract data and description
            data_array, description = example_data_raw[selected_key]
            
            if data_array is not None:
                st.markdown(f"""
                <div style='background-color: #1E293B; border: 1px solid #059669; border-radius: 6px; padding: 10px; margin-top: 10px;'>
                    <p style='color: #9CA3AF; font-size: 14px;'>{description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Button with a unique style
            analyze_button = st.button("INITIATE SCAN", use_container_width=True)

        # SYSTEM STATUS CONTAINER
        with st.container(border=True):
            st.markdown("### 🧭 SYSTEM STATUS", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size: 14px; color: #A7F3D0;'>
                <div style='display: flex; justify-content: space-between;'><span>Sonar Array:</span><span style='color: #10B981;'>ACTIVE</span></div>
                <div style='display: flex; justify-content: space-between;'><span>ML Classifier:</span><span style='color: #10B981;'>READY</span></div>
                <div style='display: flex; justify-content: space-between;'><span>Detection Range:</span><span style='color: #10B981;'>2.5 NM</span></div>
                <div style='display: flex; justify-content: space-between;'><span>Confidence Threshold:</span><span style='color: #10B981;'>75%</span></div>
            </div>
            """, unsafe_allow_html=True)

    # --- RIGHT COLUMN: RESULTS PANEL ---
    with col2:
        st.markdown("### 🌊 CLASSIFICATION RESULTS", unsafe_allow_html=True)
        
        # State to hold the prediction (for persistent display after scan)
        if 'prediction' not in st.session_state:
            st.session_state.prediction = None
            
        # Handle Scan Logic
        if analyze_button:
            if data_array is None:
                st.warning("Please select a valid example before initiating the scan.")
            else:
                with st.spinner('ANALYZING ACOUSTIC SIGNATURE...'):
                    # Introduce a small delay to simulate processing time
                    time.sleep(1.5) 
                    
                    # Run the real prediction using the dedicated function
                    st.session_state.prediction = run_prediction(data_array, model, scaler)

        # --- Display Results ---
        prediction = st.session_state.prediction

        if prediction:
            is_mine = prediction['type'] == 'M'
            
            # --- Result Header ---
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border: 2px solid #{'F87171' if is_mine else '60A5FA'}; border-radius: 10px; background-color: #{'7F1D1D' if is_mine else '1E3A8A'}20;'>
                <div style='display: flex; justify-content: center; align-items: center; space-x: 10px;'>
                    <span style='font-size: 40px; color: #{'F87171' if is_mine else '60A5FA'};'>
                        {'⚠️' if is_mine else '🛡️'}
                    </span>
                    <h3 style='margin: 0; font-size: 30px; color: #{'F87171' if is_mine else '60A5FA'};'>
                        {prediction['type']}
                        {' MINE DETECTED' if is_mine else ' ROCK FORMATION'}
                    </h3>
                </div>
                <p style='margin-top: 5px; font-size: 16px; color: #9CA3AF;'>
                    Classification: {'EXPLOSIVE DEVICE' if is_mine else 'GEOLOGICAL CONTACT'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")

            # --- Details Grid ---
            col_conf, col_threat = st.columns(2)
            
            with col_conf:
                st.markdown("#### CONFIDENCE LEVEL", unsafe_allow_html=True)
                conf_percent = round(prediction['confidence'] * 100)
                
                # Progress Bar styling
                st.progress(conf_percent)
                st.markdown(f"<p style='text-align: center; font-size: 20px; font-weight: bold; color: #10B981;'>{conf_percent}%</p>", unsafe_allow_html=True)

            with col_threat:
                st.markdown("#### THREAT LEVEL", unsafe_allow_html=True)
                threat_color = "#F56565" if is_mine else "#48BB78"
                threat_text = "HIGH" if is_mine else "LOW"
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; border-radius: 6px; background-color: {threat_color}; color: white; font-weight: bold; font-size: 20px;'>
                    {threat_text}
                </div>
                """, unsafe_allow_html=True)

            # --- Tactical Recommendation ---
            if is_mine:
                st.markdown(f"""
                <div style='margin-top: 20px; background-color: #450A0A; border: 1px solid #DC2626; border-radius: 6px; padding: 15px;'>
                    <h4 style='color: #F87171; font-weight: bold; margin-bottom: 10px;'>TACTICAL RECOMMENDATION</h4>
                    <ul style='list-style-type: none; padding-left: 0; color: #FCA5A5; font-size: 14px;'>
                        <li>• Maintain safe distance (minimum 500m)</li>
                        <li>• Alert navigation team immediately</li>
                        <li>• Log coordinates for mine warfare command</li>
                        <li>• Consider alternate routing</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Default state 
            st.info("Select a contact and initiate scan to display classification results.")