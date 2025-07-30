import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and data
model = joblib.load("model_DecisionTree.pkl")
historical_df = pd.read_csv("Clean_Dataset.csv")
historical_df.columns = historical_df.columns.str.strip()

# Model columns
model_columns = [
    'stops', 'days_left', 'duration_mins', 'red_eye', 'is_peak_departure', 'cross_region',
    'days_duration_interaction', 'stops_per_hour',
    'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
    'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
    'departure_time_Afternoon', 'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night', 'departure_time_Morning', 'departure_time_Night',
    'arrival_time_Afternoon', 'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night',
    'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai',
    'class_Business', 'class_Economy',
    'airline_tier_High-Cost', 'airline_tier_Low-cost',
    'booking_type_Advance', 'booking_type_Last_Minute', 'booking_type_Near',
    'duration_category_Long', 'duration_category_Medium'
]

# Helper functions
def categorize_booking_type(days_left):
    if days_left <= 3:
        return 'Last_Minute'
    elif days_left <= 20:
        return 'Near'
    else:
        return 'Advance'

def determine_airline_tier(airline):
    return 'High-Cost' if airline in ['Vistara', 'Air_India'] else 'Low-cost'

def is_red_eye(departure_time, arrival_time):
    return int(departure_time in ['Late_Night', 'Night'] and arrival_time in ['Early_Morning', 'Morning'])

def categorize_duration(duration_mins):
    return 'Medium' if duration_mins < 180 else 'Long'

def get_price_tips(features):
    tips = []
    if features.get('class_Business', 0) == 1:
        tips.append("✈️ Consider flying Economy class to save costs.")
    if features.get('booking_type_Last_Minute', 0) == 1 or features.get('booking_type_Near', 0) == 1:
        tips.append("📅 Booking earlier can help get better prices.")
    if features.get('airline_tier_High-Cost', 0) == 1:
        tips.append("💰 Choosing low-cost airlines can save you money.")
    if not tips:
        tips.append("🎉 Your flight details look optimized for the best price!")
    return tips

# Region mapping
region_map = {
    'Delhi': 'North', 'Mumbai': 'West', 'Bangalore': 'South',
    'Kolkata': 'East', 'Hyderabad': 'South', 'Chennai': 'South'
}

# Data options
airlines = ['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara']
cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
departure_times = ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']
arrival_times = ['Afternoon', 'Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']
classes = ['Economy', 'Business']

# Custom CSS
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background-image: url('https://cdn.dribbble.com/userupload/20522838/file/original-2783b0b7999528949d3a7e66870bfc72.gif');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(240, 248, 255, 0.95), rgba(230, 245, 255, 0.97));
        z-index: -1;
        backdrop-filter: blur(3px);
    }

    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .app-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 2px solid rgba(0, 102, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }

    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0066FF, #4285F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -2px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .app-subtitle {
        font-size: 1.3rem;
        color: #1A1A1A;
        font-weight: 600;
        margin-top: 1rem;
        opacity: 0.8;
    }

    .form-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(0, 102, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }

    .form-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        border-color: rgba(0, 102, 255, 0.15);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1A1A1A;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0066FF;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: white !important;
        border: 2px solid #D1D5DB !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        font-weight: 600 !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #0066FF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(0, 102, 255, 0.15) !important;
    }

    .stSelectbox label,
    .stNumberInput label,
    .stSlider label {
        font-weight: 600 !important;
        color: #1A1A1A !important;
        font-size: 1rem !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #0066FF !important;
        box-shadow: 0 0 0 4px rgba(0, 102, 255, 0.1) !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #0066FF, #4285F4) !important;
    }

    .info-box {
        background: linear-gradient(135deg, #E8F4FD, #D6EFFF) !important;
        border: 2px solid #64B5F6 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin: 1.5rem 0 !important;
        font-weight: 600 !important;
        color: #0D47A1 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.1) !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0066FF, #4285F4) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.2rem 2.5rem !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.3) !important;
        width: 100% !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 32px rgba(0, 102, 255, 0.4) !important;
        background: linear-gradient(135deg, #0052CC, #0066FF) !important;
    }

    .time-reference {
        background: linear-gradient(135deg, #FFF9C4, #FFF59D) !important;
        border: 2px solid #FFD54F !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        font-size: 0.95rem !important;
        color: #E65100 !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(230, 81, 0, 0.1) !important;
        line-height: 1.6 !important;
    }

    .results-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(0, 200, 81, 0.2);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        animation: fadeInUp 0.6s ease-out;
    }

    .price-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #00C851, #00E676);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 200, 81, 0.3);
    }

    .price-label {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .price-amount {
        color: white;
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .tips-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1A1A1A;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FF6B35;
    }

    .tip-item {
        background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        border-left: 4px solid #FF6B35;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        font-weight: 600;
        color: #1A1A1A;
        transition: all 0.3s ease;
        font-size: 1rem;
    }

    .tip-item:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        background: linear-gradient(135deg, #E9ECEF, #DEE2E6);
    }

    .form-section {
        margin-bottom: 2rem;
    }

    .stColumns > div {
        padding: 0 0.75rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    .error-box {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2) !important;
        border: 2px solid #F44336 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        font-weight: 600 !important;
        color: #B71C1C !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(183, 28, 28, 0.1) !important;
        font-size: 1rem !important;
    }

    @media (max-width: 768px) {
        .app-title { font-size: 2.5rem; }
        .form-card { padding: 2rem; }
        .price-amount { font-size: 2.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# Configure page settings
st.set_page_config(
    page_title="Flight Price Predictor ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styles
add_custom_css()

# App header
st.markdown('''
<div class="app-header fade-in-up">
    <h1 class="app-title">✈️ Flight Price Predictor</h1>
    <p class="app-subtitle">Get instant AI-powered flight price predictions with smart recommendations</p>
</div>
''', unsafe_allow_html=True)

# Main columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Flight route section
    st.markdown('''
    <div class="form-card fade-in-up">
        <div class="section-title">🗺️ Flight Route</div>
    </div>
    ''', unsafe_allow_html=True)
    
    route_col1, route_col2 = st.columns(2)
    with route_col1:
        source = st.selectbox("✈️ From", cities, key="source")
    with route_col2:
        available_destinations = [c for c in cities if c != source]
        destination = st.selectbox("🛬 To", available_destinations, key="destination")
    
    cross_region = int(region_map[source] != region_map[destination])
    route_type = "Cross-region flight" if cross_region else "Same-region flight"
    st.markdown(f'<div class="info-box">📍 {route_type}</div>', unsafe_allow_html=True)
    
    # Flight details section
    st.markdown('''
    <div class="form-card fade-in-up">
        <div class="section-title">✈️ Flight Details</div>
    </div>
    ''', unsafe_allow_html=True)
    
    airline_col1, airline_col2 = st.columns(2)
    with airline_col1:
        airline = st.selectbox("🏢 Airline", airlines, key="airline")
    with airline_col2:
        stops = st.selectbox("🔄 Number of Stops", options=[0, 1, 2, 3, 4, 5], index=0, key="stops")

    stop_map = {0: 'zero', 1: 'one', 2: 'two_or_more', 3: 'two_or_more', 4: 'two_or_more', 5: 'two_or_more'}
    stop_label = stop_map.get(stops, 'two_or_more')
    matching_rows = historical_df[
        (historical_df['source_city'] == source) &
        (historical_df['destination_city'] == destination) &
        (historical_df['stops'] == stop_label)
    ]
    suggested_duration = matching_rows['duration'].median() * 60 if not matching_rows.empty else 130
    suggested_duration = int(suggested_duration)
    
    if "duration_mins" not in st.session_state:
        st.session_state.duration_mins = suggested_duration
    elif (st.session_state.get("last_stops") != stops or 
          st.session_state.get("last_src") != source or 
          st.session_state.get("last_dst") != destination):
        st.session_state.duration_mins = suggested_duration
    
    st.session_state["last_stops"] = stops
    st.session_state["last_src"] = source
    st.session_state["last_dst"] = destination

with col2:
    # Schedule & Preferences section
    st.markdown('''
    <div class="form-card fade-in-up">
        <div class="section-title">⏰ Schedule & Preferences</div>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.form("flight_prediction_form", clear_on_submit=False):
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            departure = st.selectbox("🛫 Departure Time", departure_times, key="dep_time")
            duration_mins = st.number_input("⏱️ Flight Duration (minutes)", 
                                          min_value=1,
                                          value=st.session_state.duration_mins, 
                                          key="duration")
        with time_col2:
            arrival = st.selectbox("🛬 Arrival Time", arrival_times, key="arr_time")
            days_left = st.slider("📅 Days Until Departure", 
                                min_value=0, max_value=60, value=7, key="days")
        
        if airline in ['Air_India', 'Vistara']:
            flight_class = st.selectbox("💺 Travel Class", classes, key="class")
        else:
            flight_class = 'Economy'
            st.markdown('<div class="info-box">💺 Economy class (standard for this airline)</div>', 
                       unsafe_allow_html=True)
        
        st.markdown('''
        <div class="time-reference">
            <strong>⏰ Time Reference Guide:</strong><br>
            <strong>Early Morning:</strong> 4:00-8:00 AM • <strong>Morning:</strong> 8:00 AM-12:00 PM<br>
            <strong>Afternoon:</strong> 12:00-5:00 PM • <strong>Evening:</strong> 5:00-8:00 PM<br>
            <strong>Night:</strong> 8:00-11:00 PM • <strong>Late Night:</strong> 11:00 PM-4:00 AM
        </div>
        ''', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict Flight Price")

# Handle form submission
if submitted:
    if duration_mins <= 30:
        st.markdown('<div class="error-box">❌ Error: Flight duration must be more than 30 minutes. Commercial flights typically don\'t operate with such short durations.</div>', unsafe_allow_html=True)
        # Clear any previous predictions
        if "price" in st.session_state:
            del st.session_state["price"]
    else:
        # Calculate features
        red_eye = is_red_eye(departure, arrival)
        is_peak_departure = int(departure in ['Morning', 'Early_Morning'])
        days_duration_interaction = days_left * duration_mins
        stops_per_hour = stops / (duration_mins / 60) if duration_mins > 0 else 0
        booking_type = categorize_booking_type(days_left)
        airline_tier = determine_airline_tier(airline)
        duration_category = categorize_duration(duration_mins)
        
        features = {
            'stops': stops,
            'days_left': days_left,
            'duration_mins': duration_mins,
            'red_eye': red_eye,
            'is_peak_departure': is_peak_departure,
            'cross_region': cross_region,
            'days_duration_interaction': days_duration_interaction,
            'stops_per_hour': stops_per_hour
        }
        
        features_full = {col: 0 for col in model_columns}
        features_full.update(features)
        
        features_full[f'airline_{airline}'] = 1
        features_full[f'source_city_{source}'] = 1
        features_full[f'destination_city_{destination}'] = 1
        features_full[f'departure_time_{departure}'] = 1
        features_full[f'arrival_time_{arrival}'] = 1
        features_full[f'class_{flight_class}'] = 1
        features_full[f'airline_tier_{airline_tier}'] = 1
        features_full[f'booking_type_{booking_type}'] = 1
        features_full[f'duration_category_{duration_category}'] = 1

        input_df = pd.DataFrame([features_full])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Store prediction and tips in session state
        st.session_state.price = model.predict(input_df)[0]
        st.session_state.tips = get_price_tips(features_full)

# Display results if available and duration is valid
if "price" in st.session_state and ("duration" not in st.session_state or st.session_state.duration > 30):
    price = st.session_state.price
    tips = st.session_state.tips

    SGD_TO_INR_RATE = 61.5
    price_in_inr = price * SGD_TO_INR_RATE
    show_inr = st.checkbox("Show price in Indian Rupees (INR)")

    if show_inr:
        st.markdown(f'''
        <div class="results-card">
            <div class="price-display">
                <div class="price-label">🎉 Your Flight Price Prediction</div>
                <div class="price-amount">₹ {price_in_inr:,.2f}</div>
                <div style="color:white; font-size: 1rem; opacity: 0.8;">(SGD {price:,.2f})</div>
            </div>
            <div class="tips-title">💡 Smart Money-Saving Tips</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="results-card">
            <div class="price-display">
                <div class="price-label">🎉 Your Flight Price Prediction</div>
                <div class="price-amount">SGD {price:,.2f}</div>
            </div>
            <div class="tips-title">💡 Smart Money-Saving Tips</div>
        </div>
        ''', unsafe_allow_html=True)

    for tip in tips:
        st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)
