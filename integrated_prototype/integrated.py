import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import requests
import json
import os
import re
from datetime import datetime, timedelta

global pothole_cases_df, pavement_latlon_df, complaint_df # Declare globals here

# ---------- App Configuration ----------
st.set_page_config(layout="wide")
st.title('🗺️ Integrated Chat with Map')

# ---------- Groq AI Configuration ----------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize global DataFrames
pothole_cases_df = pd.DataFrame()
pavement_latlon_df = pd.DataFrame()
complaint_df = pd.DataFrame()

# --- Analysis Functions (from Visualization.ipynb) ---

@st.cache_data
def load_pothole_cases_data(path):
    try:
        df = pd.read_csv(path)
        df['OpenDate'] = pd.to_datetime(df['OpenDate'])
        # st.success(f"Successfully loaded {os.path.basename(path)}")
        return df
    except Exception as e:
        st.warning(f"File not found or error loading {os.path.basename(path)}: {e}. Some chatbot features may be limited.")
        return pd.DataFrame()

@st.cache_data
def load_pavement_data(path):
    try:
        df = pd.read_csv(path)
        # Extract latitude and longitude from 'GoogleMapView' column
        def extract_lat_lon(url):
            if pd.isna(url) or url == 'Not Available':
                return None, None
            match = re.search(r'place/(-?\d+\.?\d*)[N|S]\s*(-?\d+\.?\d*)[E|W]', url)
            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if 'W' in match.group(2): # Adjust longitude sign if it's West
                    lon = -abs(lon)
                return lat, lon
            return None, None

        df[['Latitude', 'Longitude']] = df['GoogleMapView'].apply(
            lambda x: pd.Series(extract_lat_lon(x))
        )
        df.dropna(subset=['MSAG_Name', 'Latitude', 'Longitude'], inplace=True)
        # st.success(f"Successfully loaded and cleaned {os.path.basename(path)}")
        return df
    except Exception as e:
        st.warning(f"File not found or error loading {os.path.basename(path)}: {e}. Some chatbot features may be limited.")
        return pd.DataFrame()

@st.cache_data
def load_complaint_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        df['OPENEDDATETIME'] = pd.to_datetime(df['OPENEDDATETIME'], errors='coerce')
        df['InstallDate'] = pd.to_datetime(df['InstallDate'], errors='coerce')
        # st.success(f"Successfully loaded and cleaned {os.path.basename(path)}")
        return df
    except Exception as e:
        st.warning(f"File not found or error loading {os.path.basename(path)}: {e}. Some chatbot features may be limited.")
        return pd.DataFrame()

def get_pavement_condition_prediction(street_name):
    if pavement_latlon_df.empty:
        return "I don't have pavement condition data to answer that question. Please ensure the 'COSA_Pavement.csv' file is loaded correctly."

    target_street_data = pavement_latlon_df[pavement_latlon_df['MSAG_Name'].str.contains(street_name, case=False, na=False)].copy()

    if not target_street_data.empty:
        avg_pci = target_street_data['PCI'].mean()
        if avg_pci < 50:
            prediction = "High likelihood of facing potholes due to generally poor pavement conditions."
        elif avg_pci < 70:
            prediction = "Moderate likelihood of facing potholes due to fair pavement conditions."
        else:
            prediction = "Low likelihood of facing potholes due to generally good pavement conditions."
        return f"For {street_name}, the average Pavement Condition Index (PCI) is {avg_pci:.2f}. Prediction: {prediction}"
    else:
        return f"No pavement data found for the street: {street_name}. Please check the street name or expand the search area."

def get_monthly_pothole_count():
    if pothole_cases_df.empty:
        return "I don't have monthly pothole case data to answer that question. Please ensure the '311_Pothole_Cases_18_24.csv' file is loaded correctly."

    pothole_cases_df['YearMonth'] = pothole_cases_df['OpenDate'].dt.to_period('M')
    monthly_potholes = pothole_cases_df.groupby('YearMonth')['cases'].sum().sort_index()

    if not monthly_potholes.empty:
        latest_month_period = monthly_potholes.index.max()
        potholes_this_month = monthly_potholes.loc[latest_month_period]
        latest_month_str = latest_month_period.strftime('%B %Y')
        return f"In {latest_month_str}, a total of {potholes_this_month} potholes were reported."
    else:
        return "No monthly pothole cases data available to show trends."

def get_worst_pothole_streets():
    if pavement_latlon_df.empty:
        return "I don't have pavement data to identify streets with the worst potholes. Please ensure the 'COSA_Pavement.csv' file is loaded correctly."

    street_pci_avg = pavement_latlon_df.groupby('MSAG_Name')['PCI'].mean()

    if not street_pci_avg.empty:
        street_deterioration_score = 100 - street_pci_avg
        top_worst_streets_data = street_deterioration_score.sort_values(ascending=False).head(10)

        response = "Here are the Top 10 streets with the worst road conditions (most prone to potholes):\n"
        for rank, (street_name, score) in enumerate(top_worst_streets_data.items()):
            response += f"{rank + 1}. {street_name} (Deterioration Score: {score:.2f})\n"
        return response
    else:
        return "No street-level road condition data available to identify worst streets."

def get_top_complaint_locations():
    if complaint_df.empty:
        return "I don't have complaint data to identify top locations. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly."

    df_cosa_pavement_311_complaints = complaint_df.copy()
    df_cosa_pavement_311_complaints.dropna(subset=['MSAG_Name'], inplace=True)

    if not df_cosa_pavement_311_complaints.empty:
        top_10_complaint_locations = df_cosa_pavement_311_complaints['MSAG_Name'].value_counts().head(10)

        response = "Here are the Top 10 most frequently reported complaint locations (streets, all types of complaints):\n"
        for rank, (street_name, count) in enumerate(top_10_complaint_locations.items()):
            response += f"{rank + 1}. {street_name}: {count} total reports\n"
        return response
    else:
        return "No valid street names found in the complaint data after cleaning."

def get_unresolved_complaints_by_year():
    if complaint_df.empty:
        return "I don't have complaint data to determine unresolved complaints. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly."

    df_complaints_yearly = complaint_df.copy()
    df_complaints_yearly['OPENEDDATETIME'] = pd.to_datetime(df_complaints_yearly['OPENEDDATETIME'], errors='coerce')
    df_complaints_yearly.dropna(subset=['OPENEDDATETIME'], inplace=True)

    if not df_complaints_yearly.empty:
        df_complaints_yearly['OpenedYear'] = df_complaints_yearly['OPENEDDATETIME'].dt.year
        df_complaints_yearly['IsUnresolved'] = df_complaints_yearly['CLOSEDDATETIME'].isna()

        yearly_status = df_complaints_yearly.groupby('OpenedYear').agg(
            TotalComplaints=('OPENEDDATETIME', 'count'),
            UnresolvedComplaints=('IsUnresolved', 'sum')
        ).reset_index()
        yearly_status['UnresolvedComplaints'] = yearly_status['UnresolvedComplaints'].astype(int)

        if not yearly_status.empty:
            response = "Complaint Status by Year:\n"
            for index, row in yearly_status.iterrows():
                if row['TotalComplaints'] > 0:
                    percent_unresolved = (row['UnresolvedComplaints'] / row['TotalComplaints']) * 100
                    response += f"Year {int(row['OpenedYear'])}: Total = {int(row['TotalComplaints'])}, Unresolved = {int(row['UnresolvedComplaints'])} ({percent_unresolved:.2f}%)\n"
                else:
                    response += f"Year {int(row['OpenedYear'])}: No complaints reported.\n"
            return response
        else:
            return "No complaints found to summarize by year."
    else:
        return "No valid complaint data with opened dates found after initial cleaning."

def get_seasonal_pothole_impact():
    if complaint_df.empty:
        return "I don't have complaint data to analyze seasonal impact on potholes. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly."

    pothole_complaints_seasonal = complaint_df.copy()
    pothole_complaints_seasonal['Month'] = pothole_complaints_seasonal['OPENEDDATETIME'].dt.month
    pothole_complaints_seasonal.dropna(subset=['Month'], inplace=True)

    if not pothole_complaints_seasonal.empty:
        monthly_complaints_potholes = pothole_complaints_seasonal.groupby('Month').size().reset_index(name='Total_Complaints')
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly_complaints_potholes['Month_Name'] = monthly_complaints_potholes['Month'].map(month_names)

        response = "Seasonal Trend of Road-Related Complaints:\n"
        for index, row in monthly_complaints_potholes.iterrows():
            response += f"{row['Month_Name']}: {row['Total_Complaints']} complaints\n"
        response += "\nTypically, increased precipitation and freeze-thaw cycles (large temperature differences) in winter/early spring contribute to more potholes."
        return response
    else:
        return "No road-related complaints found for seasonal analysis."

def get_pothole_formation_prediction():
    if pavement_latlon_df.empty or complaint_df.empty:
        return "I need both pavement and complaint data to predict pothole formation. Please ensure 'COSA_Pavement.csv' and 'COSA_pavement_311.csv' are loaded correctly."

    # 1. Calculate Average PCI and Road Deterioration Score per MSAG_Name
    pci_by_msag = pavement_latlon_df.groupby('MSAG_Name')['PCI'].mean().reset_index()
    pci_by_msag['Road_Deterioration_Score'] = 100 - pci_by_msag['PCI']

    # 2. Calculate Recent Complaint Count per MSAG_Name
    current_year = datetime.now().year
    recent_complaints_period = complaint_df[
        (complaint_df['OPENEDDATETIME'].dt.year >= current_year - 2) &
        (complaint_df['OPENEDDATETIME'].dt.year < current_year) # Exclude current incomplete year
    ].copy()
    recent_complaint_counts = recent_complaints_period['MSAG_Name'].value_counts().reset_index()
    recent_complaint_counts.columns = ['MSAG_Name', 'Recent_Complaint_Count']

    # 3. Calculate Maintenance Age per MSAG_Name
    latest_install_date = complaint_df.groupby('MSAG_Name')['InstallDate'].max().reset_index()
    latest_data_date = complaint_df['OPENEDDATETIME'].max()
    if pd.isna(latest_data_date):
        latest_data_date = datetime.now()
    latest_install_date['Maintenance_Age_Years'] = (latest_data_date - latest_install_date['InstallDate']).dt.days / 365.25
    latest_install_date['Maintenance_Age_Years'].fillna(latest_install_date['Maintenance_Age_Years'].max() * 2, inplace=True)

    # 4. Merge all relevant dataframes
    pothole_risk_df = pd.merge(pci_by_msag, recent_complaint_counts, on='MSAG_Name', how='outer')
    pothole_risk_df = pd.merge(pothole_risk_df, latest_install_date[['MSAG_Name', 'Maintenance_Age_Years']], on='MSAG_Name', how='outer')

    # Fill NaN values
    pothole_risk_df['Road_Deterioration_Score'].fillna(pothole_risk_df['Road_Deterioration_Score'].mean(), inplace=True)
    pothole_risk_df['Recent_Complaint_Count'].fillna(0, inplace=True)
    pothole_risk_df['Maintenance_Age_Years'].fillna(pothole_risk_df['Maintenance_Age_Years'].max(), inplace=True)

    # 5. Create a composite Pothole Formation Risk Score (normalize and sum)
    for col in ['Road_Deterioration_Score', 'Recent_Complaint_Count', 'Maintenance_Age_Years']:
        min_val = pothole_risk_df[col].min()
        max_val = pothole_risk_df[col].max()
        if (max_val - min_val) != 0:
            pothole_risk_df[f'{col}_Scaled'] = (pothole_risk_df[col] - min_val) / (max_val - min_val)
        else:
            pothole_risk_df[f'{col}_Scaled'] = 0.5 # Assign a neutral value if all are the same

    pothole_risk_df['Pothole_Formation_Risk_Score'] = \
        pothole_risk_df['Road_Deterioration_Score_Scaled'] * 0.5 + \
        pothole_risk_df['Recent_Complaint_Count_Scaled'] * 0.3 + \
        pothole_risk_df['Maintenance_Age_Years_Scaled'] * 0.2 

    pothole_risk_df.sort_values(by='Pothole_Formation_Risk_Score', ascending=False, inplace=True)

    top_risk_areas = pothole_risk_df.head(10)
    
    response = "Predicted Top 10 Areas for New Pothole Formation in the next 2 years (Higher Score = Higher Risk):\n"
    for index, row in top_risk_areas.iterrows():
        response += f"{index + 1}. {row['MSAG_Name']}: Risk Score = {row['Pothole_Formation_Risk_Score']:.2f} (Deterioration: {row['Road_Deterioration_Score']:.2f}, Recent Complaints: {int(row['Recent_Complaint_Count'])}, Maint. Age: {row['Maintenance_Age_Years']:.1f} yrs)\n"
    return response

def get_groq_response(prompt):
    prompt_lower = prompt.lower()

    # Check for new, more specific analytical questions
    if "pavement condition for" in prompt_lower or "potholes on" in prompt_lower:
        match = re.search(r'(pavement condition for|potholes on)\s+(.+)', prompt_lower)
        if match:
            street_name = match.group(2).strip()
            return get_pavement_condition_prediction(street_name)
    if "how many potholes this month" in prompt_lower or "monthly pothole count" in prompt_lower:
        return get_monthly_pothole_count()
    if "worst potholes" in prompt_lower or "streets with bad roads" in prompt_lower:
        return get_worst_pothole_streets()
    if "top complaint locations" in prompt_lower or "most reported streets" in prompt_lower:
        return get_top_complaint_locations()
    if "unresolved complaints" in prompt_lower or "open complaints by year" in prompt_lower:
        return get_unresolved_complaints_by_year()
    if "seasonal impact on potholes" in prompt_lower or "potholes by season" in prompt_lower:
        return get_seasonal_pothole_impact()
    if "predict new potholes" in prompt_lower or "pothole formation prediction" in prompt_lower or "where will new potholes form" in prompt_lower:
        return get_pothole_formation_prediction()

    # Keyword-based logic based on Visualization.ipynb insights
    keyword_responses = {
        "how many potholes": f"There are {len(my_subset)} potholes recorded in the dataset.",
        "number of potholes": f"The dataset contains {len(my_subset)} potholes.",
        "pavement condition": "Pavement condition ratings were joined with pothole data to analyze correlation.",
        "correlation": "The correlation matrix visualizes relationships among Vibration, Speed, and Acceleration.",
        "heatmap": "The heatmap shows which features are strongly related, such as Vibration vs Speed.",
        "scatter": "The scatter plot visualizes Speed vs Acceleration and uses color to represent Vibration.",
        "vibration": "Vibration data can be used to detect uneven road surfaces and possible potholes.",
        "acceleration": "Acceleration measures help identify rapid speed changes, useful for rough patch detection.",
        "speed": "Speed is a key feature used to assess road behavior and potential hazards.",
        "latitude" or "longitude": "GPS coordinates are used to map pothole locations on a Folium map.",
        "map" or "folium": "The Folium map is used to visualize pothole data with markers based on GPS coordinates.",
        "time series": "The notebook shows monthly and yearly trends of pothole cases using line plots.",
        "monthly trend": "Monthly pothole reports were visualized using time series plots to detect peaks.",
        "yearly trend": "The data was grouped by year to identify whether potholes are increasing or decreasing.",
        "merge": "Two datasets—pothole reports and pavement ratings—were merged on location info.",
        "dataset" or "data columns": "The pothole dataset includes date, latitude, longitude, severity, and more.",
        "missing values": "Missing values in location and date fields were handled during preprocessing."
    }

    # Respond using keyword matches
    for keyword, response in keyword_responses.items():
        if keyword in prompt_lower:
            return response

    # Fallback to Groq API for other prompts
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- Load and Plot Map ----------
try:
    my_subset = gpd.read_file("integrated_prototype/potholes_data.gpkg", layer="potholes")
    small_df = my_subset.head(10)

    def plot_from_df(df, folium_map):
        """
        Function to plot markers on the map
        Args:
            df (DataFrame): Data containing coordinates
            folium_map (Map): Folium map object
        Returns:
            Map: Updated map with markers
        """
        for i, row in df.iterrows():
            folium.Marker(
                location=[row.Latitude, row.Longitude],
                tooltip=f"Location {i}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(folium_map)
        return folium_map

    center = [29.358488, -98.626591]
    zoom_start = 10
    m = plot_from_df(small_df, folium.Map(location=center, zoom_start=zoom_start))

    # --- Load additional datasets for chatbot analysis ---
    # Define paths relative to the integrated.py file
    data_folder_path = os.path.join(os.path.dirname(__file__), "..", "Data")

    pothole_cases_path = os.path.join(data_folder_path, "311_Pothole_Cases_18_24.csv")
    pavement_path = os.path.join(data_folder_path, "COSA_Pavement.csv")
    complaint_full_path = os.path.join(data_folder_path, "COSA_pavement_311.csv")

    try:
        pothole_cases_df = load_pothole_cases_data(pothole_cases_path)
        pavement_latlon_df = load_pavement_data(pavement_path)
        complaint_df = load_complaint_data(complaint_full_path)

    except Exception as e:
        st.error(f"An error occurred while loading additional data: {e}")
        st.info("Some chatbot features related to detailed data analysis may be unavailable.")

    # ---------- Layout ----------
    col1, col2 = st.columns([3, 1])

    # -- Main Map Area --
    with col1:
        st.markdown("### Pothole Map")
        st_data = st_folium(m, width=900, height=500)

    # -- Chatbot Sidebar --
    with col2:
        st.markdown("### Chatbot")
        messages = st.container(height=300)
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with messages.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input and response handling
        if prompt := st.chat_input("Ask about potholes or road conditions"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with messages.chat_message("user"):
                st.write(prompt)

            # Get response from Groq AI
            response = get_groq_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with messages.chat_message("assistant"):
                st.write(response)

except FileNotFoundError:
    st.error("Error: potholes_data.gpkg file not found. Please ensure the file is in the correct location.")
    st.info("The map visualization is currently unavailable. However, you can still use the chatbot feature.")
    
    # Show only the chatbot if map data is missing
    st.markdown("### Chatbot")
    messages = st.container(height=300)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with messages.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about potholes or road conditions"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with messages.chat_message("user"):
            st.write(prompt)

        # Get response from Groq AI
        response = get_groq_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with messages.chat_message("assistant"):
            st.write(response)



# General Pothole & Map Information:
# "How many potholes?"
# "Number of potholes"
# "What is pavement condition?"
# "Tell me about correlation."
# "What does the heatmap show?"
# "Explain the scatter plot."
# "What is vibration data used for?"
# "How does acceleration relate to road conditions?"
# "What about speed data?"
# "How are latitude and longitude used?"
# "What about the map or Folium?"
# "What does the time series show?"
# "Tell me about monthly trends."
# "Tell me about yearly trends."
# "How are datasets merged?"
# "What are the dataset or data columns?"
# "How are missing values handled?"
# Specific Analytical Questions (New Capabilities):
# Pavement Condition by Street:
# "What is the pavement condition for [street name]?"
# "Are there potholes on [street name]?"
# Monthly Pothole Reports:
# "How many potholes this month?"
# "What's the monthly pothole count?"
# Worst Pothole Streets:
# "Display streets with the worst potholes."
# "Show me streets with bad roads."
# Top Complaint Locations:
# "What are the top complaint locations?"
# "Which streets are most reported?"
# Unresolved Complaints:
# "How many unresolved complaints are there?"
# "Show open complaints by year."
# Seasonal Pothole Impact:
# "What is the seasonal impact on potholes?"
# "Potholes by season?"
# Pothole Formation Prediction:
# "Predict new potholes."
# "What's the pothole formation prediction?"
# "Where will new potholes form?"
# You can try these questions in your Streamlit application!
