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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hashlib

global pothole_cases_df, pavement_latlon_df, complaint_df # Declare globals here

# Helper function to convert numeric types in DataFrame to native Python types
def _convert_dataframe_numerics_to_native_types(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: float(x) if pd.notna(x) else None)
    return df

# Initialize chat history in session state to ensure it's always available before any access
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set page configuration
st.set_page_config(layout="wide")

st.title("San Antonio Pothole Map & Chatbot")

# Define map center and zoom level
center = [29.358488, -98.626591]
zoom_start = 10

# Initialize the base map globally in session state, only once
# if "m" not in st.session_state:
#     st.session_state.m = folium.Map(location=center, zoom_start=zoom_start)
#     st.session_state.highlight_feature_group = folium.FeatureGroup(name="Highlighted Streets").add_to(st.session_state.m)

# Access the map and feature group from session state
# m = st.session_state.m
# highlight_feature_group = st.session_state.highlight_feature_group

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
            match = re.search(r'place/(-?\d+\.?\d*)[NS]\s*(-?\d+\.?\d*)([EW])', url)
            if match:
                lat = float(match.group(1))
                lon_numeric = float(match.group(2))
                lon_direction = match.group(3)
                
                lon = lon_numeric
                if lon_direction == 'W': # Adjust longitude sign if it's West
                    lon = -abs(lon)
                return lat, lon
            return None, None

        df[['Latitude', 'Longitude']] = df['GoogleMapView'].apply(
            lambda x: pd.Series(extract_lat_lon(x))
        )
        df = df.dropna(subset=['MSAG_Name', 'Latitude', 'Longitude'])
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
        return "I don't have pavement data to identify streets with the worst potholes. Please ensure the 'COSA_Pavement.csv' file is loaded correctly.", None, pd.DataFrame()

    street_pci_avg = pavement_latlon_df.groupby('MSAG_Name')['PCI'].mean()

    if not street_pci_avg.empty:
        street_deterioration_score = 100 - street_pci_avg
        top_worst_streets_data = street_deterioration_score.sort_values(ascending=False).head(10)

        response = "Here are the Top 10 streets with the worst road conditions (most prone to potholes):\n"
        for rank, (street_name, score) in enumerate(top_worst_streets_data.items()):
            response += f"{rank + 1}. {street_name} (Deterioration Score: {score:.2f})\n"

        # Create a bar chart for visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_worst_streets_data.values, y=top_worst_streets_data.index, ax=ax, palette="viridis", hue=top_worst_streets_data.index, legend=False)
        ax.set_title('Top 10 Streets with Worst Road Conditions')
        ax.set_xlabel('Pavement Deterioration Score (100 - PCI)')
        ax.set_ylabel('Street Name')
        plt.tight_layout()

        # Prepare highlight_data_df for map
        highlight_data_df = pavement_latlon_df[pavement_latlon_df['MSAG_Name'].isin(top_worst_streets_data.index)].copy()
        highlight_data_df = highlight_data_df.drop_duplicates(subset=['MSAG_Name'])
        highlight_data_df = highlight_data_df[['MSAG_Name', 'Latitude', 'Longitude']]
        highlight_data_df['color'] = 'darkblue' # Assign darkblue color for worst streets

        return response, fig, highlight_data_df
    else:
        return "No street-level road condition data available to identify worst streets.", None, pd.DataFrame()

def get_top_complaint_locations():
    if complaint_df.empty:
        return "I don't have complaint data to identify top locations. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly.", None, pd.DataFrame()

    df_cosa_pavement_311_complaints = complaint_df.copy()
    df_cosa_pavement_311_complaints.dropna(subset=['MSAG_Name'], inplace=True)

    if not df_cosa_pavement_311_complaints.empty:
        top_10_complaint_locations = df_cosa_pavement_311_complaints['MSAG_Name'].value_counts().head(10)

        response = "Here are the Top 10 most frequently reported complaint locations (streets, all types of complaints):\n"
        for rank, (street_name, count) in enumerate(top_10_complaint_locations.items()):
            response += f"{rank + 1}. {street_name}: {count} total reports\n"

        # Create a bar chart for visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_10_complaint_locations.values, y=top_10_complaint_locations.index, ax=ax, palette="viridis", hue=top_10_complaint_locations.index, legend=False)
        ax.set_title('Top 10 Most Frequently Reported Complaint Locations')
        ax.set_xlabel('Number of Complaints')
        ax.set_ylabel('Street Name')
        plt.tight_layout()

        # Prepare highlight_data_df for map: get lat/lon for top 10 complaint streets
        # Merge with pavement_latlon_df to get coordinates
        highlight_data_df = pd.DataFrame({'MSAG_Name': top_10_complaint_locations.index})
        highlight_data_df = pd.merge(highlight_data_df, pavement_latlon_df[['MSAG_Name', 'Latitude', 'Longitude']],
                                     on='MSAG_Name', how='left')
        highlight_data_df = highlight_data_df.drop_duplicates(subset=['MSAG_Name'])
        highlight_data_df = highlight_data_df.dropna(subset=['Latitude', 'Longitude'])
        highlight_data_df['color'] = 'darkblue' # Assign darkblue color for top complaint locations

        return response, fig, highlight_data_df
    else:
        return "No valid street names found in the complaint data after cleaning.", None, pd.DataFrame()

def get_unresolved_complaints_by_year():
    if complaint_df.empty:
        return "I don't have complaint data to determine unresolved complaints. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly.", None, pd.DataFrame()

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
            return response, None, pd.DataFrame()
        else:
            return "No complaints found to summarize by year.", None, pd.DataFrame()
    else:
        return "No valid complaint data with opened dates found after initial cleaning.", None, pd.DataFrame()

def get_seasonal_pothole_impact():
    if complaint_df.empty:
        return "I don't have complaint data to analyze seasonal impact on potholes. Please ensure the 'COSA_pavement_311.csv' file is loaded correctly.", None, pd.DataFrame()

    pothole_complaints_seasonal = complaint_df.copy()
    pothole_complaints_seasonal['Month'] = pothole_complaints_seasonal['OPENEDDATETIME'].dt.month
    pothole_complaints_seasonal = pothole_complaints_seasonal.dropna(subset=['Month'])

    if not pothole_complaints_seasonal.empty:
        monthly_complaints_potholes = pothole_complaints_seasonal.groupby('Month').size().reset_index(name='Total_Complaints')
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly_complaints_potholes['Month_Name'] = monthly_complaints_potholes['Month'].map(month_names)

        response = "Seasonal Trend of Road-Related Complaints:\n"
        for index, row in monthly_complaints_potholes.iterrows():
            response += f"{row['Month_Name']}: {row['Total_Complaints']} complaints\n"
        response += "\nTypically, increased precipitation and freeze-thaw cycles (large temperature differences) in winter/early spring contribute to more potholes."
        
        # Create a line plot for seasonal trends
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Month_Name', y='Total_Complaints', data=monthly_complaints_potholes, marker='o', ax=ax)
        ax.set_title('Seasonal Trend of Road-Related Complaints')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Complaints')
        plt.tight_layout()

        return response, fig, pd.DataFrame() # No specific highlight data for this plot
    else:
        return "No road-related complaints found for seasonal analysis.", None, pd.DataFrame()

def get_pothole_formation_prediction():
    if pavement_latlon_df.empty or complaint_df.empty:
        return "I need both pavement and complaint data to predict pothole formation. Please ensure 'COSA_Pavement.csv' and 'COSA_pavement_311.csv' are loaded correctly.", None, pd.DataFrame()

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
    latest_install_date['Maintenance_Age_Years'] = latest_install_date['Maintenance_Age_Years'].fillna(latest_install_date['Maintenance_Age_Years'].max() * 2)

    # 4. Merge all relevant dataframes
    pothole_risk_df = pd.merge(pci_by_msag, recent_complaint_counts, on='MSAG_Name', how='outer')
    pothole_risk_df = pd.merge(pothole_risk_df, latest_install_date[['MSAG_Name', 'Maintenance_Age_Years']], on='MSAG_Name', how='outer')

    # Fill NaN values
    pothole_risk_df['Road_Deterioration_Score'] = pothole_risk_df['Road_Deterioration_Score'].fillna(pothole_risk_df['Road_Deterioration_Score'].mean())
    pothole_risk_df['Recent_Complaint_Count'] = pothole_risk_df['Recent_Complaint_Count'].fillna(0)
    pothole_risk_df['Maintenance_Age_Years'] = pothole_risk_df['Maintenance_Age_Years'].fillna(pothole_risk_df['Maintenance_Age_Years'].max())

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
    
    # Create a bar chart for predicted pothole formation risk
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Pothole_Formation_Risk_Score', y='MSAG_Name', data=top_risk_areas, ax=ax, palette="coolwarm", hue='MSAG_Name', legend=False)
    ax.set_title('Top 10 Areas for Pothole Formation Prediction')
    ax.set_xlabel('Pothole Formation Risk Score')
    ax.set_ylabel('Street Name')
    plt.tight_layout()

    # Prepare highlight_data_df for map
    highlight_data_df = pd.merge(top_risk_areas, pavement_latlon_df[['MSAG_Name', 'Latitude', 'Longitude']],
                                 on='MSAG_Name', how='left')
    highlight_data_df = highlight_data_df.drop_duplicates(subset=['MSAG_Name'])
    highlight_data_df = highlight_data_df.dropna(subset=['Latitude', 'Longitude'])

    # Ensure numeric columns are standard Python types for JSON serialization
    for col in ['Latitude', 'Longitude', 'Pothole_Formation_Risk_Score', 'Road_Deterioration_Score', 'Recent_Complaint_Count', 'Maintenance_Age_Years']:
        if col in highlight_data_df.columns:
            if highlight_data_df[col].dtype == 'float64':
                highlight_data_df[col] = highlight_data_df[col].astype(float)
            elif highlight_data_df[col].dtype == 'int64':
                highlight_data_df[col] = highlight_data_df[col].astype(int)

    highlight_data_df['color'] = 'darkblue' # Assign darkblue color for predicted risk
    highlight_data_df['marker_radius'] = 15 # Assign radius 15 for predicted risk

    return response, fig, highlight_data_df

def get_groq_response(prompt):
    prompt_lower = prompt.lower()
    plot_object = None
    highlight_data_df = pd.DataFrame() # Initialize empty DataFrame for map highlighting

    # Check for new, more specific analytical questions
    if "pavement condition for" in prompt_lower or "potholes on" in prompt_lower:
        match = re.search(r'(pavement condition for|potholes on)\s+(.+)', prompt_lower)
        if match:
            street_name = match.group(2).strip()
            response_text = get_pavement_condition_prediction(street_name)
            return response_text, plot_object, highlight_data_df
    if "how many potholes this month" in prompt_lower or "monthly pothole count" in prompt_lower:
        response_text = get_monthly_pothole_count()
        return response_text, plot_object, highlight_data_df
    if "worst potholes" in prompt_lower or "streets with bad roads" in prompt_lower:
        response_text, plot_object, highlight_data_df = get_worst_pothole_streets()
        return response_text, plot_object, highlight_data_df
    if "top complaint locations" in prompt_lower or "most reported streets" in prompt_lower:
        response_text, plot_object, highlight_data_df = get_top_complaint_locations()
        return response_text, plot_object, highlight_data_df
    if "unresolved complaints" in prompt_lower or "open complaints by year" in prompt_lower:
        response_text, plot_object, highlight_data_df = get_unresolved_complaints_by_year()
        return response_text, plot_object, highlight_data_df
    if "seasonal impact on potholes" in prompt_lower or "potholes by season" in prompt_lower:
        response_text, plot_object, highlight_data_df = get_seasonal_pothole_impact()
        return response_text, plot_object, highlight_data_df
    if "predict new potholes" in prompt_lower or "pothole formation prediction" in prompt_lower or "where will new potholes form" in prompt_lower:
        response_text, plot_object, highlight_data_df = get_pothole_formation_prediction()
        return response_text, plot_object, highlight_data_df

    # Keyword-based logic
    keyword_responses = {
        "how many potholes": f"There are {len(pothole_cases_df.index) if not pothole_cases_df.empty else 'no'} potholes recorded in the dataset.",
        "number of potholes": f"The dataset contains {len(pothole_cases_df.index) if not pothole_cases_df.empty else 'no'} potholes.",
        "pavement condition": "Pavement condition ratings were joined with pothole data to analyze correlation.",
        "correlation": "The correlation matrix visualizes relationships among Vibration, Speed, and Acceleration.",
        "heatmap": "The heatmap shows which features are strongly related, such as Vibration vs Speed.",
        "scatter plot": "The scatter plot illustrates the distribution of potholes based on latitude and longitude.",
        "vibration data": "Vibration data, collected by sensors, helps in assessing road roughness and potential pothole formation.",
        "acceleration relate": "Acceleration data can indicate sudden jolts or bumps, which are signs of poor road conditions or potholes.",
        "speed data": "Speed data helps understand how vehicle speed interacts with road conditions, affecting the impact of potholes.",
        "latitude and longitude": "Latitude and longitude provide the precise geographical location of potholes and road segments for mapping.",
        "map or folium": "The Folium map displays potholes and road conditions, allowing for interactive geographical analysis.",
        "time series": "The time series chart visualizes the trend of pothole incidents over time, identifying patterns.",
        "monthly trends": "Monthly trends show fluctuations in pothole reports throughout the year, highlighting peak seasons.",
        "yearly trends": "Yearly trends provide an overview of pothole incidents across different years, indicating long-term changes.",
        "datasets merged": "Various datasets, including 311 service requests, pavement conditions, and sensor data, were merged for comprehensive analysis.",
        "dataset or data columns": "The datasets include columns such as Service Request Type, Latitude, Longitude, Open Date, Close Date, MSAG Name, PCI, etc.",
        "missing values": "Missing values in datasets were handled through imputation or removal, depending on the extent and impact of the missing data."
    }

    # Fallback to Groq API for general questions
    response_text = None
    for keyword, resp in keyword_responses.items():
        if keyword in prompt_lower:
            response_text = resp
            break

    if response_text is None:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
            }
            groq_response = requests.post(GROQ_API_URL, headers=headers, json=data)
            groq_response.raise_for_status() # Raise an exception for HTTP errors
            response_data = groq_response.json()
            response_text = response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Groq API: {e}")
            response_text = "I am currently unable to connect to the Groq AI. Please try again later."
        except KeyError:
            response_text = "I received an unexpected response from the Groq AI. Please try rephrasing your question."

    # Convert numeric types in highlight_data_df to native Python types for JSON serialization
    if not highlight_data_df.empty:
        highlight_data_df = _convert_dataframe_numerics_to_native_types(highlight_data_df)
    
    return response_text, plot_object, highlight_data_df

# Function to plot markers on the map
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

def add_pothole_markers(df, folium_map, feature_group, color_column='color', marker_radius=8):
    for _, row in df.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            marker_color = row[color_column] if color_column in row else 'blue' # Default to blue if no color column
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=int(marker_radius),  # Ensure radius is a standard Python int
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                tooltip=f"{row['MSAG_Name']}: {row.get('ComplaintCount', 'N/A')} Complaints",
            ).add_to(feature_group)
    return feature_group # Return the feature group

# --- Load additional datasets for chatbot analysis ---
# Define paths relative to the integrated.py file
data_folder_path = "Data"

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
    
    # Show only the chatbot if map data is missing
    st.markdown("### Chatbot")
    messages = st.container(height=300)

    # Display chat history (already initialized at the top)
    for message in st.session_state.messages:
        with messages.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about potholes or road conditions"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with messages.chat_message("user"):
            st.write(prompt)

        # Get response from Groq AI
        response_text, plot_object, highlight_data_df = get_groq_response(prompt)
        
        # Convert numeric types in highlight_data_df to native Python types for JSON serialization
        if not highlight_data_df.empty:
            highlight_data_df = _convert_dataframe_numerics_to_native_types(highlight_data_df)

        # Add assistant response to chat history, storing plot and highlight data
        message_entry = {
            "role": "assistant",
            "content": response_text,
            "vis_data": {
                "plot_object": plot_object,
                "highlight_data_df": highlight_data_df.to_dict('records') if not highlight_data_df.empty else []
            },
            "prompt": prompt # Keep prompt for re-running visualizations if needed
        }
        st.session_state.messages.append(message_entry)
        with messages.chat_message("assistant"):
            st.write(response_text)
            if plot_object is not None:
                st.pyplot(plot_object)
                plt.close(plot_object) # Ensure plot is closed after display for new responses
    st.stop() # Stop execution if data loading failed and only chatbot is available

# ---------- Layout ----------
col1, col2 = st.columns([3, 1])

# -- Main Map Area --
with col1:
    st.markdown("### Pothole Map")
    
    # Use st.empty() to control the map's rendering lifecycle explicitly
    map_placeholder = st.empty()

    # Create a new map object and its layers within the placeholder
    with map_placeholder.container():
        current_map = folium.Map(location=center, zoom_start=zoom_start)

        current_highlight_feature_group = folium.FeatureGroup(name="Highlighted Streets")

        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            
            if latest_message.get("vis_data") and latest_message["vis_data"].get("highlight_data_df"):
                latest_highlight_df = pd.DataFrame(latest_message["vis_data"]["highlight_data_df"])
                
                if not latest_highlight_df.empty:
                    # If not, it will use the default value from the function definition
                    add_pothole_markers(
                        latest_highlight_df,
                        current_map,
                        current_highlight_feature_group,
                        color_column='color',
                        marker_radius=latest_highlight_df['marker_radius'].iloc[0] if 'marker_radius' in latest_highlight_df.columns else 8
                    )
            
            current_highlight_feature_group.add_to(current_map)

        # Use a simple key for now; if it still doesn't update, we can make it dynamic based on content
        map_key = f"folium_map_initial"
        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            # Create a copy of vis_data and remove plot_object before JSON serialization
            vis_data_for_key = latest_message.get("vis_data", {}).copy()
            if "plot_object" in vis_data_for_key:
                del vis_data_for_key["plot_object"]
            latest_vis_data_json = json.dumps(vis_data_for_key, sort_keys=True)
            latest_vis_data_hash = hashlib.md5(latest_vis_data_json.encode()).hexdigest()
            map_key = f"folium_map_{len(st.session_state.messages)}_{latest_vis_data_hash}"

        st_data = st_folium(current_map, width=900, height=500, key=map_key)

# -- Chatbot Sidebar --
with col2:
    st.markdown("### Chatbot")
    messages = st.container(height=300)

    # Display chat history with permanent visualizations
    for i, message in enumerate(st.session_state.messages):
        with messages.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("vis_data"):
                plot_obj = message["vis_data"].get("plot_object")
                highlight_df_dict = message["vis_data"].get("highlight_data_df")
                highlight_df = pd.DataFrame(highlight_df_dict) if highlight_df_dict else pd.DataFrame()

                if plot_obj is not None:
                    st.pyplot(plot_obj)
                    plt.close(plot_obj)

                # Map highlight logic is now before main map rendering
                # No st_folium call here for historical maps in chatbot

    # Chat input and response handling
    if prompt := st.chat_input("Ask about potholes or road conditions"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with messages.chat_message("user"):
            st.write(prompt)

        # Get response from Groq AI
        response_text, plot_object, highlight_data_df = get_groq_response(prompt)
        
        # Convert numeric types in highlight_data_df to native Python types for JSON serialization
        if not highlight_data_df.empty:
            highlight_data_df = _convert_dataframe_numerics_to_native_types(highlight_data_df)

        # Add assistant response to chat history, storing plot and highlight data
        message_entry = {
            "role": "assistant",
            "content": response_text,
            "vis_data": {
                "plot_object": plot_object,
                "highlight_data_df": highlight_data_df.to_dict('records') if not highlight_data_df.empty else []
            },
            "prompt": prompt # Keep prompt for re-running visualizations if needed
        }
        st.session_state.messages.append(message_entry)
        with messages.chat_message("assistant"):
            st.write(response_text)
            if plot_object is not None:
                st.pyplot(plot_object)
                plt.close(plot_object) # Ensure plot is closed after display for new responses

            # Map highlight logic is now handled before main map rendering
            # No st_folium call here for new responses in chatbot

# You can try these questions in your Streamlit application!
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
